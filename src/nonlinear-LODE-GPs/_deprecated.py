class LODEGP_Deprecated(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, A):
        super(LODEGP_Deprecated, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.model_parameters = torch.nn.ParameterDict()
        #self.num_tasks = num_tasks

        D, U, V = A.smith_form()
        print(f"D:{D}")
        print(f"V:{V}")
        x, a, b = var("x, a, b")
        V_temp = [list(b) for b in V.rows()]
        #print(V_temp)
        V = sage_eval(f"matrix({str(V_temp)})", locals={"x":x, "a":a, "b":b})
        Vt = V.transpose()
        kernel_matrix, self.kernel_translation_dict, parameter_dict = create_kernel_matrix_from_diagonal(D)
        self.ode_count = A.nrows()
        self.kernelsize = len(kernel_matrix)
        self.model_parameters.update(parameter_dict)
        #print(self.model_parameters)
        var(["x", "dx1", "dx2"] + ["t1", "t2"] + [f"LODEGP_kernel_{i}" for i in range(len(kernel_matrix[Integer(0)]))])
        k = matrix(Integer(len(kernel_matrix)), Integer(len(kernel_matrix)), kernel_matrix)
        V = V.substitute(x=dx1)
        Vt = Vt.substitute(x=dx2)

        #train_x = self._slice_input(train_x)

        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }
        self.V = V
        self.matrix_multiplication = matrix(k.base_ring(), len(k[0]), len(k[0]), (V*k*Vt))
        self.diffed_kernel = differentiate_kernel_matrix(k, V, Vt, self.kernel_translation_dict)
        self.sum_diff_replaced = replace_sum_and_diff(self.diffed_kernel)
        self.covar_description = translate_kernel_matrix_to_gpytorch_kernel(self.sum_diff_replaced, self.model_parameters, common_terms=self.common_terms)
        self.covar_module = LODE_Kernel(self.covar_description, self.model_parameters)


    def __str__(self, substituted=False):
        if substituted:
            return pprint.pformat(str(self.sum_diff_replaced), indent=self.kernelsize)
        else:
            return pprint.pformat(str(self.diffed_kernel), indent=self.kernelsize)

    def __latexify_kernel__(self, substituted=False):
        if substituted:
            return pprint.pformat(latex(self.sum_diff_replaced), indent=self.kernelsize)
        else:
            return pprint.pformat(latex(self.diffed_kernel), indent=self.kernelsize)

    def __pretty_print_kernel__(self, substituted=False):
        if substituted:
            return pprint.pformat(pretty_print(self.matrix_multiplication), indent=self.kernelsize)
        else:
            pretty_print(self.matrix_multiplication)
            print(str(self.kernel_translation_dict))

    def _slice_input(self, X):
            r"""
            Slices :math:`X` according to ``self.active_dims``. If ``X`` is 1D then returns
            a 2D tensor with shape :math:`N \times 1`.
            :param torch.Tensor X: A 1D or 2D input tensor.
            :returns: a 2D slice of :math:`X`
            :rtype: torch.Tensor
            """
            if X.dim() == 2:
                #return X[:, self.active_dims]
                return X[:, 0]
            elif X.dim() == 1:
                return X.unsqueeze(1)
            else:
                raise ValueError("Input X must be either 1 or 2 dimensional.")

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        #print(torch.linalg.eigvalsh(covar_x.evaluate()))
        #covar_x = covar_x.flatten()
        #print(list(torch.linalg.eigh(covar_x)[0])[::-1])
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 
    

class Param_LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, A, x0):
        super(Param_LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        
        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }
        
        self.covar_module = Param_LODE_Kernel(A, x0, self.common_terms)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 
    
class Changepoint_LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, A_list, changepoints:List[float]):
        super(Changepoint_LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        
        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }

        covar_modules = []
        for A in A_list:
            covar_modules.append(LODE_Kernel(A, self.common_terms))
        

        self.covar_module = Drastic_changepoint_Kernel(covar_modules, changepoints, num_tasks)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 
    

class Drastic_changepoint_Kernel(Kernel):
    '''
    This Kernel implements the drastic change in covariance as described by 
    R. Garnett, M. A. Osborne, and S. J. Roberts, “Sequential Bayesian prediction in the presence of changepoints,” 
    in Proceedings of the 26th Annual International Conference on Machine Learning, Montreal Quebec Canada: ACM, Jun. 2009, pp. 345-352. doi: 10.1145/1553374.1553418.

    '''
    def __init__(self, covar_modules:List[Kernel], changepoints:List[float], num_tasks:int, active_dims=None):
        super(Drastic_changepoint_Kernel, self).__init__(active_dims=active_dims)

        if len(covar_modules) != len(changepoints) + 1:
            raise ValueError("The number of changepoints must be one less than the number of covar_modules")
        
        self.covar_modules = covar_modules
        self.changepoints = changepoints
        self.num_tasks = num_tasks 

    def forward(self, x1, x2, diag=False, **params):
        #K = torch.zeros(x1.size(0) * self.num_tasks, x2.size(0)* self.num_tasks)

        K = torch.zeros(self.num_tasks, x1.size(0), self.num_tasks, x2.size(0))

        for i in range(x1.size(0)):
            for j in range(x2.size(0)):
                idx_1 = max([k for k in range(len(self.changepoints)) if self.changepoints[k] < x1[i]], default=-1) + 1
                idx_2 = max([k for k in range(len(self.changepoints)) if self.changepoints[k] < x2[j]], default=-1) + 1

                if idx_1 == idx_2:
                    K[:, i, :, j] = self.covar_modules[idx_1](x1[i], x2[j], diag=False, **params).evaluate()
                    #K[i*self.num_tasks:(i+1)*self.num_tasks, j*self.num_tasks:(j+1)*self.num_tasks] = self.covar_modules[idx_1](x1[i], x2[j], diag=diag, **params)
                    
                    #if idx_1 ==1:
                    #    print(f"idx_1: {idx_1}, idx_2: {idx_2}")
                    
                else:
                    K[:, i, :, j] = torch.zeros(self.num_tasks, self.num_tasks)
                    #print('this should not happen')
                    #K[i*self.num_tasks:(i+1)*self.num_tasks, j*self.num_tasks:(j+1)*self.num_tasks] = torch.zeros(self.num_tasks, self.num_tasks)
        
        K = K.view(x1.size(0)*self.num_tasks, x2.size(0)*self.num_tasks)
        return K
    
    def num_outputs_per_input(self, x1, x2):
            """
            Given `n` data points `x1` and `m` datapoints `x2`, this multitask
            kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
            """
            return self.num_tasks


class Param_LODE_Kernel(Kernel):
        def __init__(self, A, x0, common_terms, active_dims=None):
            super(Param_LODE_Kernel, self).__init__(active_dims=active_dims)

            self.model_parameters = torch.nn.ParameterDict()
            #self.num_tasks = num_tasks

            D, U, V = A.smith_form()
            print(f"D:{D}")
            print(f"V:{V}")
            x, a, b = var("x, a, b")
            V_temp = [list(b) for b in V.rows()]
            #print(V_temp)
            V = sage_eval(f"matrix({str(V_temp)})", locals={"x":x, "a":a, "b":b})
            Vt = V.transpose()
            kernel_matrix, self.kernel_translation_dict, parameter_dict = create_kernel_matrix_from_diagonal(D)

            # add equilibrium vars to the param dict
            parameter_dict["x1"] = torch.nn.Parameter(torch.tensor(x0[0]), requires_grad=False)
            parameter_dict["x2"] = torch.nn.Parameter(torch.tensor(x0[1]), requires_grad=False)
            parameter_dict["x3"] = torch.nn.Parameter(torch.tensor(x0[2]), requires_grad=False)
            parameter_dict["u"] = torch.nn.Parameter(torch.tensor(x0[3]), requires_grad=False)

            self.ode_count = A.nrows()
            self.kernelsize = len(kernel_matrix)
            self.model_parameters.update(parameter_dict)
            #print(self.model_parameters)
            x, dx1, dx2, t1, t2, *_ = var(["x", "dx1", "dx2"] + ["t1", "t2"] + [f"LODEGP_kernel_{i}" for i in range(len(kernel_matrix[Integer(0)]))])
            k = matrix(Integer(len(kernel_matrix)), Integer(len(kernel_matrix)), kernel_matrix)
            V = V.substitute(x=dx1)
            Vt = Vt.substitute(x=dx2)

            self.V = V
            self.matrix_multiplication = matrix(k.base_ring(), len(k[0]), len(k[0]), (V*k*Vt))
            self.diffed_kernel = differentiate_kernel_matrix(k, V, Vt, self.kernel_translation_dict)
            self.sum_diff_replaced = replace_sum_and_diff(self.diffed_kernel)
            self.covar_description = translate_kernel_matrix_to_gpytorch_kernel(self.sum_diff_replaced, self.model_parameters, common_terms=common_terms)

            self.num_tasks = len(self.covar_description)

        def num_outputs_per_input(self, x1, x2):
            """
            Given `n` data points `x1` and `m` datapoints `x2`, this multitask
            kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
            """
            return self.num_tasks

        #def forward(self, X, Z=None, common_terms=None):
        def forward(self, x1, x2, diag=False, **params):
            common_terms = params["common_terms"]
            model_parameters = self.model_parameters
            if not x2 is None:
                common_terms["t_diff"] = x1-x2.t()
                common_terms["t_sum"] = x1+x2.t()
                common_terms["t_ones"] = torch.ones_like(x1+x2.t())
                common_terms["t_zeroes"] = torch.zeros_like(x1+x2.t())
            K_list = list() 
            for rownum, row in enumerate(self.covar_description):
                for cell in row:
                    K_list.append(eval(cell))
            kernel_count = len(self.covar_description)
            # from https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/6
            #if K_list[0].ndim == 1:
            #    K_list = [kk.unsqueeze(1) for kk in K_list]
            K = einops.rearrange(K_list, '(t1 t2) h w -> (h t1) (w t2)', t1=kernel_count, t2=kernel_count)  

            return K 
        
        def __str__(self, substituted=False):
            if substituted:
                return pprint.pformat(str(self.sum_diff_replaced), indent=self.kernelsize)
            else:
                return pprint.pformat(str(self.diffed_kernel), indent=self.kernelsize)

        def __latexify_kernel__(self, substituted=False):
            if substituted:
                return pprint.pformat(latex(self.sum_diff_replaced), indent=self.kernelsize)
            else:
                return pprint.pformat(latex(self.diffed_kernel), indent=self.kernelsize)

        def __pretty_print_kernel__(self, substituted=False):
            if substituted:
                return pprint.pformat(pretty_print(self.matrix_multiplication), indent=self.kernelsize)
            else:
                pretty_print(self.matrix_multiplication)
                print(str(self.kernel_translation_dict))
