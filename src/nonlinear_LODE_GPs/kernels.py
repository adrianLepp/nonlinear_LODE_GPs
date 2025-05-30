import copy as cp
from einops import rearrange
import re
import itertools
from itertools import zip_longest
from torch.distributions import constraints
import torch
from functools import reduce
import gpytorch
# from gpytorch.lazy import *
# from gpytorch.lazy.non_lazy_tensor import  lazify
from gpytorch.kernels.kernel import Kernel
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from sage.arith.misc import factorial
import numpy as np
import pdb
from gpytorch.constraints import Positive
from linear_operator import to_linear_operator
import random
import einops
import pprint
from linear_operator.operators import DiagLinearOperator, ConstantDiagLinearOperator
from typing import List
torch_operations = {'mul': torch.mul, 'add': torch.add,
                    'pow': torch.pow, 'exp':torch.exp,
                    'sin':torch.sin, 'cos':torch.cos,
                    'log': torch.log}


DEBUG =False


class FeedbackControl_Kernel(Kernel):
    def __init__(
            self,
            a:torch.Tensor,
            v:torch.Tensor,
            active_dims=None
        ):
        super().__init__(active_dims=active_dims)
        self.num_tasks = 3
        
        task_range = range(3)

        K = [[None for j in task_range] for i in task_range]

        self.cov_alpha = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
        self.cov_beta = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
        self.cov_controller = Controller_Kernel(self.cov_beta, a, v)

        K[0][0] = self.cov_alpha
        K[0][1] = Zero_Kernel()
        K[0][2] = self.cov_alpha
        K[1][0] = Zero_Kernel()
        K[1][1] = self.cov_beta         
        K[1][2] = self.cov_controller                                       #5
        K[2][0] = self.cov_alpha        
        K[2][1] = self.cov_controller                                       #7
        K[2][2] = self.cov_alpha + self.cov_controller #+ Noise_Kernel(1e-8) #8

        self.K = K

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        
        K_list = list() 
        for row in self.K:
            for kernel in row:
                K_list.append(kernel(x1, x2, diag=False, last_dim_is_batch=False, **params).to_dense())
        K = einops.rearrange(K_list, '(t1 t2) h w -> (h t1) (w t2)', t1=self.num_tasks, t2=self.num_tasks)
        return K
    
    def num_outputs_per_input(self, x1, x2):
        return self.num_tasks
    
class Controller_Kernel(Kernel):
    def __init__(self, sub_kernel:Kernel, a:torch.Tensor, v:torch.Tensor):
        super().__init__(active_dims=None)
        self.a = a
        self.v = v
        self.sub_kernel = sub_kernel

    def control_law(self, x:torch.Tensor, y_ref=0):
        return - x @ self.a # + self.v * y_ref

    def forward(self, x1:torch.Tensor, x2:torch.Tensor, **params):
        k = self.sub_kernel(x1, x2, **params)
        u_1 = torch.diag(self.control_law(x1).squeeze(1))
        u_2 = torch.diag(self.control_law(x2).squeeze(1))
        if torch.equal(x1, x2):
            ret = u_1 * k * u_2.t()
            return ret
        else:
            ret = u_1 @ k @ u_2
            # _u_1 = self.control_law(x1)
            # _u_2 = self.control_law(x2)
            # _ret = _u_1.t() * k * _u_2
        return ret
        
        

class _Diagonal_Canonical_Kernel(Kernel):
    def __init__(
            self, 
            num_tasks, 
            eigenvalues:torch.Tensor, 
            eigenvectors:torch.Tensor,
            control:torch.Tensor,
            u:torch.Tensor,
            active_dims=None
        ):
        super(_Diagonal_Canonical_Kernel, self).__init__(active_dims=active_dims)
        self.num_tasks = num_tasks
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        eigenvectors_inv = eigenvectors.inverse()
        eigenvectors_t = eigenvectors.t()

        # b = eigenvectors_inv @ control
        b = control

        task_range = range(num_tasks)

        K = [[None for j in task_range] for i in task_range]
        _K = K
        for i in task_range:
            for j in task_range:
                sol1 = First_Order_Differential_Solution(eigenvalues[i], b[i])
                sol2 = First_Order_Differential_Solution(eigenvalues[j], b[j])
                _K[i][j] = _First_Order_Differential_Kernel(sol1, sol2, u)

        self.K = _K



    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        
        K_list = list() 
        for row in self.K:
            for kernel in row:
                K_list.append(kernel(x1, x2, diag=False, last_dim_is_batch=False, **params).to_dense())
        # from https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/6
        #if K_list[0].ndim == 1:
        #    K_list = [kk.unsqueeze(1) for kk in K_list]
        K = einops.rearrange(K_list, '(t1 t2) h w -> (h t1) (w t2)', t1=self.num_tasks, t2=self.num_tasks)
        return K
    
    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

class Diagonal_Canonical_Kernel(Kernel):
    def __init__(
            self, 
            num_tasks, 
            eigenvalues:torch.Tensor, 
            eigenvectors:torch.Tensor,
            control:torch.Tensor,
            active_dims=None
        ):
        super(Diagonal_Canonical_Kernel, self).__init__(active_dims=active_dims)
        self.num_tasks = num_tasks
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        eigenvectors_inv = eigenvectors.inverse()
        eigenvectors_t = eigenvectors.t()

        b = eigenvectors_inv @ control

        # kernels = []
        # for i in range(num_tasks):
        #     if b[i] == 0:
        #         kernels.append(First_Order_Differential_Kernel(eigenvalues[i]))
        #     else:
        #         kernels.append(First_Order_Differential_Kernel(eigenvalues[i], u=b[i]))
        # self.kernels = torch.nn.ModuleList(kernels)

        task_range = range(num_tasks)

        k_eigen= [[Constant_Kernel(eigenvectors[i, j]) for j in task_range] for i in task_range]
        k_eigen_t= [[Constant_Kernel(eigenvectors_t[i, j]) for j in task_range] for i in task_range]
        k_eigen_inv = [[Constant_Kernel(eigenvectors_inv[i, j]) for j in task_range] for i in task_range]

        K = [[None for j in task_range] for i in task_range]
        _K = K
        for i in task_range:
            for j in task_range:
                # _K[i].append(Constant_Kernel(k_eigen[i][j]) * First_Order_Differential_Kernel(eigenvalues[j], b[j]))
                #_K[i].append(k_eigen[i][j] * First_Order_Differential_Kernel(eigenvalues[j], b[j]))
                _K[i][j] = (k_eigen[i][j] * First_Order_Differential_Kernel(eigenvalues[j], b[j]))

        for i in task_range:
            for j in task_range:
                for l in task_range:
                    if l == 0:
                        K[i][j] = _K[i][l] * k_eigen_t[l][j]
                    else:
                        K[i][j] += _K[i][l] * k_eigen_t[l][j]
                #K_2[i].append(K[i][0] * Constant_Kernel(k_eigen_inv[0][j])+ K[i][1] * Constant_Kernel(k_eigen_inv[1][j]))
                # K[i].append(_K[i][0] * k_eigen_t[0][j]+ _K[i][1] * k_eigen_t[1][j])
                # K_2[i].append(sum([K[i][l] * Constant_Kernel(k_eigen_inv[l][j]) for l in range(len(K))])) #FIXME: how is sum possible
                # K[i][j] = (_K[i][0] * k_eigen_t[0][j]+ _K[i][1] * k_eigen_t[1][j])

        self.K = K



    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if last_dim_is_batch:
            raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        
        K_list = list() 
        for row in self.K:
            for kernel in row:
                K_list.append(kernel(x1, x2, diag=False, last_dim_is_batch=False, **params).to_dense())
        # from https://discuss.pytorch.org/t/how-to-interleave-two-tensors-along-certain-dimension/11332/6
        #if K_list[0].ndim == 1:
        #    K_list = [kk.unsqueeze(1) for kk in K_list]
        K = einops.rearrange(K_list, '(t1 t2) h w -> (h t1) (w t2)', t1=self.num_tasks, t2=self.num_tasks)
        return K
    
    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks


class First_Order_Differential_Solution(torch.nn.Module):
    def __init__(self, a, b):
        super(First_Order_Differential_Solution, self).__init__()
        self.a = a
        self.b = b

    def forward(self, t:torch.Tensor, u:torch.Tensor):
        if self.b is None or self.b == 0:
            return 1 / self.a * torch.exp(self.a*t)
        else:
            return u * self.b / self.a * torch.exp(self.a*t)
        
class _First_Order_Differential_Kernel(Kernel):
    def __init__(self, sol1:First_Order_Differential_Solution, sol2:First_Order_Differential_Solution, u):
        super(_First_Order_Differential_Kernel, self).__init__()
        self.sol1 = sol1
        self.sol2 = sol2
        self.u = u

    def forward(self, x1, x2, **params):
        if x2 is None:
            x2 = x1.t()

        return self.sol1(x1,self.u) * self.sol2(x2.t(),self.u)
        

class First_Order_Differential_Kernel(Kernel):
    def __init__(self, a, u=None):
        super(First_Order_Differential_Kernel, self).__init__()
        self.a = a
        if u == 0:
            self.u = None
        else:
            self.u = u

    def forward(self, x1, x2, **params):
        if x2 is None:
            input_sum = x1+x1.t()
        else:
            input_sum = x1+x2.t()
        # input_sum = self.covar_sum(x1, x2, **params)
        exponent = torch.exp(self.a*input_sum)
        if self.u is None:
            return 1 / self.a**2 * exponent
        else:
            return self.u**2 / self.a**2 * exponent
            
    
    def covar_sum(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params,
    ) -> torch.Tensor:
        r"""
        computes to sum of the inputs. Inspired by self.covar_dist
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        res = None

        if diag:
            res = x1 + x2
            return res
        else:
            return self.sum(x1, x2)
        
    def sum(self, x1, x2):
        """
        Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
        """
        res = torch.cdist(x1, -x2)
        return res


class Constant_Kernel(Kernel):
    def __init__(self, c, active_dims=None):
        super(Constant_Kernel, self).__init__(active_dims=active_dims)
        self.c = c

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # if last_dim_is_batch:
        #     raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        if diag:
            return self.c
        return self.c * torch.ones(x1.size(0), x2.size(0))

    def num_outputs_per_input(self, x1, x2):
        return 1
    
class Zero_Kernel(Kernel):
    def __init__(self, active_dims=None):
        super(Zero_Kernel, self).__init__(active_dims=active_dims)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # if last_dim_is_batch:
        #     raise RuntimeError("MultitaskKernel does not accept the last_dim_is_batch argument.")
        if diag:
            return 0
        return torch.zeros(x1.size(0), x2.size(0))

    def num_outputs_per_input(self, x1, x2):
        return 1
    
class Noise_Kernel(Kernel):
    def __init__(self, noise_variance=1.0, active_dims=None):
        super(Noise_Kernel, self).__init__(active_dims=active_dims)
        self.noise_variance = noise_variance

    def forward(self, x1, x2, diag=False, **params):
        x1_eq_x2 = torch.equal(x1, x2)
        if x1_eq_x2:
            if diag:
                return self.noise_variance * torch.ones(x1.size(0))
            else:
                return self.noise_variance * torch.eye(x1.size(0))
        else:
            if diag:
                return torch.zeros(x1.size(0))
            else:
                return torch.zeros(x1.size(0), x2.size(0))
class LODE_Kernel(Kernel):
    def __init__(self, A, common_terms, active_dims=None, verbose=False):
        super(LODE_Kernel, self).__init__(active_dims=active_dims)

        # self.model_parameters = torch.nn.ParameterDict()
        # FIXME:
        self.model_parameters = torch.nn.ParameterDict({
           "a":torch.nn.Parameter(torch.tensor(0.0)),
           "b":torch.nn.Parameter(torch.tensor(0.0))
        })
        #self.num_tasks = num_tasks

        D, U, V = A.smith_form()
        if verbose:
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

        if diag:
            return K.diag()
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





def create_kernel_matrix_from_diagonal(D):
    t1, t2 = var("t1, t2")
    translation_dictionary = dict()
    param_dict = torch.nn.ParameterDict()
    #sage_covariance_matrix = [[0 for cell in range(max(len(D.rows()), len(D.columns())))] for row in range(max(len(D.rows()), len(D.columns())))]
    sage_covariance_matrix = [[0 for cell in range(len(D.columns()))] for row in range(len(D.columns()))]
    #for i in range(max(len(D.rows()), len(D.columns()))):
    for i in range(len(D.columns())):
        if i > len(D.diagonal())-1:
            entry = 0
        else:
            entry = D[i][i]
        var(f"LODEGP_kernel_{i}")
        if entry == 0:
            param_dict[f"signal_variance_{i}"] = torch.nn.Parameter(torch.tensor(float(0.)))
            param_dict[f"lengthscale_{i}"] = torch.nn.Parameter(torch.tensor(float(0.)))
            # Create an SE kernel
            var(f"signal_variance_{i}")
            var(f"lengthscale_{i}")
            #translation_dictionary[f"LODEGP_kernel_{i}"] = globals()[f"signal_variance_{i}"]**2 * exp(-1/2*(t1-t2)**2/globals()[f"lengthscale_{i}"]**2)
            translation_dictionary[f"LODEGP_kernel_{i}"] = globals()[f"signal_variance_{i}"]**2 * exp(-(t1-t2)**2/globals()[f"lengthscale_{i}"]**2)
        elif entry == 1:
            translation_dictionary[f"LODEGP_kernel_{i}"] = 0 
        else:
            kernel_translation_kernel = 0
            roots = entry.roots(ring=CC)
            roots_copy = cp.deepcopy(roots)
            for rootnum, root in enumerate(roots):
                # Complex root, i.e. sinusoidal exponential
                #if root[0].is_complex():
                param_dict[f"signal_variance_{i}_{rootnum}"] = torch.nn.Parameter(torch.tensor(float(0.)))
                var(f"signal_variance_{i}_{rootnum}")
                if root[0].is_imaginary() and not root[0].imag() == 0.0:
                    # Check to prevent conjugates creating additional kernels
                    if not root[0].conjugate() in [r[0] for r in roots_copy]:
                        continue

                    # If it doesn't exist then it's new so find and pop the complex conjugate of the current root
                    roots_copy.remove((root[0].conjugate(), root[1]))
                    roots_copy.remove(root)

                    # Create sinusoidal kernel
                    var("exponent_runner")
                    kernel_translation_kernel += globals()[f"signal_variance_{i}_{rootnum}"]**2*sum(t1**globals()["exponent_runner"] * t2**globals()["exponent_runner"], globals()["exponent_runner"], 0, root[1]-1) *\
                                                    exp(root[0].real()*(t1 + t2)) * cos(root[0].imag()*(t1-t2))
                else:
                    var("exponent_runner")
                    # Create the exponential kernel functions
                    kernel_translation_kernel += globals()[f"signal_variance_{i}_{rootnum}"]**2*sum(t1**globals()["exponent_runner"] * t2**globals()["exponent_runner"], globals()["exponent_runner"], 0, root[1]-1) * exp(root[0]*(t1+t2))
            translation_dictionary[f"LODEGP_kernel_{i}"] = kernel_translation_kernel 
        sage_covariance_matrix[i][i] = globals()[f"LODEGP_kernel_{i}"]
    return sage_covariance_matrix, translation_dictionary, param_dict


def build_dict_for_SR_expression(expression):
    final_dict = {}
    for coeff_dx1 in expression.coefficients(dx1):
        final_dict.update({(Integer(coeff_dx1[1]), Integer(coeff_dx2[1])): coeff_dx2[0] for coeff_dx2 in coeff_dx1[0].coefficients(dx2)})
    return final_dict

def differentiate_kernel_matrix(K, V, Vt, kernel_translation_dictionary):
    """
    This code takes the sage covariance matrix and differentiation matrices
    and returns a list of lists containing the results of the `compile` 
    commands that calculate the respective cov. fct. entry
    """
    sage_multiplication_kernel_matrix = matrix(K.base_ring(), len(K[0]), len(K[0]), (V*K*Vt))
    final_kernel_matrix = [[None for i in range(len(K[0]))] for j in range(len(K[0]))]
    for i, row in  enumerate(sage_multiplication_kernel_matrix):
        for j, cell in enumerate(row):
            cell_expression = 0
            diff_dictionary = build_dict_for_SR_expression(cell)
            for summand in diff_dictionary:
                #temp_cell_expression = mul([K[i][i] for i, multiplicant in enumerate(summand[3:]) if multiplicant > 0])
                temp_cell_expression = diff_dictionary[summand]
                for kernel_translation in kernel_translation_dictionary:
                    if kernel_translation in str(temp_cell_expression):
                        temp_cell_expression = SR(temp_cell_expression)
                        #cell = cell.factor()
                        #replace
                        temp_cell_expression = temp_cell_expression.substitute(globals()[kernel_translation]==kernel_translation_dictionary[kernel_translation])

                # And now that everything is replaced: diff that bad boy!
                cell_expression += SR(temp_cell_expression).diff(t1, summand[0]).diff(t2, summand[1])
            final_kernel_matrix[i][j] = cell_expression
    return final_kernel_matrix 


def replace_sum_and_diff(kernelmatrix, sumname="t_sum", diffname="t_diff", onesname="t_ones", zerosname="t_zeroes"):
    result_kernel_matrix = cp.deepcopy(kernelmatrix)
    var(sumname, diffname)
    for i, row in enumerate(kernelmatrix):
        for j, cell in enumerate(row):
            # Check if the cell is just a number
            if type(cell) == sage.symbolic.expression.Expression and not cell.is_numeric():
                #result_kernel_matrix[i][j] = cell.substitute({t1-t2:globals()[diffname], t1+t2:globals()[sumname]})
                result_kernel_matrix[i][j] = cell.substitute({t1:0.5*globals()[sumname] + 0.5*globals()[diffname], t2:0.5*globals()[sumname] - 0.5*globals()[diffname]})
            # This case is assumed to be just a constant, but we require it to be of 
            # the same size as the other covariance submatrices
            else:
                if cell == 0:
                    var(zerosname)
                    result_kernel_matrix[i][j] = globals()[zerosname]
                else:
                    var(onesname)
                    result_kernel_matrix[i][j] = cell * globals()[onesname]
    return result_kernel_matrix


def replace_basic_operations(kernel_string):
    # Define the regex replacement rules for the text
    regex_replacements_multi_group = {
        "exp" : [r'(e\^)\((([^()]*|\(([^()]*|\([^()]*\))*\))*)\)', "torch.exp"],
        "exp_singular" : [r'(e\^)([0-9a-zA-Z_]*)', "torch.exp"]
    }
    regex_replacements_single_group = {
        "sin" : [r'sin', "torch.sin"],
        "cos" : [r'cos', "torch.cos"],
        "pow" : [r'\^', "**"]
    }
    for replace_term in regex_replacements_multi_group:
        m = re.search(regex_replacements_multi_group[replace_term][0], kernel_string)
        if not m is None:
            # There is a second group, i.e. we have exp(something)
            kernel_string = re.sub(regex_replacements_multi_group[replace_term][0], f'{regex_replacements_multi_group[replace_term][1]}'+r"(\2)", kernel_string)
    for replace_term in regex_replacements_single_group:
        m = re.search(regex_replacements_single_group[replace_term][0], kernel_string)
        if not m is None:
            kernel_string = re.sub(regex_replacements_single_group[replace_term][0], f'{regex_replacements_single_group[replace_term][1]}', kernel_string)

    return kernel_string 


def replace_parameters(kernel_string, model_parameters, common_terms = []):
    regex_replace_string = r"(^|[\*\+\/\(\)\-\s])(REPLACE)([\*\+\/\(\)\-\s]|$)"
    
    for term in common_terms:
        if term in kernel_string:
            kernel_string = re.sub(regex_replace_string.replace("REPLACE", term), r"\1" + f"common_terms[\"{term}\"]" + r"\3", kernel_string)

    for model_param in model_parameters:
        kernel_string = re.sub(regex_replace_string.replace("REPLACE", model_param), r"\1"+f"torch.exp(model_parameters[\"{model_param}\"])"+r"\3", kernel_string)

    return kernel_string 


def verify_sage_entry(kernel_string, local_vars):
    # This is a call to willingly produce an error if the string is not originally coming from sage
    try:
        if type(kernel_string) == sage.symbolic.expression.Expression:
            kernel_string = kernel_string.simplify()
        kernel_string = str(kernel_string)
        sage_eval(kernel_string, locals = local_vars)
    except Exception as E:
        raise Exception(f"The string was not safe and has not been used to construct the Kernel.\nPlease ensure that only valid operations are part of the kernel and all variables have been declared.\nYour kernel string was:\n'{kernel_string}'")


def translate_kernel_matrix_to_gpytorch_kernel(kernelmatrix, paramdict, common_terms=[]):
    kernel_call_matrix = [[] for i in range(len(kernelmatrix))]
    for rownum, row in enumerate(kernelmatrix):
        for colnum, cell in enumerate(row):
            # First thing I do: Verify that the entry is a valid sage command
            local_vars = {str(v):v for v in SR(cell).variables()}
            verify_sage_entry(cell, local_vars)
            # Now translate the cell to a call
            replaced_op_cell = replace_basic_operations(str(cell))
            replaced_var_cell = replace_parameters(replaced_op_cell, paramdict, common_terms)
            #print("DEBUG: replaced_var_cell:")
            #print(replaced_var_cell)
            kernel_call_matrix[rownum].append(compile(replaced_var_cell, "", "eval"))



    return kernel_call_matrix
