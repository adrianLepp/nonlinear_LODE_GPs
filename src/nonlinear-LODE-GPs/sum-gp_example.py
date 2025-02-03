


import gpytorch 
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
from mean_modules import *
import torch

# ----------------------------------------------------------------------------
from  lodegp import *
from helpers import *
from weighting import Weighting_Function
from sum_gp import Local_GP_Sum

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_watertank"

optim_steps = 100

equilibrium_controls = [
    0.1, 
    0.2, 
    0.3, 
    0.4, 
    0.5,
]

u_ctrl = 1

x0 = torch.tensor([0.0, 0.0])

t0 = 0.0
t1 = 150.0

train_time = Time_Def(
    t0, 
    t1, 
    step=1
)

test_delta = 0


test_time = Time_Def(
    t0 + test_delta, 
    t1 - test_delta, 
    step=0.1
)

system = load_system(system_name)
num_tasks = system.dimension

system_matrices = []
equilibriums = []
centers = []
for i in range(len(equilibrium_controls)):
    system_matrix , x_e = system.get_ODEmatrix(equilibrium_controls[i])
    system_matrices.append(system_matrix)
    equilibriums.append(torch.tensor(x_e))
    centers.append(torch.tensor([x_e]))

#l  = 1
#l = 2.65e-3
#l = 44194
w_func = Weighting_Function(centers[0])
d = w_func.covar_dist(centers[1], w_func.center, square_dist=True)
l = d*torch.sqrt(torch.tensor(2))/8
# print(l)
#l = 1000
l = l*4

u = np.linspace(u_ctrl * system.param.u, u_ctrl * system.param.u, train_time.count,axis=-1)


train_x, train_y= simulate_system(system, x0, train_time, u)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
#model = Weighted_Sum_GP(train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=l)
model = Local_GP_Sum(train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=l)#, 
model.optimize(optim_steps)

# mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
# model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)
#optimize_gp(model, optim_steps)

test_x = test_time.linspace()
model.eval()
likelihood.eval()


#output = model(test_x)
# with torch.no_grad() and gpytorch.settings.debug(False):
#     output = likelihood(model(test_x))
#     estimate = output.mean
with torch.no_grad():
    estimate, weights = model.predict(test_x)


train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension)
test_data = Data_Def(test_x.numpy(), estimate.numpy(), system.state_dimension, system.control_dimension)

plot_results(train_data, test_data, equilibrium=equilibriums[4])#, 
plot_weights(test_x, weights, title="Weighting Function")


# ----------------------------------------------------------------------------  
