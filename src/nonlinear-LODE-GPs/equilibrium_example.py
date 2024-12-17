import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from  lodegp import Equilibrium_LODEGP, optimize_gp
from helpers import *
from likelihoods import FixedTaskNoiseMultitaskLikelihood2

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_threetank"


train_time = Time_Def(0, 100, step=1)
test_time = Time_Def(0, 200, step=0.1)

u_e_rel = 0.1

u_rel = 0.3


system = load_system(system_name)

num_tasks = system.dimension

_ , x0 = system.get_ODEmatrix(u_e_rel)
system_matrix , equilibrium = system.get_ODEmatrix(u_e_rel)
#system_matrix = system.get_parameterized_ODEmatrix()

#D, U, V = system_matrix.smith_form()

#x_0 = np.array(equilibrium)
x_0 = np.array(x0)


u = np.linspace(u_rel * system.param.u, u_rel * system.param.u, train_time.count)

train_x, train_y= simulate_system(system, x_0[0:system.state_dimension], train_time.start, train_time.end, train_time.count, u)
#train_y = train_y - torch.tensor(x_0)

# %% train

task_noise = [1e-8, 1e-8, 1e-8, 1e-7]
train_noise = [1.0,1.0]

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
#likelihood2 = FixedTaskNoiseMultitaskLikelihood2(num_tasks=num_tasks ,data_noise=torch.tensor(train_noise), task_noise=task_noise, rank=0)
model = Equilibrium_LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, equilibrium) #system.state_var, system.control_var
#model = Param_LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, x_0) #system.state_var, system.control_var

#likelihood.noise = torch.tensor(1e-8)
optimize_gp(model,100)
#likelihood.noise = torch.tensor(1e-8)

# %% test

print(f"the equilibrium is {equilibrium}")

test_x = create_test_inputs(test_time.count, test_time.step, test_time.start, test_time.end, 1)

model.eval()
likelihood.eval()

#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))

# %% recreate ODE from splines of prediction

test_data = Data_Def(test_x.numpy(), output.mean.numpy(), system.state_dimension, system.control_dimension)
train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension)

plot_results(train_data, test_data)