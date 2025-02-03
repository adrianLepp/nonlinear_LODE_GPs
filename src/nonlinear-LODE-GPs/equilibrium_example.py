import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import torch

# ----------------------------------------------------------------------------
from  lodegp import  optimize_gp, LODEGP
from helpers import *
from mean_modules import Equilibrium_Mean

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_watertank"


train_time = Time_Def(0, 100, step=1)
test_time = Time_Def(0, 100, step=0.1)

u_e_rel = 0.1

u_rel = 0.2


system = load_system(system_name)

num_tasks = system.dimension

_ , x0 = system.get_ODEmatrix(u_e_rel)
system_matrix , equilibrium = system.get_ODEmatrix(u_e_rel)
#system_matrix = system.get_parameterized_ODEmatrix()

#D, U, V = system_matrix.smith_form()

x_0 = np.array(x0)


#u = np.linspace(u_rel * system.param.u, u_rel * system.param.u, train_time.count)
u = np.ones((train_time.count,1)) * u_rel * system.param.u

train_x, train_y= simulate_system(system, x_0[0:system.state_dimension], train_time, u)

# %% train

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
    num_tasks, 
    #rank=num_tasks, 
    #noise_constraint=gpytorch.constraints.Positive(), 
    has_global_noise=False, 
    has_task_noise=True
)

mean_module = Equilibrium_Mean(equilibrium, num_tasks)
model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module) #system.state_var, system.control_var

optimize_gp(model,100)

# %% test

test_x = create_test_inputs(test_time.count, test_time.step, test_time.start, test_time.end, 1)

model.eval()
likelihood.eval()

with torch.no_grad():
    output = likelihood(model(test_x))

# %% recreate ODE from splines of prediction

test_data = Data_Def(test_x.numpy(), output.mean.numpy()-np.array(equilibrium), system.state_dimension, system.control_dimension)
train_data = Data_Def(train_x.numpy(), train_y.numpy()-np.array(equilibrium), system.state_dimension, system.control_dimension)

plot_results(train_data, test_data)