


import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
from mean_modules import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from  lodegp import LODEGP, Sum_LODEGP, optimize_gp
from helpers import *

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_watertank"

train_time = Time_Def(
    0, 
    100, 
    step=1
)

test_time = Time_Def(
    0, 
    100, 
    step=0.1
)

u_0 = 0.2
u_1 = 0.21

system = load_system(system_name)
num_tasks = system.dimension


system_matrix_0 , x0 = system.get_ODEmatrix(u_0)
system_matrix_1 , x1 = system.get_ODEmatrix(u_1)

system_matrices = [system_matrix_0, system_matrix_1]
equilibriums = [torch.tensor(x0), torch.tensor(x1)]
centers = [0, 1000]

# x_0 = torch.tensor(x0)
# system_matrix , equilibrium = system.get_ODEmatrix(u_1)
# x_e = torch.tensor(equilibrium)

#states_0 = State_Description(x_e, x_0)

u = np.linspace(u_0 * system.param.u, u_0 * system.param.u, train_time.count,axis=-1)


train_x, train_y= simulate_system(system, equilibriums[1][0:system.state_dimension], train_time.start, train_time.end, train_time.count, u)

#sol = solve_ivp(system.stateTransition, [train_time.start, train_time.end], x_0[0:system.state_dimension], method='RK45', t_eval=t_reference.numpy(), args=(u, train_time.step ), max_step=train_time.step)#, max_step=dt ,  atol = 1, rtol = 1


#x_sim_current = np.concatenate([sol.y.transpose()[1::], u_ref[1::]], axis=1)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = Sum_LODEGP(train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=44194)


# mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
# model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)
optimize_gp(model,100)

test_x = torch.linspace(test_time.start, test_time.end, test_time.count)
model.eval()
likelihood.eval()


#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))

train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension)
test_data = Data_Def(test_x.numpy(), output.mean.numpy(), system.state_dimension, system.control_dimension)

plot_results(train_data, test_data, equilibrium=equilibriums[1])