import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from  lodegp import Param_LODEGP, optimize_gp
from helpers import *

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_threetank"

class Time_Def():
    def __init__(self, start, end, count):
        self.start = start
        self.end = end
        self.count = count
        self.step = (end-start)/count



train_time = Time_Def(0, 20, 200)
test_time = Time_Def(0, 20, 200)



u_e_rel = 0.1

u_rel = 0.3


system = load_system(system_name)

num_tasks = system.dimension


_ , equilibrium = system.get_ODEmatrix(u_e_rel)
system_matrix = system.get_parameterized_ODEmatrix()

D, U, V = system_matrix.smith_form()

x_0 = np.array(equilibrium)

u = np.linspace(u_rel * system.param.u, u_rel * system.param.u, test_time.count)

train_x, train_y= simulate_system(system, x_0[0:system.state_dimension], test_time.start, test_time.end, test_time.count, u)
train_y = train_y - torch.tensor(x_0)

# %% train

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = Param_LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, x_0) #system.state_var, system.control_var

optimize_gp(model,10)

# %% test

test_x = create_test_inputs(test_time.count, test_time.step, test_time.start, test_time.end, 1)

model.eval()
likelihood.eval()

#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))

# %% recreate ODE from splines of prediction


train_x_np = train_x.numpy()

train_x_np = train_x.numpy()
train_y_np = train_y.numpy() + x_e
test_x_np = test_x.numpy()
estimation = output.mean.numpy() + x_e

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

for i in range(system.state_dimension):
    color = f'C{i}'
    ax1.plot(train_x_np, train_y_np[:, i], '.', color=color, label=f'x{i+1}_train')    
    ax1.plot(test_x_np, estimation[:, i], color=color, label=f'x{i+1}_est', alpha=0.5)

for i in range(system.control_dimension):
    idx = system.state_dimension + i
    color = f'C{idx}'
    ax2.plot(train_x_np, train_y_np[:, idx], '.', color=color, label=f'x{idx+1}_train')
    ax2.plot(test_x_np, estimation[:, idx], color=color, label=f'x{idx+1}est', alpha=0.5)

ax2.tick_params(axis='y', labelcolor=color)
ax1.legend()
ax2.legend()
ax1.grid(True)

plt.show()