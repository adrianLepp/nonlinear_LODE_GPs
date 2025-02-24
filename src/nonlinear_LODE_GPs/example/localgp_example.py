


import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from nonlinear_LODE_GPs.kernels import *
from nonlinear_LODE_GPs.mean_modules import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import optimize_gp
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.weighting import Gaussian_Weight
from nonlinear_LODE_GPs.localgp import Sum_LODEGP

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_watertank"

optim_steps = 200

u_0 = 0.2
u_1 = 0.3
controls = [u_0, u_1]#, u_1

t0 = 0.0
t1 = 1000.0
timestamps = [0.0, t1]

output_distance = False

downsample = 100
sim_time = Time_Def(t0, t1, step=0.1)
train_time = Time_Def(t0, t1, step=sim_time.step*downsample)
test_time = Time_Def(t0, t1, step=0.1)

system = load_system(system_name)
num_tasks = system.dimension
system_matrices = []
equilibriums = []

for i in range(len(controls)):
    system_matrix, x = system.get_ODEmatrix(controls[i])
    system_matrices.append(system_matrix)
    equilibriums.append(torch.tensor(x))

if output_distance is True:
    centers = [torch.tensor([x_i]) for x_i in equilibriums]
else:
    centers = [torch.tensor([[t_i]]) for t_i in timestamps]

#l = 2.65e-3
#l = 44194
# w_func = Gaussian_Weight(centers[0])
# w_func.length = 1
# d = w_func.covar_dist(centers[1], w_func.center, square_dist=True)
# l = d*torch.sqrt(torch.tensor(2))/8
# print(l)
#l = 1000
l = torch.tensor([[176776.6953]])
# print(w_func.covar_dist(centers[1], w_func.center, square_dist=True))
# print(w_func(centers[1]))

# x_0 = torch.tensor(x0)
# system_matrix , equilibrium = system.get_ODEmatrix(u_1)
# x_e = torch.tensor(equilibrium)

#states_0 = State_Description(x_e, x_0)

# u = np.linspace(u_1 * system.param.u, u_1 * system.param.u, train_time.count,axis=-1)
u = np.ones((sim_time.count,1)) * u_1 * system.param.u

_train_x, _train_y= simulate_system(system, equilibriums[0][0:system.state_dimension], sim_time, u)
train_x, train_y = downsample_data(_train_x, _train_y, downsample)

#x_sim_current = np.concatenate([sol.y.transpose()[1::], u_ref[1::]], axis=1)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = Sum_LODEGP(train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=l, output_distance=output_distance)

optimize_gp(model, optim_steps)

test_x = test_time.linspace()
model.eval()
likelihood.eval()


#output = model(test_x)
with torch.no_grad(): #gpytorch.settings.prior_mode(True)
    output = likelihood(model(test_x)).mean
    # output = model.mean_module(test_x)

train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension)
test_data = Data_Def(test_x.numpy(), output.numpy(), system.state_dimension, system.control_dimension)

plot_results(train_data, test_data, equilibrium=equilibriums[-1])#,, equilibrium=equilibriums[4]