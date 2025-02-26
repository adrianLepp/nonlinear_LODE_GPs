


import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from nonlinear_LODE_GPs.kernels import *
from nonlinear_LODE_GPs.mean_modules import *
import torch
import matplotlib.pyplot as plt
from result_reporter.latex_exporter import plot_states

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import optimize_gp
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.weighting import Gaussian_Weight
from nonlinear_LODE_GPs.localgp import Sum_LODEGP
from nonlinear_LODE_GPs.gp import MultitaskGPModel, BatchIndependentMultitaskGPModel

torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_watertank"

optim_steps = 300

u_0 = 0.2
u_1 = 0.25
u_2 = 0.3
u_ctrl = 0.5
controls = [
    0.1,
    # 0.25,
    0.2,
    # 0.25, 
    0.3,
    # 0.4,
    # 0.5,
    ]
t0 = 0.0
t1 = 100.0
timestamps = [0.0, 500.0, t1]

output_distance = True

downsample = 20
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
    centers = [x_i.unsqueeze(0) for x_i in equilibriums]
else:
    centers = [torch.tensor([[t_i]]) for t_i in timestamps]

#l = 2.65e-3
#l = 44194
w_func = Gaussian_Weight(centers[0])
# w_func.length = 1
d = w_func.covar_dist(centers[1], w_func.center, square_dist=True)
l = d*torch.sqrt(torch.tensor(2)) /8
l.requires_grad = False
# print(l)
#l = 1000
# l = torch.tensor([[176776.6953]])

u = np.ones((sim_time.count,1)) * u_ctrl * system.param.u

_train_x, _train_y= simulate_system(system, equilibriums[0][0:system.state_dimension], sim_time, u)
train_x, train_y = downsample_data(_train_x, _train_y, downsample)


test_x = test_time.linspace()

#x_sim_current = np.concatenate([sol.y.transpose()[1::], u_ref[1::]], axis=1)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks) #, noise_constraint=gpytorch.constraints.Positive()
pre_model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, num_tasks) #MultitaskGPModel
optimize_gp(pre_model, optim_steps)
pre_model.eval()
pre_model.likelihood.eval()

# with torch.no_grad():
#     pre_estimate = pre_model.estimate(test_x)

train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension)
# pre_data = Data_Def(test_x.numpy(), pre_estimate.numpy(), system.state_dimension, system.control_dimension)
# fig_results = plot_states([ pre_data, train_data],data_names = ['pre', 'train'], )

# plt.show()

model = Sum_LODEGP(
    train_x, 
    train_y, 
    likelihood, 
    num_tasks, 
    system_matrices, 
    equilibriums, 
    centers, 
    weight_lengthscale=l, 
    output_distance=output_distance,
    additive_kernel=True,
    pre_model=pre_model
    )


optimize_gp(model, optim_steps)

model.eval()
likelihood.eval()


#output = model(test_x)
with torch.no_grad(): #gpytorch.settings.prior_mode(True)
    output = likelihood(model(test_x)).mean
    # output = model.mean_module(test_x)

test_data = Data_Def(test_x.numpy(), output.numpy(), system.state_dimension, system.control_dimension)

plot_results(train_data, test_data, equilibrium=equilibriums[-1])#,, equilibrium=equilibriums[4]