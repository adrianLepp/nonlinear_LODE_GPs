from nonlinear_LODE_GPs.kernels import First_Order_Differential_Kernel, Diagonal_Canonical_Kernel, Constant_Kernel
from nonlinear_LODE_GPs.systems.linearize import solve_for_equilibrium
from nonlinear_LODE_GPs.helpers import load_system
from nonlinear_LODE_GPs.lodegp import Diagonal_Canonical_GP, optimize_gp
from nonlinear_LODE_GPs.mean_modules import Equilibrium_Mean
from nonlinear_LODE_GPs.helpers import simulate_system, downsample_data, State_Description, Time_Def, Data_Def
from result_reporter.latex_exporter import plot_states
import matplotlib.pyplot as plt
# from gpytorch.kernels import Constant_Kernel, Constant_Kernel
import torch
import numpy
import gpytorch
import scipy.signal as signal


# dist_kernel = Constant_Kernel()
u_e_rel = 0.1

system_name = "nonlinear_watertank"
system = load_system(system_name)
num_tasks = system.state_dimension
_ , equilibrium = system.get_ODEmatrix(u_e_rel)
#x_0 = numpy.array(equilibrium)
x_0 = numpy.array([0,0,0])
states = State_Description(equilibrium=torch.tensor(equilibrium), init=torch.tensor(equilibrium))

t  = 100
optim_steps = 10
downsample = 10
sim_time = Time_Def(0, t, step=0.1)
train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t-0, step=0.1)

A_r = system.A_r
b_r = system.b_r
D_k, T_k = A_r.eigenmatrix_right()
eigenval = torch.tensor(numpy.array(D_k.n())).diag()
eigenvec = torch.tensor(numpy.array(T_k.n()))
control = torch.tensor([0.0, 0.0])



u_eq = numpy.ones((sim_time.count,1)) * (u_e_rel * system.param.u) 
u = u_eq + equilibrium[-1] 


_train_x, _train_y= simulate_system(system, x_0[0:system.state_dimension], sim_time, u)
train_x, train_y = downsample_data(_train_x, _train_y, downsample)

# Equilibrium simulation
eq_sys = signal.StateSpace(numpy.array(A_r.n()), numpy.array(b_r.n()), numpy.eye(num_tasks), numpy.zeros((num_tasks,1)))
t, y_eq, x_eq = signal.lsim(eq_sys, u_eq , sim_time.linspace(), x_0[0:system.state_dimension]-equilibrium[0:system.state_dimension])


#Diagonal Form simulation
A_d = numpy.array(D_k.n())
V_d = numpy.array(T_k.n())
V_d_inv = numpy.linalg.inv(V_d)
B_d = V_d_inv @ numpy.array(b_r.n())
C_d = V_d

diag_sys = signal.StateSpace(A_d, B_d, C_d, numpy.zeros((num_tasks,1)))
t, y_diag, x_diag = signal.lsim(diag_sys, u_eq , sim_time.linspace(), V_d_inv@(x_0[0:system.state_dimension]-equilibrium[0:system.state_dimension]))

#'''
# ALL GP STUFF 

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
mean_module = Equilibrium_Mean(equilibrium[0:-1], num_tasks)

model = Diagonal_Canonical_GP(
    train_x,
    train_y[:,:-1], 
    likelihood, 
    num_tasks, 
    eigenvec,
    eigenval,
    control, 
    mean_module
)

training_loss = optimize_gp(model,optim_steps)

test_x = test_time.linspace()

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.debug(False):
    output = likelihood(model(test_x))

full_out  = numpy.concatenate([output.mean.numpy(), u ],1)

test_data = Data_Def(test_x.numpy(), full_out, system.state_dimension, system.control_dimension, test_time)
#'''


train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension, train_time)
eq_data = Data_Def(sim_time.linspace(), numpy.concatenate([x_eq, u_eq ],1)+equilibrium, system.state_dimension, system.control_dimension, sim_time)

_diag_data = Data_Def(sim_time.linspace(), numpy.concatenate([x_diag, u_eq ],1)+equilibrium, system.state_dimension, system.control_dimension, sim_time)
diag_data = Data_Def(sim_time.linspace(), numpy.concatenate([y_diag, u_eq ],1)+equilibrium, system.state_dimension, system.control_dimension, sim_time)
fig_results = plot_states([test_data, diag_data, train_data], header=['x1', 'x2', 'u'])

plt.show()

# kern = Diagonal_Canonical_Kernel(num_tasks, eigenvalues=eigenval, eigenvectors=eigenvec, control=control)

# diff_kernel = First_Order_Differential_Kernel(a=1, u=2)


# x1 = torch.tensor([1.0, 2.0, 3.0])
# x2 = torch.tensor([2.0, 3.0, 4.0, 5.0])


# print(dist_kernel(x1, x2))
# diff_kernel_2 = eigenvectors[0].unsqueeze(0) @ diff_kernel(x1, x2) @ eigenvectors[0].unsqueeze(1)
#diff_kernel_2 = eigenvectors[0,0] @ diff_kernel(x1, x2) @ eigenvectors.t()[0,0]
# diff_kernel_2 = Constant_Kernel(eigenvectors[0,0]) * diff_kernel * Constant_Kernel(eigenvectors_inv[0,0])


# print(kern(x1, x2).to_dense())