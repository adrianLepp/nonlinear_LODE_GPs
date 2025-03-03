from nonlinear_LODE_GPs.kernels import First_Order_Differential_Kernel, _Diagonal_Canonical_Kernel, Constant_Kernel
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
u_e_rel = 0.2
u_rel = 0   

system_name = "nonlinear_watertank"
system = load_system(system_name)
num_tasks = system.state_dimension
_ , equilibrium = system.get_ODEmatrix(u_e_rel)

states = State_Description(
    equilibrium=torch.tensor(equilibrium), 
    # init=torch.tensor([0,0,0]),
    init=torch.tensor(equilibrium),
    )

t  = 100
optim_steps = 300
downsample = 10
sim_time = Time_Def(0, t, step=0.1)
train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t-0, step=0.1)

A_r = system.A_r
b_r = system.b_r
D_k, T_k = A_r.eigenmatrix_right()
eigenval = torch.tensor(numpy.array(D_k.n())).diag()
eigenvec = torch.tensor(numpy.array(T_k.n()))
control = torch.tensor([0.0]) - states.equilibrium[-1]


u = numpy.ones((sim_time.count,1)) * (u_rel * system.param.u) 
u_eq = u - states.equilibrium[-1].numpy() 


_train_x, _train_y= simulate_system(system, states.init[0:system.state_dimension], sim_time, u)
train_x, train_y = downsample_data(_train_x, _train_y, downsample)

# Equilibrium simulation
eq_sys = signal.StateSpace(numpy.array(A_r.n()), numpy.array(b_r.n()), numpy.eye(num_tasks), numpy.zeros((num_tasks,1)))
t, y_eq, x_eq = signal.lsim(eq_sys, u_eq , sim_time.linspace(), states.init[0:system.state_dimension]-states.equilibrium[0:system.state_dimension])


#Diagonal Form simulation
A_d = numpy.array(D_k.n())
V_d = numpy.array(T_k.n())
V_d_inv = numpy.linalg.inv(V_d)
B_d = V_d_inv @ numpy.array(b_r.n())
C_d = V_d

diag_sys = signal.StateSpace(A_d, B_d, C_d, numpy.zeros((num_tasks,1)))
t, y_diag, x_diag = signal.lsim(diag_sys, u_eq , sim_time.linspace(), V_d_inv@(states.init[0:system.state_dimension].numpy()-states.equilibrium[0:system.state_dimension].numpy()))

#'''
# ALL GP STUFF 


control_d = torch.tensor(B_d) #@ control

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
mean_module = Equilibrium_Mean(states.equilibrium[0:-1], num_tasks)

train_y_d = torch.tensor((V_d_inv @ (train_y[:,:-1]-states.equilibrium[0:system.state_dimension]).numpy().transpose()).transpose())

model = Diagonal_Canonical_GP(
    train_x,
    # train_y[:,:-1], 
    train_y_d,
    likelihood, 
    num_tasks, 
    eigenvec,
    eigenval,
    control_d, 
    control,
    # mean_module
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
train_data_d = Data_Def(train_x.numpy(), numpy.concatenate([train_y_d.numpy(), train_data.y[:,-1:] ],1), system.state_dimension, system.control_dimension, train_time)
# eq_data = Data_Def(sim_time.linspace(), numpy.concatenate([x_eq.unsqueeze(1), u_eq ],1)+equilibrium, system.state_dimension, system.control_dimension, sim_time)

_diag_data = Data_Def(sim_time.linspace(), numpy.concatenate([numpy.expand_dims(x_diag,1), u_eq ],1), system.state_dimension, system.control_dimension, sim_time)

# _diag_data = Data_Def(sim_time.linspace(), numpy.concatenate([x_diag, u_eq ],1), system.state_dimension, system.control_dimension, sim_time)
# diag_data = Data_Def(sim_time.linspace(), numpy.concatenate([y_diag, u_eq ],1)+equilibrium, system.state_dimension, system.control_dimension, sim_time)
#fig_results = plot_states([test_data, diag_data, train_data], header=['x1', 'x2', 'u'])
fig_results = plot_states([test_data, _diag_data, train_data_d], header=['x1', 'u'])

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