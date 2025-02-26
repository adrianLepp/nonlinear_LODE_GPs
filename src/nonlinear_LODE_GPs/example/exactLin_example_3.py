import gpytorch 
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
# from nonlinear_LODE_GPs.kernels import *
import torch
from result_reporter.latex_exporter import plot_loss, plot_error, plot_states, save_plot_to_pdf
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import  optimize_gp, LODEGP
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.mean_modules import Equilibrium_Mean
from nonlinear_LODE_GPs.gp import GP, Linearizing_Control, Linearizing_Control_2
from mpl_toolkits.mplot3d import Axes3D

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False


system_name = "inverted_pendulum"

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)

t  = 20
optim_steps = 100
downsample = 10
sim_time = Time_Def(0, t, step=0.01)
train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t, step=0.01)
system = load_system(system_name, a0=0, a1=0, v=1)

num_tasks = system.dimension

ode_matrix  = system.get_ODEmatrix()

x_0 = np.array([ np.pi/2 - 0.1 , 0 ,0]) #+ 0.1 TODO
x_1 = np.array([ np.pi/2 , 0 ,0])

states_0 = State_Description(init=torch.tensor(x_0))
states_1 = State_Description(init=torch.tensor(x_1))

u_0 = np.zeros((sim_time.count,1))
u_1 = np.ones((sim_time.count,1)) 


_train_x_0, _train_y_0= simulate_system(system, x_0[0:system.state_dimension], sim_time, u_0)
train_x_0, train_y_0 = downsample_data(_train_x_0, _train_y_0, downsample)

_train_x_1, _train_y_1= simulate_system(system, x_1[0:system.state_dimension], sim_time, u_1)
train_x_1, train_y_1 = downsample_data(_train_x_1, _train_y_1, downsample)

train_data_0 = Data_Def(train_x_0.numpy(), train_y_0.numpy(), system.state_dimension, system.control_dimension, train_time)
train_data_1 = Data_Def(train_x_1.numpy(), train_y_1.numpy(), system.state_dimension, system.control_dimension, train_time)

# train_data_0.y = system.rad_to_deg(train_data_0.y)
# train_data_1.y = system.rad_to_deg(train_data_1.y) FIXME

plot_states(
    [train_data_0, train_data_1],
    data_names = ['uncontrolled', 'const control'], 
    header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
    title = f'Inverted Pendulum Training Data'
    )
# plt.show()
# train_y = torch.cat((train_y_0, train_y_1), dim=0)
# train_x = torch.cat((train_x_0, train_x_1), dim=0)

train_y_0[:,-1] = torch.nan #torch.tensor(np.nan)
train_y_1[:,-1] = torch.nan #torch.tensor(np.nan)


with gpytorch.settings.observation_nan_policy('mask'):
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks,
            has_global_noise=False, 
            has_task_noise=True,
            noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
    )

    model = LODEGP(train_x_0, train_y_0, likelihood, num_tasks, ode_matrix)

    training_loss = optimize_gp(model,optim_steps)

    test_x = test_time.linspace() #TODO

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        output_0 = likelihood(model(test_x))

    test_data_0 = Data_Def(test_x.numpy(), output_0.mean.numpy(), system.state_dimension, system.control_dimension, test_time)#, uncertainty

    model.set_train_data(train_x_1, train_y_1, strict=False)
    training_loss = optimize_gp(model,optim_steps)

    test_x = test_time.linspace() #TODO

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        output_1 = likelihood(model(test_x))

test_data_1 = Data_Def(test_x.numpy(), output_1.mean.numpy(), system.state_dimension, system.control_dimension, test_time)#, uncertainty
# train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension, train_time)

# test_data.y = system.rad_to_deg(test_data.y)
# train_data.y = system.rad_to_deg(train_data.y)

fig_results = plot_states(
    [test_data_0, test_data_1],
    data_names = ['uncontrolled', 'const control'], 
    header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
    title = f'Inverted Pendulum LODE GP.'
    )

# plt.show()

# ----------------------------------------------------------------------------
# PART II: get mapping (x,u)-> y_ref
# ----------------------------------------------------------------------------

_u_train_x_0 = _train_y_0
_u_train_y_0 = output_0.mean[:,-1]
u_train_y_0, u_train_x_0  = downsample_data(_u_train_y_0, _u_train_x_0, downsample)

_u_train_x_1 = _train_y_1
_u_train_y_1 = output_1.mean[:,-1]
u_train_y_1, u_train_x_1  = downsample_data(_u_train_y_1, _u_train_x_1, downsample)

u_likelihood = gpytorch.likelihoods.GaussianLikelihood()

control_gp = Linearizing_Control_2(
     u_train_x_0[:,0:-1], 
     u_train_x_0[:,-1],
     u_train_y_0, 
     u_train_x_1[:,0:-1], 
     u_train_x_1[:,-1],
     u_train_y_1, 
     u_likelihood,
    #  system,
    #  a_0,
    #  a_1,
    #  v
     )

control_gp.optimize(optim_steps, verbose=True)

u_likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var(): # and gpytorch.settings.debug(False):
    y_ref_output_0 = control_gp.y_ref(_u_train_x_0[:,:-1], _u_train_x_0[:,-1])
    y_ref_output_1 = control_gp.y_ref(_u_train_x_1[:,:-1], _u_train_x_1[:,-1])

    alpha_0 = u_likelihood(control_gp.alpha(_u_train_x_0[:,:-1])).mean
    alpha_1 = u_likelihood(control_gp.alpha(_u_train_x_1[:,:-1])).mean
    beta_1 = u_likelihood(control_gp.beta(_u_train_x_1[:,:-1])).mean

# y_ref = np.zeros_like(_train_x)
# y_pred_2 = alpha_output.mean.detach().numpy() + beta_output.mean.detach().numpy() * _u_train_x[:,-1].detach().numpy()

# for i in range(len(_train_x)):
#     y_ref[i] = system.get_latent_control(u[i].squeeze(), _train_y[i,0:2].numpy())

plt.rcParams['text.usetex'] = True
fig, (ax1, ax2) = plt.subplots(2, 1)


ax1.plot(_train_x_0.numpy(), y_ref_output_0, label = r'$y_{ref,0}(x,u)$')
ax2.plot(_train_x_1.numpy(), y_ref_output_1, label = r'$y_{ref,1}(x,u)$')
ax2.plot(_train_x_1.numpy(), alpha_1, label = r'$\alpha_{pred,1}(x)$')
ax2.plot(_train_x_1.numpy(), beta_1, label = r'$\beta_{pred,1}(x)$')

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Force [N]')
ax1.set_title('Control estimation inverted pendulum')
ax1.grid(True)
ax1.legend()
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Force [N]')
ax2.set_title('Control estimation inverted pendulum')
ax2.grid(True)
ax2.legend()

# ----------------------------------------------------------------------------
# PART III: control system with GP
# ----------------------------------------------------------------------------

def alpha(x):
    return control_gp.alpha(torch.tensor(x).unsqueeze(0)).mean.detach().numpy()

def beta(x):
    return control_gp.beta(torch.tensor(x).unsqueeze(0)).mean.detach().numpy()


a0 = 2
a1 = 3
v = 1
system_2 = load_system(system_name, a0=a0, a1=a1, v=1)
system_2.alpha = alpha
system_2.beta = beta

y_ref_control = np.zeros((sim_time.count,1))
ts = sim_time.linspace()
sol = solve_ivp(
    system_2.stateTransition_2, 
    [sim_time.start, sim_time.end], 
    x_1[0:system.state_dimension], 
    method='RK45', 
    t_eval=ts, args=(y_ref_control.squeeze(),sim_time.step)
)
x = sol.y.transpose()

solution = []
for i in range (x.shape[1]):
    solution.append(torch.tensor(x[:,i]))
solution.append(torch.tensor(y_ref_control.squeeze()))

control_y = torch.stack(solution, -1)

for i in range(len(ts)):
    control_y[i,-1] = system.get_control_from_latent(y_ref_control[i].squeeze(), control_y[i,0:2])

control_data = Data_Def(ts.numpy(), control_y.numpy(), system.state_dimension, system.control_dimension, sim_time)

# control_data.y = system.rad_to_deg(control_data.y)
# control_data.y = system.rad_to_deg(control_data.y) FIXME

fig_results = plot_states(
    [ control_data],
    data_names = ['Sim'], 
    header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
    title = f'Inverted Pendulum GP Control: a0: {a0}, a1: {a1}, v: {v}'
    )

plt.show()

# save_plot_to_pdf(fig_results, f'results_plot_{SIM_ID}')

