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
from nonlinear_LODE_GPs.gp import GP, Linearizing_Control
from mpl_toolkits.mplot3d import Axes3D

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False


system_name = "inverted_pendulum"

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)

t  = 20
optim_steps = 300
downsample = 10
sim_time = Time_Def(0, t, step=0.01)
train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t, step=0.01)

a0 = 0
a1 = 0
v = 1
ccf_param = [a0, a1, v]
system = load_system(system_name, a0=0, a1=0, v=1)

num_tasks = system.dimension

ode_matrix  = system.get_ODEmatrix()
#system_matrix = system.get_parameterized_ODEmatrix()

#D, U, V = system_matrix.smith_form()

x_0 = np.array([ np.pi/2 , 0 ,0]) #+ 0.1 TODO
u = 1 #TODO

states = State_Description(init=torch.tensor(x_0))


#u = np.linspace(u_rel * system.param.u, u_rel * system.param.u, train_time.count)
u = np.ones((sim_time.count,1)) * u
# u[0]=1

u[0:100] = 0
u[1000::] = 0


_train_x, _train_y= simulate_system(system, x_0[0:system.state_dimension], sim_time, u)
train_x, train_y = downsample_data(_train_x, _train_y, downsample)

# for i in range(len(train_x)):
#     train_y[i,-1] = system.get_latent_control(u[i].squeeze(), train_y[i,0:2])

# for i in range(len(_train_x)):
#     _train_y[i,-1] = system.get_latent_control(u[i].squeeze(), _train_y[i,0:2])

train_y[:,-1] = torch.nan #torch.tensor(np.nan)

ref_x, ref_y= simulate_system(system, x_0[0:system.state_dimension], sim_time, u, linear=True)
# for i in range(len(ref_x)):
#     ref_y[i,-1] = system.get_latent_control(u[i].squeeze(), ref_y[i,0:2])


with gpytorch.settings.observation_nan_policy('mask'):

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks,
            has_global_noise=False, 
            has_task_noise=True,
            noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
    )

    model = LODEGP(train_x, train_y, likelihood, num_tasks, ode_matrix)

    training_loss = optimize_gp(model,optim_steps)

    test_x = test_time.linspace() #TODO

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        output = likelihood(model(test_x))
        lower, upper = output.confidence_region()
        
# _, _ = get_ode_from_spline(system, output.mean, test_x)

uncertainty = {
    'lower': lower.numpy(),
    'upper': upper.numpy()
}

ref_x, ref_y= simulate_system(system, x_0[0:system.state_dimension], sim_time, u, linear=True)

test_data = Data_Def(test_x.numpy(), output.mean.numpy(), system.state_dimension, system.control_dimension, test_time)#, uncertainty
ref_data = Data_Def(ref_x.numpy(), system.rad_to_deg(ref_y.numpy()), system.state_dimension, system.control_dimension, sim_time)
train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension, train_time)
_train_data = Data_Def(_train_x.numpy(), _train_y.numpy(), system.state_dimension, system.control_dimension, sim_time)


# for i in range(len(test_data.time)):
#     test_data.y[i,-1] = system.get_control_from_latent(test_data.y[i,-1].squeeze(), test_data.y[i,0:2])

# for i in range(len(train_data.time)):
#     train_data.y[i,-1] = system.get_control_from_latent(train_data.y[i,-1].squeeze(), train_data.y[i,0:2])

test_data.y = system.rad_to_deg(test_data.y)
train_data.y = system.rad_to_deg(train_data.y)
_train_data.y = system.rad_to_deg(_train_data.y)

# for i in range(len(ref_data.time)):
#     ref_data.y[i,-1] = system.get_control_from_latent(ref_data.y[i,-1].squeeze(), ref_data.y[i,0:2])


fig_results = plot_states(train_data, test_data, _train_data,header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'])


# plt.show()


# PART II: get mapping (x,u)-> y_ref

_u_train_x = _train_y
_u_train_y = output.mean[:,-1]
u_train_y, u_train_x  = downsample_data(_u_train_y, _u_train_x, downsample)

u_likelihood = gpytorch.likelihoods.GaussianLikelihood()
control_gp = Linearizing_Control(u_train_x, u_train_y, u_likelihood)

model.train()
likelihood.train()



# control_gp.b_covar_module.raw_outputscale = torch.nn.Parameter(torch.tensor(0.0, requires_grad = false))
# control_gp.beta.covar_module.initialize(outputscale = torch.tensor(0.0, requires_grad = false))
control_gp.alpha.mean_module.initialize(constant = torch.tensor(0.0, requires_grad = false))

# control_gp.beta.covar_module.initialize(raw_outputscale = torch.tensor(0.0, requires_grad = false))
control_gp.alpha.mean_module.initialize(raw_constant = torch.tensor(0.0, requires_grad = false))
#This works
control_gp.alpha.mean_module.raw_constant.requires_grad = False 
# control_gp.beta.covar_module.raw_outputscale.requires_grad = False
training_loss = optimize_gp(control_gp,optim_steps)


control_gp.eval()
u_likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var(): # and gpytorch.settings.debug(False):
        u_output = u_likelihood(control_gp(_u_train_x))

        alpha_output = control_gp.alpha(_u_train_x[:,:-1])
        beta_output = control_gp.beta(_u_train_x[:,:-1])
        # u_lower, u_upper = u_output.confidence_region()


y_ref = np.zeros_like(_train_x)

for i in range(len(_train_x)):
    y_ref[i] = system.get_latent_control(u[i].squeeze(), _train_y[i,0:2].numpy())

plt.rcParams['text.usetex'] = True
plt.figure()
plt.plot(_train_x.detach().numpy(), y_ref, '--', label = r'$y_{ref}(x,u)$')
plt.plot(_train_x.detach().numpy(), u_output.mean.detach().numpy(), label = r'$y_{pred}(x,u)$')
plt.plot(_train_x.detach().numpy(), _u_train_x[:,-1].detach().numpy(), label = 'u')
plt.plot(_train_x.detach().numpy(), alpha_output.mean.detach().numpy(), '--', label = r'$\alpha_{pred}(x)$')
plt.plot(_train_x.detach().numpy(), beta_output.mean.detach().numpy(), '--', label = r'$\beta_{pred}(x)$')
plt.xlabel('Time [s]')
plt.ylabel('Force [N]')
plt.title('Control estimation inverted pendulum')
plt.legend()
# plt.show()


# plt.figure()
# plt.plot(_train_y[:,0].numpy(), y_ref, '.' , label = 'x_1')
# plt.plot(_train_y[:,1].numpy(), y_ref, '.', label = 'x_2')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x = _train_y[:, 0].detach().numpy()
# y = _train_y[:, 1].detach().numpy()
# z = u_output.mean.detach().numpy()

# ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

# ax.set_xlabel('State 1')
# ax.set_ylabel('State 2')
# ax.set_zlabel('u_output.mean')

plt.show()

# save_plot_to_pdf(fig_results, f'results_plot_{SIM_ID}')

if SAVE:
    config['model_id'] = MODEL_ID
    config['simulation_id'] = SIM_ID
    # save_plot_to_pdf(fig_loss, f'loss_plot_{SIM_ID}')
    # save_plot_to_pdf(fig_error, f'error_plot_{SIM_ID}')
    save_plot_to_pdf(fig_results, f'results_plot_{SIM_ID}')
    save_everything(
        system_name, 
        model_path, 
        config, 
        train_data, 
        test_data, 
        sim_data=ref_data, 
        init_state=states.init.numpy(), 
        system_param=ccf_param, 
        model_dict=model.state_dict()
    )