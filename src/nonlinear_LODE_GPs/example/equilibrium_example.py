import gpytorch 
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from nonlinear_LODE_GPs.kernels import *
import torch
from result_reporter.latex_exporter import plot_loss, plot_error, plot_states, save_plot_to_pdf
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import  optimize_gp, LODEGP
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.mean_modules import Equilibrium_Mean

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = True


system_name = "nonlinear_watertank"
loss_file = '../data/losses/equilibrium.csv'

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)

t  = 100
optim_steps = 300
downsample = 20 # TODO  100 50 10
sim_time = Time_Def(0, t, step=0.1)
# train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t-0, step=0.1)

noise = torch.tensor([1e-5, 1e-5, 1e-7])

u_e_rel = 0.2

u_rel = .3


system = load_system(system_name)

num_tasks = system.dimension

_ , x0 = system.get_ODEmatrix(u_e_rel)
system_matrix , equilibrium = system.get_ODEmatrix(u_e_rel)
#system_matrix = system.get_parameterized_ODEmatrix()

#D, U, V = system_matrix.smith_form()

x_0 = np.array(x0)

states = State_Description(equilibrium=torch.tensor(equilibrium), init=torch.tensor(x0))


#u = np.linspace(u_rel * system.param.u, u_rel * system.param.u, train_time.count)
u = np.ones((sim_time.count,1)) * u_rel * system.param.u

_train_x, _train_y= simulate_system(system, x_0[0:system.state_dimension], sim_time, u)
sim_data = Data_Def(_train_x, _train_y,system.state_dimension, system.control_dimension, sim_time)
train_data = sim_data.downsample(downsample).add_noise(noise)
# train_x, train_y = downsample_data(_train_x, _train_y, downsample)

# %% train


with gpytorch.settings.observation_nan_policy('mask'):
    # train_y[1:-1,0:2] = torch.tensor(np.nan)
    # train_data.y[1:-1,0:2] = torch.nan

    #train_y[:,2] = torch.tensor(np.nan)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks, 
        # rank=num_tasks, 
        #noise_constraint=gpytorch.constraints.Positive(), 
        has_global_noise=False, 
        # has_task_noise=True,
        noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
    )
    # likelihood.task_noises = torch.tensor(noise, requires_grad=False)

    # noise=1e-8
    #init_noise = [1e-3, 1e-3, 1e-6]
    #task_noise = torch.diag(torch.tensor(init_noise, requires_grad=False))
    # covar_factor = torch.eye(num_tasks, requires_grad=False)
    
    # likelihood.initialize(task_noises=torch.tensor(noise, requires_grad=False))
    # likelihood.raw_task_noises.requires_grad = False

    # likelihood.task_noise_covar_factor.requires_grad = False

    #likelihood.task_noises= likelihood.task_noises.detach()
    #likelihood.raw_task_noises.requires_grad = False

    mean_module = Equilibrium_Mean(equilibrium, num_tasks)
    model = LODEGP(train_data.time, train_data.y, likelihood, num_tasks, system_matrix, mean_module) #system.state_var, system.control_var

    training_loss = optimize_gp(model,optim_steps)

    # %% test

    #test_x = create_test_inputs(test_time, 1)
    test_x = test_time.linspace() #TODO

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        output = likelihood(model(test_x))
        lower, upper = output.confidence_region()
        
    _, _ = get_ode_from_spline(system, output.mean, test_x)

uncertainty = {
    'lower': lower.numpy(),
    'upper': upper.numpy()
}

x0_e = x0 - np.array(equilibrium)
ref_x, ref_y= simulate_system(system, x0_e[0:system.state_dimension], sim_time, u-equilibrium[-1], linear=True)
ref_data = Data_Def(ref_x.numpy(), ref_y.numpy() + np.array(equilibrium), system.state_dimension, system.control_dimension, sim_time)

_, _ = get_ode_from_spline(system, ref_data.y, ref_data.time)


test_data = Data_Def(test_x.numpy(), output.mean.numpy(), system.state_dimension, system.control_dimension, test_time, uncertainty)
# train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension, train_time)


error_gp = Data_Def(test_data.time, abs(test_data.y - _train_y.numpy()), system.state_dimension, system.control_dimension) 
error_de = Data_Def(ref_data.time, abs(ref_data.y - _train_y.numpy()), system.state_dimension, system.control_dimension) 
# error_de = abs(ref_data.y - _train_y.numpy())

# Calculate RMSE for GP model
rmse_gp = np.sqrt(mean_squared_error(_train_y.numpy(), test_data.y))
std_gp = np.std(error_gp.y)

# Calculate RMSE for DE model
rmse_de = np.sqrt(mean_squared_error(_train_y.numpy(), ref_data.y))
std_de = np.std(error_de.y)

print(f"GP Model RMSE: {rmse_gp}, Standard Deviation: {std_gp}")
print(f"DE Model RMSE: {rmse_de}, Standard Deviation: {std_de}")

error_data_gp = error_gp.to_report_data()
error_data_de = error_de.to_report_data()

loss_tracker = LossTracker(loss_file)

# loss_tracker.add_loss(f'D = {train_time.count}', training_loss)
fig_loss = loss_tracker.plot_losses()
# loss_tracker.to_csv()
# fig_loss = plot_loss(training_loss)

fig_error = plot_error(error_data_gp, error_data_de, ['x1', 'x2', 'u1'])
fig_results = plot_states([test_data, ref_data, train_data])

plt.show()

if SAVE:
    config['model_id'] = MODEL_ID
    config['simulation_id'] = SIM_ID
    save_plot_to_pdf(fig_loss, f'loss_plot_{SIM_ID}')
    save_plot_to_pdf(fig_error, f'error_plot_{SIM_ID}')
    save_plot_to_pdf(fig_results, f'results_plot_{SIM_ID}')
    save_everything(
        system_name, 
        model_path, 
        config, 
        train_data, 
        test_data, 
        sim_data=ref_data, 
        init_state=states.init.numpy(), 
        system_param=states.equilibrium.numpy(), 
        model_dict=model.state_dict()
    )
