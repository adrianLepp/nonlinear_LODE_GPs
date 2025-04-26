


import gpytorch 
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from result_reporter.latex_exporter import plot_states, plot_weights, plot_trajectory, plot_error, save_plot_to_pdf, plot_single_states

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import get_config, Time_Def, load_system, simulate_system, downsample_data, save_everything, plot_results,  Data_Def, State_Description, get_ode_from_spline
from nonlinear_LODE_GPs.weighting import Gaussian_Weight, KL_Divergence_Weight, Epanechnikov_Weight, Mahalanobis_Distance
from nonlinear_LODE_GPs.combined_posterior import CombinedPosterior_ELODEGP

torch.set_default_dtype(torch.float64)
device = 'cpu'


local_predictions = False
SAVE = False
output_weights=False
system_name = "nonlinear_watertank"

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)

optim_steps_single = 300
optim_steps =500
learning_rate = 3

equilibrium_controls = [
    0.1, # [2.0897e-02, 1.2742e-02, 1.0000e-05]
    0.2, # [8.3588e-02, 5.0968e-02, 2.0000e-05]
    0.3, # [1.8807e-01, 1.1468e-01, 3.0000e-05]
    0.4, # [3.3435e-01, 2.0387e-01, 4.0000e-05]
    0.5, # [5.2243e-01, 3.1855e-01, 5.0000e-05]
    # 0.6, # [7.5229e-01, 4.5872e-01, 6.0000e-05]
    # 0.7, # [1.0240e+00, 6.2436e-01, 7.0000e-05]
    # 0.8,
    # 0.9,
    # 1#
]

u_ctrl = 1

x0 = torch.tensor([0.0, 0.0])

t0 = 0.0
t1 = 200.0
downsample = int(t1 / 0.1 / 50)
# downsample =50
sim_time = Time_Def(t0, t1, step=0.1)
train_time = Time_Def(t0, t1, step=sim_time.step*downsample)
test_time = Time_Def(t0, t1, step=0.1)


system = load_system(system_name)
num_tasks = system.dimension

system_matrices = []
equilibriums = []
centers = []
for i in range(len(equilibrium_controls)):
    system_matrix , x_e = system.get_ODEmatrix(equilibrium_controls[i])
    system_matrices.append(system_matrix)
    equilibriums.append(torch.tensor(x_e))
    centers.append(torch.tensor([x_e]))

#l = 44194
w_func = Gaussian_Weight(centers[0])
d = w_func.covar_dist(centers[1], w_func.center, square_dist=True)
l = d*torch.sqrt(torch.tensor(2))/4

# l = l*4

u = np.ones((sim_time.count,1)) * u_ctrl * system.param.u


_train_x, _train_y= simulate_system(system, x0, sim_time, u)
# train_x, train_y = downsample_data(_train_x, _train_y, downsample)

sim_data = Data_Def(_train_x, _train_y, system.state_dimension, system.control_dimension,train_time)
noise = torch.tensor([1e-5, 1e-5, 1e-7])
# noise = torch.tensor([0, 0, 0])
train_data = sim_data.downsample(downsample).add_noise(noise)

# train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension,train_time)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = CombinedPosterior_ELODEGP(
    train_data.time, 
    train_data.y, 
    likelihood, 
    num_tasks, 
    system_matrices, 
    equilibriums, 
    centers,
    Gaussian_Weight, #KL_Divergence_Weight, #Gaussian_Weight,  Epanechnikov_Weight, Mahalanobis_Distance
    weight_lengthscale=torch.tensor([10]),
    shared_weightscale=False,
    # additive_se=True,
    clustering=True,
    output_weights=output_weights,
    )#, 
model._optimize(optim_steps_single)
model.optimize(optim_steps, verbose=True, learning_rate=learning_rate)

test_x = test_time.linspace()
model.eval()
likelihood.eval()


#output = model(test_x)
# with torch.no_grad() and gpytorch.settings.debug(False):
#     output = likelihood(model(test_x))
#     estimate = output.mean
with torch.no_grad():
    estimate, cov, weights = model.predict(test_x)
    output = gpytorch.distributions.MultitaskMultivariateNormal(estimate, cov)
    lower, upper = output.confidence_region()


test_data = Data_Def(test_x.numpy(), estimate.detach().numpy(), system.state_dimension, system.control_dimension, test_time, uncertainty={
                'variance': output.variance,
                'lower': lower.detach().numpy(),
                'upper': upper.detach().numpy(),
                }, )

# plot_results(train_data, test_data)
# fig_results = plot_states([test_data, sim_data, train_data ], data_names=['mixture model', 'simulation', 'training data'])

state_figure = plot_single_states(
                [test_data, sim_data, train_data],
                ['local LODE-GP', "simulation", 'training' ],
                header= ['$x_1$', '$x_2$', '$u_1$'], 
                yLabel=['$x_1$ - fill level ($m$)', '$x_2$ - fill level ($m$)', '$u_1$ - flow rate ($m^3/s$) '],
                line_styles = ['-', '--', '.'],
                # colors = [i,5]               
)


weight_plot = plot_weights(test_x, weights)


states = State_Description(
    equilibrium=equilibriums[-1],
    # equilibrium=torch.stack(equilibriums), 
    init=x0, 
    min=None, max=None)

# _, _ = get_ode_from_spline(system, np.maximum(test_data.y, 0), test_data.time)


error_gp = Data_Def(test_data.time, abs(test_data.y - sim_data.y.numpy()), system.state_dimension, system.control_dimension) 
rmse_gp = np.sqrt(mean_squared_error(sim_data.y.numpy(), test_data.y))
std_gp = np.std(error_gp.y)

print(f"GP Model RMSE: {rmse_gp}, Standard Deviation: {std_gp}")


error_data_gp = error_gp.to_report_data()


# fig_error = plot_error(error_data_gp, header=['x1', 'x2', 'u1'], uncertainty = None)# output.variance

err_figure = plot_single_states(
                [error_gp],
                ['local LODE-GP'],
                header= ['$x_1$', '$x_2$', '$u_1$'], 
                yLabel=['$x_1$ - absolute error ($m$)', '$x_2$ - absolute error ($m$)', '$u_1$ - absolute error ($m^3/s$) '],
                line_styles = ['-'],
                # colors = [i,5]               
)



equilibriums = [torch.stack(equilibriums)[:,0], torch.stack(equilibriums)[:,1]]

if output_weights is True:
    centers = None
else:
    centers = [torch.zeros_like(equilibriums[0]),torch.zeros_like(equilibriums[0])]


for i, center in enumerate(model.true_centers):
    centers[0][i] = center[0]
    centers[1][i] = center[1]

trajectory_plot = plot_trajectory(test_data, {'equilibrium points': equilibriums, 'model centers': centers})




if local_predictions:
    for i in range(len(model.models)):
        with torch.no_grad() and gpytorch.settings.debug(False):
            output = likelihood(model.models[i](test_x))
            estimate = output.mean
            lower, upper = output.confidence_region()
        train_data = Data_Def(model.train_data_subsets[i][0].numpy(), model.train_data_subsets[i][1].numpy(), system.state_dimension, system.control_dimension,train_time)
        test_data = Data_Def(test_x.numpy(), estimate.detach().numpy(), system.state_dimension, system.control_dimension, test_time, uncertainty={
                'variance': output.variance,
                'lower': lower.detach().numpy(),
                'upper': upper.detach().numpy(),
                }, )

        # plot_results(train_data, test_data)
        state_figure = plot_single_states(
                [test_data, train_data],
                ['local LODE-GP', 'training' ],
                header= ['$x_1$', '$x_2$', '$u_1$'], 
                yLabel=['$x_1$ - fill level ($m$)', '$x_2$ - fill level ($m$)', '$u_1$ - flow rate ($m^3/s$) '],
                line_styles = ['-', '.'],
                colors = [i,5]               
            )

plt.show()

if SAVE:
    config['model_id'] = MODEL_ID
    config['simulation_id'] = SIM_ID
    save_plot_to_pdf(fig_error, f'error_plot_{SIM_ID}')
    save_plot_to_pdf(fig_results, f'results_plot_{SIM_ID}')
    save_plot_to_pdf(trajectory_plot, f'trajectory_plot_{SIM_ID}')
    save_plot_to_pdf(weight_plot, f'weight_plot_{SIM_ID}')
    save_everything(
        system_name, 
        model_path, 
        config, 
        train_data, 
        test_data, 
        sim_data=sim_data, 
        init_state=states.init.numpy(), 
        system_param=states.equilibrium.numpy(), 
        model_dict=model.state_dict()
    )