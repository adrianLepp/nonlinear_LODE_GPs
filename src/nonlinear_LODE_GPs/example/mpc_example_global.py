import gpytorch 
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from result_reporter.latex_exporter import create_mpc_plot

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import get_config, Time_Def, load_system, simulate_system, downsample_data, save_everything, plot_results,  Data_Def, State_Description, get_ode_from_spline
from nonlinear_LODE_GPs.weighting import Gaussian_Weight, KL_Divergence_Weight, Epanechnikov_Weight, Mahalanobis_Distance
from nonlinear_LODE_GPs.combined_posterior import CombinedPosterior_ELODEGP
from nonlinear_LODE_GPs.likelihoods import FixedTaskNoiseMultitaskLikelihood
from nonlinear_LODE_GPs.mpc import mpc_algorithm, pretrain, optimize_mpc_gp, create_setpoints

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False

system_name = "nonlinear_watertank"

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)


init_noise = [1e-8, 1e-8, 1e-12]
target_noise = [1e-7, 1e-7, 1e-11]# [1e-8, 1e-8, 1e-12]#


equilibrium_controls = [
    0.1, 
    0.2, 
    0.3, 
    0.4, 
    0.5
]

weight_lengthscales = [
    53,
    68,
    249,
    503,
    507
]

start = 1
end = 3

# Reference
reference_strategie = {
    'target': True,
    'constraints' : 10,
    'past-values' : 0,
    'init_noise' : init_noise,
    'target_noise' : target_noise,
    'soft_constraints' : 'state_limit' # 'state_limit' or 'equilibrium'
}

# TIME
t = 200

control_time = Time_Def(
    0, 
    t, 
    step=1
)#* dt_step

sim_time = Time_Def(
    0, 
    t, 
    step=0.1
)

# GP settings
optim_steps = 0
pretrain_steps = 1
hyperparameters = {
    # 'lengthscale_2': 3.6731,
    #'signal_variance_2': 0.1, # negative
}




# ----------------------------------------------------------------------------

system = load_system(system_name)
num_tasks = system.dimension
# Equilibrium values for the system

system_matrices = []
equilibriums = []
centers = []
for i in range(len(equilibrium_controls)):
    system_matrix , x_e = system.get_ODEmatrix(equilibrium_controls[i])
    system_matrices.append(system_matrix)
    equilibriums.append(torch.tensor(x_e))
    centers.append(torch.tensor([x_e]))

state_start = equilibriums[start]
# state_end = equilibriums[end]
state_end = torch.tensor([0.3, 0.3, torch.nan])#torch.nan 1e-5



if reference_strategie['soft_constraints'] == 'state_limit':
    x_min = torch.tensor(system.x_min)
    x_max = torch.tensor(system.x_max)
elif reference_strategie['soft_constraints'] == 'equilibrium':
    x_min = x_e * 0.9
    x_max = x_e * 1.1

states = State_Description(init=state_start, target=state_end, min=x_min, max=x_max)

with gpytorch.settings.observation_nan_policy('mask'):

    train_y, train_x, task_noise = create_setpoints(reference_strategie, control_time, states)
    likelihood = FixedTaskNoiseMultitaskLikelihood(num_tasks=num_tasks, noise=torch.tensor([1e-8,1e-8]), rank=num_tasks, has_task_noise=True, task_noise=task_noise)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
    model = CombinedPosterior_ELODEGP(
        train_x, 
        train_y, 
        likelihood, 
        num_tasks, 
        system_matrices, 
        equilibriums, 
        centers,
        Mahalanobis_Distance, #KL_Divergence_Weight, #Gaussian_Weight,  Epanechnikov_Weight, Mahalanobis_Distance
        weight_lengthscale=weight_lengthscales,
        shared_weightscale=False,
        # additive_se=True,
        clustering=False,
        output_weights=True,
        )#, 
    

    model._optimize(optim_steps)
    model.optimize(0)
    model.eval()

    sim_data, ref_data, lode_data, _, _ = mpc_algorithm(system, model, states, reference_strategie,  control_time, sim_time, optim_steps)#, plot_single_steps=True

# calc error
def mse_mean(mean, ref, indx=None):
    if indx is None:
        return torch.mean(torch.square(mean - ref))
    else:
        return torch.mean(torch.square((mean - ref)[indx]))

def constr_viol(mean, ub, lb, indx=None):
    if indx is None:
        return torch.mean(torch.relu(mean - ub) + torch.relu(lb - mean))
    else:
        return torch.mean(torch.relu((mean - ub)[indx]) + torch.relu((lb - mean)[indx]))
    

# halfCount = int(t_end/dt_step)+1

# fullCount = sim_data.time.shape[0]


constraint_viol = constr_viol(
    torch.tensor(sim_data.y), 
    x_max.clone().detach().reshape(1, -1), 
    x_min.clone().detach().reshape(1, -1)
)
control_err = mse_mean(
    torch.tensor(sim_data.y[:,0:system.control_dimension]),
    torch.tile(states.target[0:system.control_dimension].clone().detach(), (sim_time.count+1, 1))
    #torch.zeros_like(torch.tensor(lode_data))
)

control_mean =  np.mean(sim_data.y[:,2])#0:control_time.count+1

print(f"mean Control: {control_mean}")
print(f"Control error: {control_err}")
print(f"Constraint violation: {constraint_viol}")

reference_data = {
    'time': sim_data.time,
    'f1': sim_data.y[:,0],
    'f2': sim_data.y[:,1],
    'f3': sim_data.y[:,2],
    }

fig = create_mpc_plot(None, None, ['x1','x2', 'u'], 'Time ($\mathrm{s})$', 'Water Level ($\mathrm{m}$)', reference_data, x_e=[states.target[0],states.target[1],states.target[2]], close_constraint=False)
plt.show()


plot_results(ref_data, lode_data, sim_data)


'''
if SAVE:
    torch.save(model.state_dict(), model_path)

    config['model_id'] = MODEL_ID
    config['simulation_id'] = SIM_ID
    save_config(config)

    add_modelConfig(MODEL_ID, system_name, states.init.numpy(), states.equilibrium.numpy(), control_time.start, control_time.end, control_time.step)
    add_training_data(MODEL_ID, ref_data.time, ref_data.y)
    add_simulationConfig(SIM_ID, MODEL_ID, system_name, states.init.numpy(), states.equilibrium.numpy(), sim_time.start, sim_time.end, sim_time.step, [control_err, constraint_viol])
    add_simulation_data(SIM_ID, lode_data.time, lode_data.y)
    add_reference_data(SIM_ID, 'nonlinear', sim_data.time, sim_data.y)

    print(f"save model with model id {MODEL_ID}")
    print(f"save data with data id {SIM_ID}")
'''
