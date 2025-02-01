
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from kernels import *
import torch

# ----------------------------------------------------------------------------
from helpers import *
from likelihoods import *
from masking import *
from mpc import mpc_algorithm, create_setpoints
from sum_gp import Local_GP_Sum
from weighting import Weighting_Function

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False
CONFIG_FILE = 'config.json'

system_name = "nonlinear_watertank"

print("\n----------------------------------------------------------------------------------\n")
try:
    with open(CONFIG_FILE,"r") as f:
        config = json.load(f)
        model_dir=config['model_dir']
        data_dir=config['data_dir']
        model_name = config['model_name']

        if SAVE:
            global SIM_ID, MODEL_ID
            SIM_ID = config['simulation_id'] + 1
            MODEL_ID = config['model_id'] + 1

            name =  '_' + model_name + "_" + 'mpc' + "_" + system_name
            model_path = f'{model_dir}/{str(MODEL_ID)}{name}.pth'
        else: 
            SIM_ID = -1
            MODEL_ID = -1
            model_path = f'{model_dir}/{model_name}.pth'

    
    print(f"simulate {system_name}")

    if SAVE:
        print(f"save model with model id {MODEL_ID}")
        print(f"save data with data id {SIM_ID}")
    
except:
    print("No config file found. Data and model will not be saved.")
print("\n----------------------------------------------------------------------------------\n")




init_noise = [1e-8, 1e-8, 1e-12]
target_noise = [1e-7, 1e-7, 1e-11]# [1e-8, 1e-8, 1e-12]#


equilibrium_controls = [
    0.1, 
    0.2, 
    0.3, 
    0.4, 
    # 0.5
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

l=1
w_func = Weighting_Function(centers[0],l)
d = w_func.covar_dist(centers[1], w_func.center, square_dist=True)
l = d*torch.sqrt(torch.tensor(2))/8

# soft constraints for states
#x_min = torch.tensor([system.x_min[0],system.x_min[1], x_e[2]])

if reference_strategie['soft_constraints'] == 'state_limit':
    x_min = torch.tensor(system.x_min)
    x_max = torch.tensor(system.x_max)
elif reference_strategie['soft_constraints'] == 'equilibrium':
    #constraint_factor = 1.1
    x_min = x_e * 0.9
    #x_min =torch.cat((x_e[0:system.state_dimension] * 0.9, x_e[system.state_dimension::] * 0.5),0)
    x_max = x_e * 1.1
    #x_max =torch.cat((x_e[0:system.state_dimension] * 1.1, x_e[system.state_dimension::] * 2),0)

#x_min[2] = x_e[2]
states = State_Description(init=state_start, target=state_end, min=x_min, max=x_max)

with gpytorch.settings.observation_nan_policy('mask'):


    train_y, train_x, task_noise = create_setpoints(reference_strategie, control_time, states)
    likelihood = FixedTaskNoiseMultitaskLikelihood(num_tasks=num_tasks, noise=torch.tensor([1e-8,1e-8]), rank=num_tasks, has_task_noise=True, task_noise=task_noise)
    
    # works
    # noise=1e-8
    # task_noise = torch.diag(torch.tensor(init_noise, requires_grad=False))
    # covar_factor = torch.eye(num_tasks, requires_grad=False)
    # likelihood.initialize(task_noise_covar=task_noise)
    # likelihood.task_noise_covar_factor.requires_grad = False

    #likelihood.initialize(task_noise_covar=torch.eye(num_tasks, requires_grad=False)*noise)
    #likelihood.initialize(task_noise_covar_factor=covar_factor)
    # likelihood.task_noise_covar.requires_grad = Falsev ar.detach()
    # likelihood.task_noise_covar = likelihood.task_noise_covar.detach()


    #model = Weighted_Sum_GP(train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=l)
    model = Local_GP_Sum(train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=l)
    model.optimize(optim_steps)

    if hyperparameters is not None:
        for key, value in hyperparameters.items():
            if hasattr(model.covar_module.model_parameters, key):
                setattr(model.covar_module.model_parameters, key, torch.nn.Parameter(torch.tensor(value), requires_grad=False))
            else:
                print(f'Hyperparameter {key} not found in model')

    print("\n----------------------------------------------------------------------------------\n")
    print('Trained model parameters:')
    named_parameters = list(model.named_parameters())
    param_conversion = torch.nn.Softplus()

    for j in range(len(named_parameters)):
        print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
    print("\n----------------------------------------------------------------------------------\n")

    sim_data, ref_data, lode_data = mpc_algorithm(system, model, states, reference_strategie,  control_time, sim_time, optim_steps)#, plot_single_steps=True


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

control_mean =  mean(sim_data.y[:,2])#0:control_time.count+1

print(f"mean Control: {control_mean}")
print(f"Control error: {control_err}")
print(f"Constraint violation: {constraint_viol}")


plot_results(ref_data, lode_data, sim_data)

if SAVE:
    torch.save(model.state_dict(), model_path)
    with open(CONFIG_FILE,"w") as f:
        config['model_id'] = MODEL_ID
        config['simulation_id'] = SIM_ID
        json.dump(config, f)
    add_modelConfig(MODEL_ID, system_name, states.init.numpy(), states.equilibrium.numpy(), control_time.start, control_time.end, control_time.step)

    add_training_data(MODEL_ID, ref_data.time, ref_data.y)

    add_simulationConfig(SIM_ID, MODEL_ID, system_name, states.init.numpy(), states.equilibrium.numpy(), sim_time.start, sim_time.end, sim_time.step, [control_err, constraint_viol])

    add_simulation_data(SIM_ID, lode_data.time, lode_data.y)

    add_reference_data(SIM_ID, 'nonlinear', sim_data.time, sim_data.y)

    print(f"save model with model id {MODEL_ID}")
    print(f"save data with data id {SIM_ID}")

