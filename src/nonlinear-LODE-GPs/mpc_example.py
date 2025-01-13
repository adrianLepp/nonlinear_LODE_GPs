
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from kernels import *
import torch

# ----------------------------------------------------------------------------
from helpers import *
from likelihoods import *
from masking import *
from mpc import mpc_algorithm, pretrain

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


init_noise = [1e-8, 1e-8, 1e-8]
target_noise = [1e-7, 1e-7, 1e-11]


# Reference
reference_strategie = {
    'target': False,
    'constraints' : 10,
    'init_noise' : init_noise,
    'target_noise' : target_noise,
}

# Equilibrium values for the system
u_1 = 0.2   # control input to find equilibrium where we start
u_2 = 0.3 # control input to find equilibrium where we want to end and linearize around

# TIME
t = 400

control_time = Time_Def(
    0, 
    t, 
    step=1
)#* dt_step

sim_time = Time_Def(
    0, 
    t + 1, 
    step=0.1
)

# GP settings
optim_steps = 0
pretrain_steps = 100
hyperparameters = {
    #'lengthscale_2': 3
}


system = load_system(system_name)
num_tasks = system.dimension

_ , x0 = system.get_ODEmatrix(u_1)
x_0 = torch.tensor(x0)
system_matrix , equilibrium = system.get_ODEmatrix(u_2)
x_e = torch.tensor(equilibrium)

# soft constraints for states
#x_min = torch.tensor([system.x_min[0],system.x_min[1], x_e[2]])

x_min = torch.tensor(system.x_min)
x_max = torch.tensor(system.x_max)
constraint_factor = 1.1
x_min = torch.tensor(x_e / constraint_factor)
x_max = torch.tensor(x_e * constraint_factor)

#x_min[2] = x_e[2]
states = State_Description(x_e, x_0, min=x_min, max=x_max)

model, mask = pretrain(system_matrix, num_tasks, control_time, pretrain_steps, reference_strategie, states, hyperparameters)# pretrain the system and generate gp model. eventually not necessary
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
    torch.tensor(x_max).reshape(1, -1), 
    torch.tensor(x_min).reshape(1, -1)
)
control_err = mse_mean(
    torch.tensor(sim_data.y[:,0:system.control_dimension]),
    torch.tile(torch.tensor(states.target[0:system.control_dimension]), (sim_time.count+1, 1))
    #torch.zeros_like(torch.tensor(lode_data))
)

control_sum =  sum(sim_data.y[0:control_time.count+1,2])

print(f"Control sum: {control_sum}")
print(f"Control error: {control_err}")
print(f"Constraint violation: {constraint_viol}")


plot_results(ref_data, lode_data, sim_data)

if SAVE:
    torch.save(model.state_dict(), model_path)
    with open(CONFIG_FILE,"w") as f:
        config['model_id'] = MODEL_ID
        config['simulation_id'] = SIM_ID
        json.dump(config, f)
    add_modelConfig(MODEL_ID, system_name,  x0, equilibrium, control_time.start, control_time.end, control_time.step)

    add_training_data(MODEL_ID, ref_data.time, ref_data.y)

    add_simulationConfig(SIM_ID, MODEL_ID, system_name, x0, equilibrium, sim_time.start, sim_time.end, sim_time.step, [control_err, constraint_viol])

    add_simulation_data(SIM_ID, lode_data.time, lode_data.y)

    add_reference_data(SIM_ID, 'nonlinear', sim_data.time, sim_data.y)

