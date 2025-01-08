
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
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

FEEDBACK = True

system_name = "nonlinear_watertank"

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

print("\n----------------------------------------------------------------------------------\n")
print(f"simulate {system_name}")

if SAVE:
    print(f"save model with model id {MODEL_ID}")
    print(f"save data with data id {SIM_ID}")
print("\n----------------------------------------------------------------------------------\n")

# ----------------------------------------------------------------------------


optim_steps = 0
pretrain_steps = 150

reference_strategie = {
    'target': True,
    'constraints' : 10,
    'start_noise' : 1e-8,
    'end_noise' : 1e-7,
}


dt_step = 0.1 # time step for simulation and gp model
t_end = 100
dt_control = t_end /t_end # time step for control loops. set to t_end for feedforward control in one step


u_1 = 0.2   # control input to find equilibrium where we start
u_2 = 0.3 # control input to find equilibrium where we want to end and linearize around


system = load_system(system_name)
num_tasks = system.dimension


_ , x0 = system.get_ODEmatrix(u_1)
x_0 = np.array(x0)


system_matrix , equilibrium = system.get_ODEmatrix(u_2)
x_e = np.array(equilibrium)

# soft constraints for states
x_min = np.array(system.x_min)
x_max = np.array(system.x_max)

x_min[2] = x_e[2]

states = State_Description(x_e, x_0, min=x_min, max=x_max)

#likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())

control_time = Time_Def(0, t_end, step=dt_control)#* dt_step
model, mask = pretrain(system_matrix, num_tasks, control_time, pretrain_steps, reference_strategie, states)# pretrain the system and generate gp model. eventually not necessary

x_sim, x_ref, x_lode = mpc_algorithm(system, model, states, t_end, dt_control, reference_strategie, optim_steps, dt_step=dt_step)
plot_results(x_ref, x_lode, x_sim)

#    mpc_feed_forward(test_time, x_0, x_e, model, likelihood, system, SIM_ID, MODEL_ID, model_path, model_dir, optim_steps, train_x, train_y)

