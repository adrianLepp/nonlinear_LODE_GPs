import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from  lodegp import LODEGP, optimize_gp
from helpers import *
from likelihoods import FixedTaskNoiseMultitaskLikelihood

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False
CONFIG_FILE = 'config.json'

system_name = "nonlinear_threetank"

with open(CONFIG_FILE,"r") as f:
    config = json.load(f)
    model_dir=config['model_dir']
    data_dir=config['data_dir']
    model_name = config['model_name']

    if SAVE:
        SIM_ID = config['simulation_id'] + 1
        MODEL_ID = config['model_id'] + 1

        name =  '_' + model_name + "_" + system_name
        model_path = f'{model_dir}/{str(MODEL_ID)}{name}.pth'

print("\n----------------------------------------------------------------------------------\n")
print(f"simulate {system_name}")

if SAVE:
    print(f"save model with model id {MODEL_ID}")
    print(f"save data with data id {SIM_ID}")
print("\n----------------------------------------------------------------------------------\n")

# ----------------------------------------------------------------------------


optim_steps = 300


train_time = Time_Def(0, 100, step=1)
test_time = Time_Def(train_time.start, 2*train_time.end, step=0.1)

u_1 = 0.1
u_2 = 0.2


system = load_system(system_name)

num_tasks = system.dimension



_ , x0 = system.get_ODEmatrix(u_1)
x_0 = np.array(x0)

system_matrix , equilibrium = system.get_ODEmatrix(u_2)
x_e = np.array(equilibrium)

def create_trajectory(version:int):
    '''
    version:
    - 1: one start and one target point
    - 2: smooth transition from start to target point
    '''
    if version == 1:
        trajectory = [x_0, x_e]
        trajectory_time = [train_time.start, train_time.end]
    elif version == 2:
        t_factor = 10
        gain_factor = 10

        trajectory = [x_0, x0]
        trajectory_time = [train_time.start - t_factor*2, train_time.start - t_factor]

        for i in range(10):
            trajectory_time.append(train_time.start + i * t_factor)
            trajectory.append(x_0 + (x_e - x_0) * i / gain_factor)
            #trajectory[i][system.state_dimension:] = [0.5 * system.param.u]

        for i in range(5):
            trajectory_time.append(train_time.end + i * t_factor)
            trajectory.append(x_e)
    else:
        raise ValueError("Version not supported")
    
    return trajectory, trajectory_time


trajectory, trajectory_time = create_trajectory(1)


train_x = torch.tensor(trajectory_time)
train_y = torch.tensor(trajectory) - torch.tensor(x_e)
train_noise = torch.tensor([1e-8, 1e-8])

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
likelihood2 = FixedTaskNoiseMultitaskLikelihood(num_tasks=num_tasks ,noise=train_noise,rank=0)
model = LODEGP(train_x, train_y, likelihood2, num_tasks, system_matrix)

model_fit = false

cnt = 0
while cnt < 3:
    optimize_gp(model,optim_steps)
    # create reference and control input
    test_x = create_test_inputs(test_time.count, test_time.step, test_time.start, test_time.end, 1)
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        output = likelihood(model(test_x))
    
    #TODO: check error of the model
    # reference = train_y[:,0:system.state_dimension]
    # state_pred = output.mean[:,0:system.state_dimension]
    # error = []
    # for i in range(reference.size(0)):
    #     if train_x[i] < test_time.start:    
    #         continue
    #     idx = floor((train_x[i] / test_time.step).item())
    #     error.append(torch.norm(reference[i] - state_pred[idx]))

    train_data = Data_Def(train_x.numpy(), train_y.numpy() + x_e, system.state_dimension, system.control_dimension)
    test_data = Data_Def(test_x.numpy(), output.mean.numpy() + x_e, system.state_dimension, system.control_dimension)

    u_ref = test_data.y[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()
    ref_x, ref_y= simulate_system(system, x_0[0:system.state_dimension], test_time.start, test_time.end, test_time.count, u_ref, linear=False)
    ref_data = Data_Def(ref_x.numpy(), ref_y.numpy(), system.state_dimension, system.control_dimension)

    plot_results(train_data, test_data, ref_data)

    #change parameters
    # model.likelihood.task_noises
    # model.likelihood.noise   
    cnt += 1  
    
    lengthscale = model.covar_module.model_parameters.lengthscale_3
    model.covar_module.model_parameters.lengthscale_3 = torch.nn.Parameter(lengthscale *0.9, requires_grad=False)

    # signal_variance = model.covar_module.model_parameters.signal_variance_3
    # model.covar_module.model_parameters.signal_variance_3 = torch.nn.Parameter(abs(signal_variance), requires_grad=False)
        
    
    # model_fit = true
    # if model_fit:
    #     break
    # else:
    #     #TODO: change control input
    #     model.set_train_data(train_x, train_y, strict=False)

# ---------------------------------------------------------------------------

# train_data = Data_Def(train_x.numpy(), train_y.numpy() + x_e, system.state_dimension, system.control_dimension)
# test_data = Data_Def(test_x.numpy(), output.mean.numpy() + x_e, system.state_dimension, system.control_dimension)

# u_ref = test_data.y[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()
# ref_x, ref_y= simulate_system(system, x_0[0:system.state_dimension], test_time.start, test_time.end, test_time.count, u_ref, linear=False)
# ref_data = Data_Def(ref_x.numpy(), ref_y.numpy(), system.state_dimension, system.control_dimension)

# plot_results(train_data, test_data, ref_data)


if SAVE:
    #error = np.mean(ode_error_list, axis=1)
    save_results(model, equilibrium, x0, SIM_ID, MODEL_ID, CONFIG_FILE, system_name, config, model_path, train_data, test_data, train_time, test_time, ref_data=ref_data, linear=False) 