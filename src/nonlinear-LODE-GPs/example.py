import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import time
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
import json
from result_reporter.sqlite import add_modelConfig, add_simulationConfig, add_simulation_data, add_training_data, add_reference_data
# ----------------------------------------------------------------------------
from  lodegp import *
from helpers import *

CONTROL = False
REFERENCE = True
linear = True

SIMULATION_ID:int = -1 
MODEL_ID:int =  -1 
CONFIG_FILE = 'config.json'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example simulationfor lodegp"
    )
    parser.add_argument("--system", required=True, type=str, help="system to simulate")
    parser.add_argument("--save",  type=bool, help="save data", default=False)#required=True,
    parser.add_argument("--model", required=True, type=str, default="None", help="load or save model")
    parser.add_argument("--model_id", type=int, help="id of the model to load")
    #parser.add_argument("--num3", required=True, type=int)

    args = parser.parse_args()
    system_name:str = args.system
    SAVE_DATA = args.save
    if args.model == 'load':
        LOAD_MODEL = True
        SAVE_MODEL = False
        MODEL_ID = args.model_id
    elif args.model == 'save':
        LOAD_MODEL = False
        SAVE_MODEL = True
    elif args.model == 'control':
        LOAD_MODEL = False
        SAVE_MODEL = False
        CONTROL = True
    else:
        LOAD_MODEL = False
        SAVE_MODEL = False

    with open(CONFIG_FILE,"r") as f:
        config = json.load(f)
        model_dir=config['model_dir']
        data_dir=config['data_dir']
        model_name = config['model_name']

        if SAVE_DATA:
            SIMULATION_ID = config['simulation_id'] + 1
        if SAVE_MODEL:
            MODEL_ID = config['model_id'] + 1

    #config = pd.read_json('config.json', lines=True)#


    print("\n----------------------------------------------------------------------------------\n")
    print(f"simulate {system_name}")

    if LOAD_MODEL:
        print(f"load model with model id {MODEL_ID}")
    if SAVE_MODEL:
        print(f"save model with model id {MODEL_ID}")
    if SAVE_DATA is True:
        print(f"save data with data id {SIMULATION_ID}")

    print("\n----------------------------------------------------------------------------------\n")

    # %% config

    #torch.set_default_tensor_type(torch.DoubleTensor)
    torch.set_default_dtype(torch.float64)
    device = 'cpu'

    name =  '_' + model_name + "_" + system_name
    model_path = f'{model_dir}/{str(MODEL_ID)}{name}.pth'# model_dir + str(MODEL_ID) + name + ".pth"
    data_path = f'{data_dir}/{str(SIMULATION_ID)}{name}.csv'

    # %% setup

    optim_steps =300

    train_time = Time_Def(0, 200, 200)
    test_time = Time_Def(0, 200, 2000)
    
    
    u_r_rel = 0.2

    u_rel = 0.4

system = load_system(system_name)
num_tasks = system.dimension
  
if LOAD_MODEL:
    train_x, train_y, x0, x_r = load_training_data(MODEL_ID)    
else:
    system_matrix , equilibrium = system.get_ODEmatrix(u_r_rel)
    x_r = np.array(equilibrium)
    x0 = x_r

    u = np.linspace(u_rel * system.param.u, u_rel * system.param.u, train_time.count)

    train_x, train_y= simulate_system(system, x0[0:system.state_dimension], train_time.start, train_time.end, train_time.count, u)
    train_y = train_y - torch.tensor(x_r)




# %% train

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix)

if LOAD_MODEL:
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    optimize_gp(model,optim_steps)

# %% test

test_x = create_test_inputs(test_time.count, test_time.step, test_time.start, test_time.end, 1)

model.eval()
likelihood.eval()


#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))

# %% recreate ODE from splines of prediction

ode, ode_error_list = get_ode_from_spline(model, system, output, test_x)

# train_x_np = train_x.numpy()
# train_y_np = train_y.numpy() + x_r
# test_x_np = test_x.numpy()
# estimation = output.mean.numpy() + x_r

train_data = Data_Def(train_x.numpy(), train_y.numpy() + x_r, system.state_dimension, system.control_dimension)
test_data = Data_Def(test_x.numpy(), output.mean.numpy() + x_r, system.state_dimension, system.control_dimension)

# variance = output.variance.numpy()
# std = output.stddev.diag_embed().numpy()
# lower, upper = output.confidence_region()

if REFERENCE:
    #u_ref = estimation[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()
    if linear:
        u_ref = np.linspace(u_rel * system.param.u, u_rel * system.param.u, test_time.count)
        u_ref = u_ref - x_r[system.state_dimension:system.state_dimension+system.control_dimension]
        x0_e = x0 - x_r

        ref_x, ref_y= simulate_system(system, x0_e[0:system.state_dimension], test_time.start, test_time.end, test_time.count, u_ref, linear=True)
        ref_data = Data_Def(ref_x.numpy(), ref_y.numpy() + x_r, system.state_dimension, system.control_dimension)
    else:
        #u_ref = np.linspace(u_rel * system.param.u, u_rel * system.param.u, test_count)
        u_ref = test_data.y[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()
        ref_x, ref_y= simulate_system(system, x0[0:system.state_dimension], test_time.start, test_time.end, test_time.count, u_ref, linear=False)

        ref_data = Data_Def(ref_x.numpy(), ref_y.numpy(), system.state_dimension, system.control_dimension)

plot_results(train_data, test_data, ref_data)


# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()

# for i in range(system.state_dimension):
#     color = f'C{i}'
#     ax1.plot(train_x_np, train_y_np[:, i], '.', color=color, label=f'x{i+1}_train')    
#     if REFERENCE:
#         ax1.plot(ref_x_np, ref_y_np[:, i], '--', color=color, label=f'x{i+1}_ref')
#     ax1.plot(test_x_np, estimation[:, i], color=color, label=f'x{i+1}_est', alpha=0.5)

# for i in range(system.control_dimension):
#     idx = system.state_dimension + i
#     color = f'C{idx}'
#     ax2.plot(train_x_np, train_y_np[:, idx], '.', color=color, label=f'x{idx+1}_train')
#     if REFERENCE:
#         ax2.plot(ref_x_np, ref_y_np[:, idx], '--', color=color, label=f'x{idx+1}_ref')
#     ax2.plot(test_x_np, estimation[:, idx], color=color, label=f'x{idx+1}est', alpha=0.5)

# ax2.tick_params(axis='y', labelcolor=color)
# ax1.legend()
# ax2.legend()
# ax1.grid(True)

# plt.show()

system_param = equilibrium
if SAVE_MODEL:
    torch.save(model.state_dict(), model_path)
    with open(CONFIG_FILE,"w") as f:
        config['model_id'] = MODEL_ID
        json.dump(config, f)
    add_modelConfig(MODEL_ID, system_name,  x0, system_param, train_time.start, train_time.end, train_time.step)
    add_training_data(MODEL_ID, train_data.time, train_data.y)

if SAVE_DATA:
    # data = {'time': test_x_np}
    # for i in range(len(estimation[1])):
    #     data[f'f{i+1}'] = estimation[:, i]
    # saveDataToCsv(data_path, data, overwrite=True)

    add_simulationConfig(SIMULATION_ID, MODEL_ID, system_name, x0, system_param, test_time.start, test_time.end, test_time.step, np.mean(ode_error_list, axis=1))
    add_simulation_data(SIMULATION_ID, test_data.time, test_data.y)

    if REFERENCE:
        if linear:
            type = 'linear'
        else:
            type = 'nonlinear'
        add_reference_data(SIMULATION_ID, type, ref_data.time, ref_data.y)

    # info = collectMetaInformation(SIMULATION_ID, model_name, system_name, model.named_parameters(), np.mean(ode_error_list))
    # saveSettingsToJson(data_path, info, overwrite=True)
    
    with open(CONFIG_FILE,"w") as f:
        config['simulation_id'] = SIMULATION_ID
        json.dump(config, f)

