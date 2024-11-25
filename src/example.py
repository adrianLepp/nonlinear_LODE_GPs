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
from scipy.integrate import solve_ivp
import json
import pandas as pd
from result_reporter.data_saver import saveDataToCsv
from result_reporter.sqlite import add_modelConfig, add_simulationConfig, add_simulation_data, add_training_data, get_training_data, get_model_config
# ----------------------------------------------------------------------------
from  lodegp import LODEGP 
from helpers import load_system, load_training_data, simulate_system, create_test_inputs

CONTROL = True

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

    num_data = 1000
    tStart = 0
    tEnd = 200
    u_r_rel = 0.1
    u_rel = 0.2

    test_start = 0
    test_end = 300
    test_count = 1000
    eval_step_size = (test_end-test_start)/test_count# 1e-4

system = load_system(system_name, u_r_rel)

if CONTROL:
    x_r = np.array(system.equilibrium)
    x0 = np.copy(x_r)
    #x0[3] = u_rel * system.param.u

    x_target = np.array([1.78e-01, 7.81e-02, 1.28e-01, 2.47e-05])

    train_x = torch.tensor([
        tStart, 
        tStart + 1, 
        tEnd- 1, 
        tEnd,
        tEnd + 1,
        tEnd + 100
    ])
    train_y = torch.tensor([
        x0- x_r, 
        x0- x_r, 
        x_target - x_r, 
        x_target - x_r, 
        x_target - x_r, 
        x_target - x_r
    ])
    
elif LOAD_MODEL:
    train_x, train_y, x0, x_r = load_training_data(MODEL_ID)    
else:
    train_x, train_y, x0, x_r = simulate_system(system, tStart, tEnd, num_data, u_rel)

num_tasks = system.dimension
system_matrix = system.get_ODEmatrix()



# %% train

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix)

if LOAD_MODEL:
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model.optimize()

# %% test

test_x = create_test_inputs(test_count, eval_step_size, test_start, test_end, 1)

model.eval()
likelihood.eval()

#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))

# %% recreate ODE from splines of prediction

fkt = list()
for dimension in range(model.kernelsize):
    output_channel = output.mean[:, dimension]
    fkt.append(spline([(t, y) for t, y in zip(test_x, output_channel)]))

ode = system.get_ODEfrom_spline(fkt)
ode_test_vals = test_x

ode_error_list = [[] for _ in range(model.ode_count)]
for val in ode_test_vals:
    for i in range(model.ode_count):
        #ode_error_list[i].append(np.abs(globals()[f"ode{i+1}"](val)))
        ode_error_list[i].append(np.abs(ode[i](val)))

print('ODE error', np.mean(ode_error_list, axis=1))

train_x_np = train_x.numpy()
train_y_np = train_y.numpy() + x_r
test_x_np = test_x.numpy()
estimation = output.mean.numpy() + x_r


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(train_x_np, train_y_np[:, 0], 'o', label="x1")
ax1.plot(train_x_np, train_y_np[:, 1], 'o', label="x2")
ax1.plot(train_x_np, train_y_np[:, 2], 'o', label="x3")
ax2.plot(train_x_np, train_y_np[:, 3], '.', label="x4")


ax1.plot(test_x_np, estimation[:, 0], label="x1_est")
ax1.plot(test_x_np, estimation[:, 1], label="x2_est")
ax1.plot(test_x_np, estimation[:, 2], label="x3_est")
ax2.plot(test_x_np, estimation[:, 3] , '--', label="x4_est")

ax1.legend()
#ax2.legend()
ax1.grid(True)

plt.show()

# fig, ax1 = plt.subplots()

# color = 'tab:blue'
# ax1.set_xlabel('time')
# ax1.set_ylabel('x1, x2, x3', color=color)
# ax1.plot(test_x_np, estimation[:, 0], label="x1_est", color='tab:blue')
# ax1.plot(test_x_np, estimation[:, 1], label="x2_est", color='tab:orange')
# ax1.plot(test_x_np, estimation[:, 2], label="x3_est", color='tab:green')
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.legend(loc='upper left')
# ax1.grid(True)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
# color = 'tab:red'
# ax2.set_ylabel('x4', color=color)  # we already handled the x-label with ax1
# ax2.plot(test_x_np, estimation[:, 3], label="x4_est", color=color)
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.legend(loc='upper right')

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

system_param = system.equilibrium
if SAVE_MODEL:
    torch.save(model.state_dict(), model_path)
    with open(CONFIG_FILE,"w") as f:
        config['model_id'] = MODEL_ID
        json.dump(config, f)
    add_modelConfig(MODEL_ID, system_name,  x0, system_param, tStart, tEnd, eval_step_size)
    add_training_data(MODEL_ID, train_x_np, train_y_np)

if SAVE_DATA:
    # data = {'time': test_x_np}
    # for i in range(len(estimation[1])):
    #     data[f'f{i+1}'] = estimation[:, i]
    # saveDataToCsv(data_path, data, overwrite=True)

    add_simulationConfig(SIMULATION_ID, MODEL_ID, system_name, x0, system_param, tStart, tEnd, eval_step_size, np.mean(ode_error_list, axis=1))
    add_simulation_data(SIMULATION_ID, test_x_np, estimation)

    # info = collectMetaInformation(SIMULATION_ID, model_name, system_name, model.named_parameters(), np.mean(ode_error_list))
    # saveSettingsToJson(data_path, info, overwrite=True)
    
    with open(CONFIG_FILE,"w") as f:
        config['simulation_id'] = SIMULATION_ID
        json.dump(config, f)

