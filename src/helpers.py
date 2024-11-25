import gpytorch 
from gpytorch.kernels.kernel import Kernel
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
import pprint
import time
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
import json
from systems import *
from result_reporter.sqlite import add_modelConfig, add_simulationConfig, add_simulation_data, add_training_data, get_training_data, get_model_config

def load_system(system_name:str, u=None):

    match system_name:
        case "bipendulum":
            system = Bipendulum()
        case "threetank":
            system = ThreeTank()
        case "system1":
            system = System1()
        case "inverted_pendulum":
            system = Inverted_Pendulum()
        case "nonlinear_threetank":
            system = Nonlinear_ThreeTank(u)
        case _:
            raise ValueError(f"System {system_name} not found")
        
    return system

def load_training_data(model_id:int):
    _training_data = get_training_data(model_id)
    _model_config = get_model_config(model_id)
    if _training_data is not None and _model_config is not None:
        x0 = _model_config['init_state']
        x_r = _model_config['system_param']

        train_x = torch.tensor(_training_data['time'])
        solution = (
            torch.tensor(_training_data[f'f{1}'])- x_r[0], 
            torch.tensor(_training_data[f'f{2}'])- x_r[1], 
            torch.tensor(_training_data[f'f{3}'])- x_r[2],
            torch.tensor(_training_data[f'f{4}'])- x_r[3]
        )
        train_y = torch.stack(solution, -1)
        return train_x, train_y, x0, x_r
        
    else:
        raise ValueError("No training data found")

def simulate_system(system, tStart, tEnd, num_data, u_rel = None):
    train_x = torch.linspace(tStart, tEnd, num_data)
    try:
        solution = system.get_ODEsolution(train_x)
        train_y = torch.stack(solution, -1)
    except NotImplementedError:
        print("No analytical solution available. Use state transition function instead.")

        ts = np.linspace(tStart, tEnd, num_data)
        x0 = np.zeros(system.dimension)
        x_r = np.array(system.equilibrium)
        x0 = np.copy(x_r)
        x0[3] = u_rel * system.param.u
        sol = solve_ivp(system.stateTransition, [tStart, tEnd], x0, method='RK45', t_eval=ts,)#, max_step=dt ,  atol = 1, rtol = 1
        x = sol.y.transpose() - x_r

        solution = (torch.tensor(x[:,0]), torch.tensor(x[:,1]), torch.tensor(x[:,2]), torch.tensor(x[:,3]))
        train_y = torch.stack(solution, -1)
    except:
        print("Error in system")

    return train_x, train_y, x0, x_r

def create_test_inputs(test_count:int, eval_step_size:int, test_start:float, test_end:float, derivatives:int):
    divider = derivatives + 1 
    #test_x = torch.linspace(test_start, test_end, test_count)
    #second_derivative=False
    #divider = 3 if second_derivative else 2
    number_of_samples = int(test_count/divider)
    test_x = torch.linspace(test_start, test_end, number_of_samples)

    data_list = [test_x]
    for i in range(derivatives):
        data_list.append(test_x + torch.tensor(i*eval_step_size))

    # if second_derivative:
    #     test_x = torch.cat([test_x, test_x+torch.tensor(eval_step_size), test_x+torch.tensor(2*eval_step_size)])
    # else:
    #     test_x = torch.cat([test_x, test_x+torch.tensor(eval_step_size)])

    test_x = torch.cat(data_list).sort()[0]
    return test_x

def calc_finite_differences(sample, point_step_size, skip=False, number_of_samples=0):
    """
    param skip: Decides whether to skip every second value of the sample.
                Useful for cases where original samples aren't equidistant
    """
    if sample.ndim == 2:
        NUM_CHANNELS = sample.shape[1]
    else:
        NUM_CHANNELS = 1
    if number_of_samples == 0:
        number_of_samples = sample.shape[0]

    gradients_list = list()
    if skip:
        step = 2
    for index in range(0, step*number_of_samples, step):
        gradients_list.append(list((-sample[index] + sample[index+1])/point_step_size))
    return gradients_list

# def saveDataToCsv(simName:str, data:dict, overwrite:bool=False):
#     fileName = simName +  '.csv'

#     if os.path.exists(fileName) and not overwrite:
#         raise FileExistsError(f"The file {fileName} already exists.")
    
#     df = pd.DataFrame(data)
#     df.to_csv(fileName, index=False)


# def collectMetaInformation(id:int,model_name:str, system_name:str, parameters, rms):
#     info = {}
#     info['date'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     #info['gp_param'] = dict(parameters)
#     info['model'] = model_name
#     info['system'] = system_name
#     info['rms'] = rms
#     info['id'] = id

#     return info

# def saveSettingsToJson(simName:str, settings:dict, overwrite:bool=False):
#     fileName = simName + '.json'
#     if os.path.exists(fileName) and not overwrite:
#         raise FileExistsError(f"The file {fileName} already exists.")

#     with open(fileName,"w") as f:
#         json.dump(settings, f)

