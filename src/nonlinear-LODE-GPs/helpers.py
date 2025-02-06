from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
import torch
import numpy as np
from scipy.integrate import solve_ivp
from systems import *
import matplotlib.pyplot as plt
import json
from result_reporter.sqlite import add_modelConfig, add_simulationConfig, add_simulation_data, add_training_data, get_training_data, get_model_config, add_reference_data

default_config ='config.json'
class Time_Def():
    start:float
    end:float
    count:int
    step:float

    def __init__(self, start, end, count=None, step=None):
        self.start = start
        self.end = end
        if count is None and step is not None:
            self.count = int(ceil((end-start)/step)) +1
            self.step = step
        elif count is not None and step is None:
            self.count = count
            self.step = (end-start)/count
        else:
            raise ValueError("Either count or step must be given")
        
    def linspace(self):
        return torch.linspace(self.start, self.end, self.count)

class State_Description():
    equilibrium:torch.Tensor
    init:torch.Tensor
    target:torch.Tensor
    min:torch.Tensor
    max:torch.Tensor

    def __init__(self, equilibrium:torch.Tensor=None, init:torch.Tensor=None, target:torch.Tensor=None,  min:torch.Tensor=None, max:torch.Tensor=None):
        self.init = init

        if equilibrium is None:
            self.equilibrium = target
        else:
            self.equilibrium = equilibrium
        if target is None:
            self.target = equilibrium
        else:
            self.target = target
        self.min = min
        self.max = max

class Data_Def():
    def __init__(self, x,y,state_dim:int, control_dim:int, time:Time_Def=None, uncertainty:dict=None):
        self.time = x
        self.y = y
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.time_def = time
        self.uncertainty = uncertainty

    def to_report_data(self):
        data = {'time': self.time}
        for i in range(self.state_dim + self.control_dim):
            data[f'f{i+1}'] = self.y[:,i]
        return data

def load_system(system_name:str):

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
            system = Nonlinear_ThreeTank()
        case "nonlinear_watertank":
            system = Nonlinear_Watertank()
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
        #FIXME: model specifig
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

def simulate_system(system, x0, time:Time_Def, u = None, linear=False):
    ts = time.linspace()
    
    try:
        solution = system.get_ODEsolution(ts)
        train_y = torch.stack(solution, -1)
    except NotImplementedError:
        #print("No analytical solution available. Use state transition function instead.")
        
        if linear:
            sol = solve_ivp(system.linear_stateTransition, [time.start, time.end], x0, method='RK45', t_eval=ts, args=(u,time.step))#, max_step=dt ,  atol = 1, rtol = 1
        else:
            sol = solve_ivp(system.stateTransition, [time.start, time.end], x0, method='RK45', t_eval=ts, args=(u,time.step))#, max_step=dt ,  atol = 1, rtol = 1
        

        x = sol.y.transpose()

        solution = []
        #solution = (torch.tensor(x[:,0]), torch.tensor(x[:,1]), torch.tensor(x[:,2]), torch.tensor(u.squeeze())) #FIXME: model specific
        for i in range (x.shape[1]):
            solution.append(torch.tensor(x[:,i]))
        solution.append(torch.tensor(u.squeeze()))
        
        train_y = torch.stack(solution, -1)
    except:
        print("Error in system")

    return ts, train_y

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

def get_ode_from_spline(model, system, output, test_x):
    fkt = list()
    for dimension in range(model.covar_module.kernelsize):
        output_channel = output.mean[:, dimension]
        fkt.append(spline([(t, y) for t, y in zip(test_x, output_channel)]))

    ode = system.get_ODEfrom_spline(fkt)
    ode_test_vals = test_x

    ode_error_list = [[] for _ in range(model.covar_module.ode_count)]
    for val in ode_test_vals:
        for i in range(model.covar_module.ode_count):
            #ode_error_list[i].append(np.abs(globals()[f"ode{i+1}"](val)))
            ode_error_list[i].append(np.abs(ode[i](val)))

    print('ODE error', np.mean(ode_error_list, axis=1))
    return ode, ode_error_list

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

def equilibrium_base_change(time, states, equilibriums, changepoints, add=True):
    
    for i in range(len(equilibriums)):
        if not add:
            equilibriums[i] = - torch.tensor(equilibriums[i])  
        else: 
            equilibriums[i] =  torch.tensor(equilibriums[i])  
    

    for i in range(len(time)):
        if time[i] <= changepoints[0]:
            states[i] = states[i] + equilibriums[0]
        else:
            states[i] = states[i] + equilibriums[1]

    return states

def plot_results(train:Data_Def, test:Data_Def,  ref:Data_Def = None, equilibrium=None):
    labels = ['train', 'gp', 'linear']
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    alpha_eq = 0.5

    if equilibrium is not None:
        eq_range = np.array([equilibrium, equilibrium])
        t_range = [test.time[0], test.time[-1]]
        

    for i in range(test.state_dim):
        color = f'C{i}'

        if test.uncertainty is not None:
            ax1.fill_between(test.time, test.uncertainty['lower'][:,i], test.uncertainty['upper'][:,i], alpha=0.2, color=color)


        ax1.plot(train.time, train.y[:, i], '.', color=color, label=f'x{i+1}_{labels[0]}')    
        ax1.plot(test.time, test.y[:, i], color=color, label=f'x{i+1}_{labels[1]}', alpha=0.5)
        if ref is not None:
            ax1.plot(ref.time, ref.y[:, i], '--', color=color, label=f'x{i+1}_{labels[2]}')
        
        if equilibrium is not None:
            ax1.plot(t_range, eq_range[:,i], '--', label=f'x{i+1}_eq',color=color, alpha=alpha_eq)

        

    for i in range(test.control_dim):
        idx = test.state_dim + i
        color = f'C{idx}'

        if test.uncertainty is not None:
            ax2.fill_between(test.time, test.uncertainty['lower'][:,idx], test.uncertainty['upper'][:,idx], alpha=0.2, color=color)

        ax2.plot(train.time, train.y[:, idx], '.', color=color, label=f'u{i+1}_{labels[1]}')
        ax2.plot(test.time, test.y[:, idx], color=color, label=f'u{i+1}_{labels[1]}', alpha=0.5)
        if ref is not None:
            ax2.plot(ref.time, ref.y[:, idx], '--', color=color, label=f'u{i+1}_{labels[2]}')

        if equilibrium is not None:
            ax2.plot(t_range, eq_range[:,idx], '--', label=f'u{i+1}_eq',color=color, alpha=alpha_eq)

    ax2.tick_params(axis='y', labelcolor=color)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    #ax1.legend()
    #ax2.legend()
    ax1.grid(True)

    plt.show()


def save_results(
        model, 
        system_param, 
        x0, 
        sim_id:int, 
        model_id:int, 
        config_file:str, 
        system_name:str, 
        config:dict, 
        model_path:str, 
        train_data:Data_Def, 
        test_data:Data_Def, 
        train_time:Time_Def, 
        test_time:Time_Def, 
        error=None, 
        ref_data:Data_Def=None, 
        linear=False
    ):

    if error is None:
        error = []
    
    torch.save(model.state_dict(), model_path)
    with open(config_file,"w") as f:
        config['model_id'] = model_id
        json.dump(config, f)
    add_modelConfig(model_id, system_name,  x0, system_param, train_time.start, train_time.end, train_time.step)
    add_training_data(model_id, train_data.time, train_data.y)

    add_simulationConfig(sim_id, model_id, system_name, x0, system_param, test_time.start, test_time.end, test_time.step, error)
    add_simulation_data(sim_id, test_data.time, test_data.y)

    if ref_data is not None:
        if linear:
            type = 'linear'
        else:
            type = 'nonlinear'
        add_reference_data(sim_id, type, ref_data.time, ref_data.y)
    
    with open(config_file,"w") as f:
        config['simulation_id'] = sim_id
        json.dump(config, f)


def stack_tensor(tensor, num_tasks, dim=-1, batch_dim=0):
    indices = torch.tensor([i for i in range(0, tensor.shape[-1], num_tasks)])
    zer = int(0)
    ind0 = indices
    ind1 = indices + int(1)
    ind2 = indices + int(2)
    ind3 = indices + int(3)
    ind4 = indices + int(4)
    # rest_dims = [i for i in range(tensor.ndim + 1) if i != batch_dim]
    if num_tasks == 5:
        return torch.stack((torch.index_select(tensor, dim, ind0),
                            torch.index_select(tensor, dim, ind1),
                            torch.index_select(tensor, dim, ind2),
                            torch.index_select(tensor, dim, ind3),
                            torch.index_select(tensor, dim, ind4)), dim=-1)
    elif num_tasks == 1:
        return ind0
    elif num_tasks == 2:
        return torch.stack((torch.index_select(tensor, dim, ind0),
                            torch.index_select(tensor, dim, ind1)), dim=-1)
    elif num_tasks == 3:
        return torch.stack((torch.index_select(tensor, dim, ind0),
                            torch.index_select(tensor, dim, ind1),
                            torch.index_select(tensor, dim, ind2)), dim=-1)
    

def stack_plot_tensors(mean,  num_tasks):#lower, upper,
    mean = stack_tensor(mean, num_tasks)
    # lower = stack_tensor(lower, num_tasks)
    # upper = stack_tensor(upper, num_tasks)
    return mean #lower, upper

def plot_weights(x, weights, title="Weighting Function"):
    plt.figure(figsize=(12, 6))
    if isinstance(weights, list):
        for i, weight in enumerate(weights):
            plt.plot(x, weight, label=f'Model {i+1}')
        
        #plt.plot(x, sum(weights), label='Sum')
        plt.legend()
    else:
        plt.plot(x, weights)
    plt.xlabel("t")
    plt.ylabel("Weight")
    #plt.title(title)
    plt.show()


def get_config(system_name:str, config_file:str=default_config, save:bool=False):
    print("\n----------------------------------------------------------------------------------\n")
    try:
        with open(config_file,"r") as f:
            config = json.load(f)
            model_dir=config['model_dir']
            data_dir=config['data_dir']
            model_name = config['model_name']

            if save:
                
                SIM_ID = config['simulation_id'] + 1
                MODEL_ID = config['model_id'] + 1

                name =  '_' + model_name + "_" + 'mpc' + "_" + system_name
                model_path = f'{model_dir}/{str(MODEL_ID)}{name}.pth'
            else: 
                SIM_ID = -1
                MODEL_ID = -1
                model_path = f'{model_dir}/{model_name}.pth'

        
        print(f"simulate {system_name}")

        if save:
            print(f"save model with model id {MODEL_ID}")
            print(f"save data with data id {SIM_ID}")
        
    except:
        print("No config file found. Data and model will not be saved.")
    print("\n----------------------------------------------------------------------------------\n")

    return SIM_ID, MODEL_ID, model_path, config

def save_config(config:dict, config_file:str=default_config):
    with open(config_file,"w") as f:
        json.dump(config, f)


def save_everything(system_name:str, model_name:str, config:dict, train_data:Data_Def, gp_data:Data_Def, sim_data:Data_Def, states:State_Description, model_dict:dict, metrics:list[float]=[]):
    torch.save(model_dict, model_name)

    add_modelConfig(config['model_id'], system_name, states.init.numpy(), states.equilibrium.numpy(), train_data.time_def.start, train_data.time_def.end, train_data.time_def.step)
    add_training_data(config['model_id'], train_data.time, train_data.y)

    add_simulationConfig(config['simulation_id'], config['model_id'], system_name, states.init.numpy(), states.equilibrium.numpy(), gp_data.time_def.start, gp_data.time_def.end, gp_data.time_def.step, metrics)
    add_simulation_data(config['simulation_id'], gp_data.time, gp_data.y)

    if sim_data is not None:
        add_reference_data(config['simulation_id'], 'nonlinear', sim_data.time, sim_data.y)

    save_config(config)
    print(f"save model with model id {config['model_id']}")
    print(f"save data with data id {config['simulation_id']}")
    pass

def downsample_data(t:torch.Tensor, y:torch.Tensor, factor=10):
    t_redux = t[::factor]
    y_redux = y[::factor,:]
    
    return t_redux, y_redux 