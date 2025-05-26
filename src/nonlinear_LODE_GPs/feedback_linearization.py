import gpytorch 
import torch
from result_reporter.latex_exporter import plot_states
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from time import sleep, perf_counter as pc

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import  optimize_gp, LODEGP
from nonlinear_LODE_GPs.helpers import Time_Def, ODE_System, simulate_system, Data_Def, downsample_data, load_system
from nonlinear_LODE_GPs.gp import GP, Linearizing_Control, Linearizing_Control_2, Linearizing_Control_4
from nonlinear_LODE_GPs.likelihoods import FixedTaskNoiseMultitaskLikelihood
from nonlinear_LODE_GPs.subsample import subsample_farthest_point


class Controller():
    def __init__(self, state_dim:int, control_dim:int, a:np.ndarray, v:np.ndarray, alpha=None, beta=None):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.a = a
        self.v = v
        self.alpha = alpha
        self.beta = beta

        self.perf_time  = []

        if self.alpha is not None and self.beta is not None:
            self.type = 'feedback_linearization'
        else:
            self.type = 'direct_control'

    def direct_control(self, x:np.ndarray, y_ref:np.ndarray):
        return  - self.a @ x + self.v * y_ref
    
    def feedback_linearization(self, x:np.ndarray, y_ref:np.ndarray):
        u0 = self.direct_control(x, y_ref)
        beta =  self.beta(x, u0)
        return  self.v / beta * y_ref -  (self.alpha(x) + self.a @ x) / beta

    def __call__(self, x:np.ndarray, y_ref:np.ndarray):
        t0= pc()
        if self.type == 'feedback_linearization':
            u= self.feedback_linearization(x, y_ref).squeeze()
        else:
            u= self.direct_control(x, y_ref).squeeze()
        t1= pc()
        self.perf_time.append(t1-t0)

        return u
    
    def get_performance(self):
        perf_time = np.mean(self.perf_time)
        perf_time_std = np.std(self.perf_time)
        return perf_time, perf_time_std

class Simulation_Config():
    def __init__(self, time:Time_Def, x_init, u, downsample:int, description:str):
        self.time = time
        self.x_init = x_init
        self.u = u
        self.downsample = downsample
        self.description = description
        

def get_state_trajectories(system:ODE_System, sim_configs:List[Simulation_Config], controller:Controller=None):
    '''
    I: simulate the system to get state trajectories dependent on different inital states and control trajectories
    '''
    data = []
    for sim_config in sim_configs:
        if controller is not None:
            y_ref = np.zeros_like(sim_config.time.linspace())
            u = np.zeros_like(sim_config.time.linspace())
        else:
            u = sim_config.u
            y_ref = None
            
        sol = solve_ivp(
            system.stateTransition_2, 
            [sim_config.time.start, sim_config.time.end], 
            sim_config.x_init[0:system.state_dimension], 
            method='RK45', 
            t_eval=sim_config.time.linspace(), 
            args=(sim_config.time.step, controller, u, y_ref), 
            max_step=0.01
        )
        
        x_u = np.concatenate((sol.y.transpose(), u.reshape(-1,1)), axis=1)
        data.append(Data_Def(torch.tensor(sim_config.time.linspace()), torch.tensor(x_u), system.state_dimension, system.control_dimension, sim_config.time, y_names=['x1','x2','u']))

        # t, x_u= simulate_system(system, sim_config.x_init[0:system.state_dimension], sim_config.time, sim_config.u)
        #t, x_u = downsample_data(_t, _x_u, sim_config.downsample)
        # data.append(Data_Def(t, x_u, system.state_dimension, system.control_dimension, y_names=['x1','x2','u']))
    return data

def get_linearizing_feedback(gp:LODEGP, sim_configs:List[Simulation_Config], system_data:List[Data_Def], optim_steps:int, model_config:dict):
    '''
    II: use the LODE-GP to learn a linearizing feedback controller trajectory for simulated state trajectories
    '''
    lodegp_data = []
    for i, sim_config in enumerate(sim_configs):
        _train_y = system_data[i].y.clone()
        _train_y[:,-1] = torch.nan
        _train_y[:,1] = torch.nan
        train_x, train_y = downsample_data(system_data[i].time, _train_y, sim_config.downsample)
        gp.set_train_data(train_x, train_y, strict=False)

        with gpytorch.settings.observation_nan_policy('mask'):
            optimize_gp(gp, optim_steps, verbose=False)

            gp.eval()
            gp.likelihood.eval()

            with torch.no_grad():
                output = gp(sim_config.time.linspace())

        lower, upper = output.confidence_region()

        lodegp_data.append(Data_Def(
            sim_config.time.linspace(), 
            output.mean, 
            system_data[i].state_dim + 1, 
            system_data[i].control_dim - 1, 
            uncertainty={
                'variance': output.variance,
                'lower': lower.numpy(),
                'upper': upper.numpy(),
                }, 
            y_names=['x1','x2','y_ref']
        )) 

    return lodegp_data

def get_feedback_controller(
        sim_configs:List[Simulation_Config], 
        system_data:List[Data_Def], 
        lodegp_data:List[Data_Def], 
        optim_steps:int, 
        ControlGP_Class: Linearizing_Control_2 | Linearizing_Control_4,
        controlGP_kwargs:dict,
        model_config,
        ):
    '''
    III: use the learned linearizing feedback controller to learn the nonlinearities of the system
    '''
    y_ref = []
    x = []
    u = []
    var = []

    for i, sim_config in enumerate(sim_configs):
        _train_x_u = lodegp_data[i].y.clone()
        _train_x_u[:,-1] = system_data[i].y[:,-1].clone()
        _train_y_ref = lodegp_data[i].y[:,-1].clone() 

        # a downsample 
        # train_y_ref, train_x_u  = downsample_data(_train_y_ref, _train_x_u, sim_config.downsample)
        # _, _var  = downsample_data(None, lodegp_data[i].uncertainty['variance'].clone(), sim_config.downsample)
        # x.append(train_x_u[:,:-1])
        # u.append(train_x_u[:,-1])
        # y_ref.append(train_y_ref)
        # var.append(_var[:,-1])


        x.append(_train_x_u[:,:-1])
        u.append(_train_x_u[:,-1])
        y_ref.append(_train_y_ref)
        var.append(lodegp_data[i].uncertainty['variance'].clone()[:,-1])

        # b not

    x_subsample, y_subsample, idx = subsample_farthest_point(torch.cat(x, dim=0), torch.cat(y_ref, dim=0) ,100)
    u_supsample = torch.cat(u, dim=0)[idx]
    var_subsample = torch.cat(var, dim=0)[idx]

    control_gp = ControlGP_Class(x_subsample, u_supsample, y_subsample, variance=var_subsample, **controlGP_kwargs)
    # control_gp = ControlGP_Class(x, u, y_ref, variance=var, **controlGP_kwargs) #, b = 0.1, controller=controller

    if model_config['load']:
        control_gp.load_state_dict(torch.load(model_config['model_path'], map_location=model_config['device']))
    else:
        with gpytorch.settings.observation_nan_policy('mask'):
            control_gp.optimize(optim_steps * 3, verbose=False)

    if model_config['save']:
                torch.save(control_gp.state_dict(), model_config['model_path'])
    
    
    return control_gp

def test_nonlinear_functions(control_gp, system:ODE_System, alpha, beta):
    with gpytorch.settings.observation_nan_policy('mask'): 

        with torch.no_grad(), gpytorch.settings.fast_pred_var(): # and gpytorch.settings.debug(False):
            beta_gp = beta(control_gp.train_inputs[0].numpy(), 0).squeeze()
            alpha_gp = alpha(control_gp.train_inputs[0].numpy()).squeeze()

            # beta_gp = torch.zeros_like(control_gp.train_targets)
            # alpha_gp = torch.zeros_like(control_gp.train_targets)

            beta_system = torch.zeros_like(beta_gp)
            alpha_system = torch.zeros_like(alpha_gp)

            for i in range(control_gp.train_targets.shape[0]):
                # beta_gp[i] = beta(control_gp.train_inputs[0][i].numpy(), 0).item()
                # alpha_gp[i] = alpha(control_gp.train_inputs[0][i].numpy()).item()

                beta_system[i] = system.beta(control_gp.train_inputs[0][i].numpy())
                alpha_system[i] = system.alpha(control_gp.train_inputs[0][i].numpy())

            y_ref = alpha_gp + beta_gp * control_gp.train_u.numpy()
            y_ref_system = alpha_system + beta_system * control_gp.train_u

    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(2, 2)

    ax[0,0].plot(beta_gp, label = r'beta gp')
    ax[0,0].plot(beta_system, label = r'beta sys')

    ax[0,1].plot(alpha_gp, label = r'alpha_gp')
    ax[0,1].plot(alpha_system, label = r'alpha_sys')

    ax[1,0].plot(y_ref, label = r'y_ref gp')
    ax[1,0].plot(y_ref_system, label = r'y_ref sys')

    ax[0,0].set_xlabel('Time [s]')
    ax[0,0].set_ylabel('Force [N]')
    ax[0,0].set_title('Control estimation inverted pendulum')
    ax[0,0].grid(True)
    ax[0,0].legend()
    ax[0,1].set_xlabel('Time [s]')
    ax[0,1].set_ylabel('Force [N]')
    ax[0,1].set_title('Control estimation inverted pendulum')
    ax[0,1].grid(True)
    ax[0,1].legend()
    return fig

def learn_system_nonlinearities(
        system, 
        sim_configs:List[Simulation_Config], 
        optim_steps:int, 
        ControlGP_Class, 
        controlGP_kwargs:dict, 
        plot=False, 
        model_config=None
    ):
    # I
    system_data = get_state_trajectories(system, sim_configs, controlGP_kwargs['controller'])

    system_data_noise = [sys_data.add_noise(controlGP_kwargs['noise']) for sys_data in system_data]

    # II
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=system.dimension,
        noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
    )
    lodegp = LODEGP(None, None, likelihood, system.dimension, system.get_ODEmatrix())
    lodegp_data = get_linearizing_feedback(lodegp, sim_configs, system_data_noise, optim_steps, model_config)

    # III
    control_gp = get_feedback_controller(sim_configs, system_data_noise, lodegp_data, optim_steps, ControlGP_Class, controlGP_kwargs, model_config)

    alpha, beta = control_gp.get_nonlinear_fcts()

    if plot is True:
        '''
        data_names = [cfg.description for cfg in sim_configs]
        plot_states(
            system_data,
            data_names, 
            header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'], #'$\ddot{\phi}$', 
            title = f'Inverted Pendulum Training Data'
        )
        plot_states(
            lodegp_data,
            data_names, 
            header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
            title = f'Inverted Pendulum LODE GP.'
        )
        '''

        figure = test_nonlinear_functions(control_gp, system, alpha, beta)
        plt.show()

        

    return alpha, beta, control_gp