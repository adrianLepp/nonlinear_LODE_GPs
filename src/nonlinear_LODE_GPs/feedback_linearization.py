import gpytorch 
import torch
from result_reporter.latex_exporter import plot_states
from typing import List
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import  optimize_gp, LODEGP
from nonlinear_LODE_GPs.helpers import Time_Def, ODE_System, simulate_system, Data_Def, downsample_data, load_system
from nonlinear_LODE_GPs.gp import GP, Linearizing_Control, Linearizing_Control_2, Linearizing_Control_3


class Simulation_Config():
    def __init__(self, time:Time_Def, x_init, u, downsample:int):
        self.time = time
        self.x_init = x_init
        self.u = u
        self.downsample = downsample
        


def get_state_trajectories(system:ODE_System, sim_configs:List[Simulation_Config]):
    '''
    I: simulate the system to get state trajectories dependent on different inital states and control trajectories
    '''
    data = []
    for sim_config in sim_configs:
        t, x_u= simulate_system(system, sim_config.x_init[0:system.state_dimension], sim_config.time, sim_config.u)
        #t, x_u = downsample_data(_t, _x_u, sim_config.downsample)
        data.append(Data_Def(t, x_u, system.state_dimension, system.control_dimension, y_names=['x1','x2','u']))
    return data

def get_linearizing_feedback(gp:LODEGP, sim_configs:List[Simulation_Config], system_data:List[Data_Def], optim_steps:int):
    '''
    II: use the LODE-GP to learn a linearizing feedback controller trajectory for simulated state trajectories
    '''
    lodegp_data = []
    for i, sim_config in enumerate(sim_configs):
        _train_y = system_data[i].y.clone()
        _train_y[:,-1] = torch.nan
        train_x, train_y = downsample_data(system_data[i].time, _train_y, sim_config.downsample)
        gp.set_train_data(train_x, train_y, strict=False)

        with gpytorch.settings.observation_nan_policy('mask'):
            optimize_gp(gp, optim_steps)
            gp.eval()
            gp.likelihood.eval()

            with torch.no_grad():
                output = gp(sim_config.time.linspace())

        lodegp_data.append(Data_Def(sim_config.time.linspace(), output.mean, system_data[i].state_dim, system_data[i].control_dim, y_names=['x1','x2','y_ref'])) 

    return lodegp_data

def get_feedback_controller(sim_configs:List[Simulation_Config], system_data:List[Data_Def], lodegp_data:List[Data_Def], optim_steps:int):
    '''
    III: use the learned linearizing feedback controller to learn the nonlinearities of the system
    '''
    y_ref = []
    x_u = []

    for i, sim_config in enumerate(sim_configs):
        _train_x_u = lodegp_data[i].y.clone()
        _train_x_u[:,-1] = torch.tensor(sim_config.u).squeeze()
        _train_y_ref = lodegp_data[i].y[:,-1].clone()

        train_y_ref, train_x_u  = downsample_data(_train_y_ref, _train_x_u, sim_config.downsample)
        x_u.append(train_x_u)
        y_ref.append(train_y_ref)


    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    control_gp = Linearizing_Control_2(
        x_u[0][:,0:-1],
        x_u[0][:,-1],
        y_ref[0],
        x_u[1][:,0:-1],
        x_u[1][:,-1],
        y_ref[1],
        # lodegp_data[0].y[:,0:-1],
        # torch.tensor(sim_configs[0].u), 
        # lodegp_data[0].y[:,-1],
        # lodegp_data[1].y[:,0:-1],
        # torch.tensor(sim_configs[1].u), 
        # lodegp_data[1].y[:,-1],  
        likelihood,
        )
    #control_gp.optimize_all(optim_steps, verbose=True)
    control_gp.optimize(optim_steps, verbose=True)
    
    return control_gp

def test_nonlinear_functions(control_gp, sim_configs:List[Simulation_Config], lodegp_data:List[Data_Def]):
    control_gp.likelihood.eval()
    control_gp.eval()
    # control_gp.model.eval()
    control_gp.alpha.eval()
    control_gp.beta.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(): # and gpytorch.settings.debug(False):
        y_ref_output_0 = control_gp.y_ref(lodegp_data[0].y[:,:-1], torch.tensor(sim_configs[0].u))
        y_ref_output_1 = control_gp.y_ref(lodegp_data[1].y[:,:-1], torch.tensor(sim_configs[1].u))

        # y_ref_output_0 = (control_gp(_u_train_x_0)).mean
        # y_ref_output_1 = (control_gp(_u_train_x_1)).mean

        # y_ref_output_0 = (control_gp(_u_train_x_0[:,:-1], _u_train_x_0[:,-1])).mean
        # y_ref_output_1 = (control_gp(_u_train_x_0[:,:-1], _u_train_x_0[:,-1])).mean

        alpha_0 = (control_gp.alpha(lodegp_data[0].y[:,:-1])).mean
        alpha_1 = (control_gp.alpha(lodegp_data[1].y[:,:-1])).mean
        beta_1 = (control_gp.beta(lodegp_data[1].y[:,:-1])).mean

    plt.rcParams['text.usetex'] = True
    fig, (ax1, ax2) = plt.subplots(2, 1)


    ax1.plot(lodegp_data[0].time.numpy(), y_ref_output_0, label = r'$y_{ref,0}(x,u)$')
    ax1.plot(lodegp_data[0].time.numpy(), lodegp_data[0].y[:,-1], label = r'$y_{ref,lode}(x,u)$')


    ax2.plot(lodegp_data[1].time.numpy(), y_ref_output_1, label = r'$y_{ref,1}(x,u)$')
    ax2.plot(lodegp_data[1].time.numpy(), alpha_1, label = r'$\alpha_{pred,1}(x)$')
    ax2.plot(lodegp_data[1].time.numpy(), beta_1, label = r'$\beta_{pred,1}(x)$')
    ax2.plot(lodegp_data[1].time.numpy(), lodegp_data[1].y[:,-1], label = r'$y_{ref,lode}(x,u)$')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Force [N]')
    ax1.set_title('Control estimation inverted pendulum')
    ax1.grid(True)
    ax1.legend()
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Force [N]')
    ax2.set_title('Control estimation inverted pendulum')
    ax2.grid(True)
    ax2.legend()
    return fig

def learn_system_nonlinearities(system_name:str, sim_configs:List[Simulation_Config], optim_steps:int, plot=False):
    # I
    system = load_system(system_name, a0=0, a1=0, v=1)

    system_data = get_state_trajectories(system, sim_configs)

    # II
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=system.dimension,
        noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
    )
    # with gpytorch.settings.observation_nan_policy('mask'):
    lodegp = LODEGP(None, None, likelihood, system.dimension, system.get_ODEmatrix())

    lodegp_data = get_linearizing_feedback(lodegp, sim_configs, system_data, optim_steps)

    # III
    control_gp = get_feedback_controller(sim_configs, system_data, lodegp_data, optim_steps)

    if plot is True:
        plot_states(
            system_data,
            data_names = ['uncontrolled', 'const control'], 
            header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
            title = f'Inverted Pendulum Training Data'
        )
        plot_states(
            lodegp_data,
            data_names = ['uncontrolled', 'const control'], 
            header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
            title = f'Inverted Pendulum LODE GP.'
        )
        figure = test_nonlinear_functions(control_gp, sim_configs,lodegp_data)

        # plt.show()

    def alpha(x):
        return control_gp.alpha(torch.tensor(x).unsqueeze(0)).mean.clone().detach().numpy()
        # with torch.no_grad:
        #     return control_gp.alpha(torch.tensor(x).unsqueeze(0)).mean.numpy()

    def beta(x):
        return control_gp.beta(torch.tensor(x).unsqueeze(0)).mean.clone().detach().numpy()
        # with torch.no_grad:
            # return control_gp.beta(torch.tensor(x).unsqueeze(0)).mean.numpy()

    return alpha, beta