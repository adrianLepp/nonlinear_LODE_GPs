
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
import torch
from result_reporter.latex_exporter import plot_states
import numpy as np
import matplotlib.pyplot as plt
import gpytorch

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import get_config, load_system, Data_Def, Time_Def
from nonlinear_LODE_GPs.feedback_linearization import Simulation_Config, learn_system_nonlinearities, Controller
from nonlinear_LODE_GPs.gp import Linearizing_Control_2, Linearizing_Control_4, Linearizing_Control_5
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False

system_name = "inverted_pendulum"

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)

model_dir=config['model_dir']
data_dir=config['data_dir']
model_name = config['model_name']
name =  '_' + model_name + "_" + system_name
model_path = f'{model_dir}/1{name}.pth'

model_config = {
    'device': device,
    'model_path': model_path,
    'load': False,
    'save': False,
}


t  = 10
optim_steps = 100
downsample = 20
sim_time = Time_Def(0, t, step=0.01)
train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t, step=0.01)

a0 = 2
a1 = 3
v = 0

system = load_system(system_name, a0=0, a1=0, v=1)
controller_0 = Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]))

sim_configs = [
    # Simulation_Config(sim_time, [np.pi/2 - 0.1 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [np.pi/2 + 0.1 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [np.pi/2 , 0 ,0], np.ones((sim_time.count,1)), downsample, 'u=1'),
    # Simulation_Config(sim_time, [np.pi/2 , 0 ,0], -np.ones((sim_time.count,1)), downsample, 'u=-1'),

    # Simulation_Config(sim_time, [np.pi/2 , 0 ,0], np.sin(sim_time.linspace()), downsample, 'sin'),
    # Simulation_Config(sim_time, [np.pi/2 , 0 ,0], -np.sin(sim_time.linspace()), downsample, '-sin'),
    # Simulation_Config(sim_time, [np.pi/2 , 0 ,0], np.sin(sim_time.linspace()**2/4), downsample, 'sin^2'),


    Simulation_Config(sim_time, [np.pi/2  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    Simulation_Config(sim_time, [np.pi/4  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    Simulation_Config(sim_time, [-np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    Simulation_Config(sim_time, [-np.pi/2 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),

    # Simulation_Config(sim_time, [-np.pi/8 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [np.pi/8 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [-np.pi*5/8 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [np.pi*5/8 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),



    # Simulation_Config(sim_time, [np.pi , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [2*np.pi , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [0 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),

    # Simulation_Config(sim_time, [np.pi*6/2  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [np.pi*12/4  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [-np.pi*12/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [-np.pi*6/2 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),

    # Simulation_Config(sim_time, [3*np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    # Simulation_Config(sim_time, [-3*np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
]


control_gp_kwargs = {
    #Linearizing_Control_2
    'consecutive_training':False,
    #Linearizing_Control_4
    'b' : 110,
    'a' : torch.tensor([[a0],[a1]], dtype=torch.float64),
    'v' : torch.tensor([v], dtype=torch.float64),
    'controller':controller_0 # controller_0 # None  
}

alpha, beta = learn_system_nonlinearities(
    system, 
    sim_configs, 
    optim_steps, 
    ControlGP_Class = Linearizing_Control_5,
    controlGP_kwargs = control_gp_kwargs,
    plot=True, 
    model_config=model_config,
    )

v = 0

# -----------------------------------------------------------------
# TEST CONTROLLER
# -----------------------------------------------------------------

test_controller = [
    Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]), alpha=alpha, beta=beta),
    Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]), alpha=system.alpha, beta=system.beta),
    controller_0
]


sim_time_u = Time_Def(0, t*1, step=0.01)

x_0 = np.array([ np.pi/2, 0 ,0])

y_ref_control = np.zeros((sim_time_u.count))
ts = sim_time_u.linspace()

with gpytorch.settings.observation_nan_policy('mask'):
    with torch.no_grad():

        control_data = []
        for j in range(len(test_controller)):
            u_control = np.zeros_like(ts)
            sol = solve_ivp(
                system.stateTransition_2, 
                [sim_time_u.start, sim_time_u.end], 
                x_0[0:system.state_dimension], 
                method='RK45', 
                t_eval=ts, args=(sim_time.step, test_controller[j], u_control, y_ref_control),
                max_step=0.01
            )
            x = sol.y.transpose()

            solution = []
            for i in range (x.shape[1]):
                solution.append(x[:,i])

            solution.append(u_control)
            control_y = np.stack(solution, -1)

            control_data.append(Data_Def(ts.numpy(), control_y, system.state_dimension, system.control_dimension, sim_time_u))

fig_results = plot_states(
    control_data,
    data_names = ['Sim_gp', "sim", 'simple'], 
    header= [r'$\phi$', r'$\dot{\phi}$', r'$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
    title = f'Inverted Pendulum GP Control: a0: {a0}, a1: {a1}, v: {v}'
    )

plt.figure()
plt.plot(control_data[0].y[:,0],control_data[0].y[:,1], label='GP')
plt.plot(control_data[1].y[:,0],control_data[1].y[:,1], label='Feedback')
plt.plot(control_data[2].y[:,0],control_data[2].y[:,1], label='Simple')
plt.xlabel('Angle [rad]')
plt.ylabel('Angular Velocity [rad/s]')
plt.legend()
plt.grid(True)

plt.show()

# save_plot_to_pdf(fig_results, f'results_plot_{SIM_ID}')

