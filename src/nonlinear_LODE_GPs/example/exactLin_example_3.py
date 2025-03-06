
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
import torch
from result_reporter.latex_exporter import plot_states
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import get_config, load_system, Data_Def, Time_Def
from nonlinear_LODE_GPs.feedback_linearization import Simulation_Config, learn_system_nonlinearities
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False

system_name = "inverted_pendulum"

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)

t  = 20
optim_steps = 100
optim_steps_2 = 100
downsample = 10
sim_time = Time_Def(0, t, step=0.01)
train_time = Time_Def(0, t, step=sim_time.step*downsample)
test_time = Time_Def(0, t, step=0.01)
system = load_system(system_name, a0=0, a1=0, v=1)

sim_configs = [
    Simulation_Config(sim_time, [np.pi/2 - 0.1 , 0 ,0], np.zeros((sim_time.count,1)), downsample),
    Simulation_Config(sim_time, [np.pi/2 , 0 ,0], np.ones((sim_time.count,1)), downsample)
]

alpha, beta = learn_system_nonlinearities(system_name, sim_configs, optim_steps, plot=True)

a0 = 2
a1 = 3
v = 0
system_2 = load_system(system_name, a0=a0, a1=a1, v=v)

system_2._alpha = system_2.alpha
system_2._beta = system_2.beta

system_2.alpha = alpha
system_2.beta = beta

sim_time_u = Time_Def(0, t/2, step=0.01)

x_0 = np.array([ np.pi/2, 0 ,0])

y_ref_control = np.zeros((sim_time_u.count))
ts = sim_time_u.linspace()

control_data = []
for j in range(2):
    if j ==1:
        system_2.alpha = system_2._alpha
        system_2.beta = system_2._beta

    sol = solve_ivp(
        system_2.stateTransition_2, 
        [sim_time_u.start, sim_time_u.end], 
        x_0[0:system_2.state_dimension], 
        method='RK45', 
        t_eval=ts, args=(y_ref_control,sim_time_u.step)
    )
    x = sol.y.transpose()

    solution = []
    for i in range (x.shape[1]):
        solution.append(x[:,i])
    solution.append(y_ref_control)

    control_y = np.stack(solution, -1)

    for i in range(len(ts)):
        control_y[i,-1] = system_2.get_control_from_latent(y_ref_control[i], control_y[i,0:2])

    control_data.append(Data_Def(ts.numpy(), control_y, system_2.state_dimension, system_2.control_dimension, sim_time_u))

# control_data.y = system.rad_to_deg(control_data.y)
# control_data.y = system.rad_to_deg(control_data.y) FIXME

fig_results = plot_states(
    control_data,
    data_names = ['Sim_gp', "sim"], 
    header= ['$\phi$', '$\dot{\phi}$', '$u_1$'], yLabel=['Angle [°]', 'Force [N]'],
    title = f'Inverted Pendulum GP Control: a0: {a0}, a1: {a1}, v: {v}'
    )

plt.show()

# save_plot_to_pdf(fig_results, f'results_plot_{SIM_ID}')

