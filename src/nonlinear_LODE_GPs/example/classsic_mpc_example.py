import numpy as np
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

from datetime import datetime

from nonlinear_LODE_GPs.systems.nonlinear_watertank import Parameter
from nonlinear_LODE_GPs.helpers import *
from result_reporter.latex_exporter import create_mpc_plot, save_plot_to_pdf

# Import do_mpc package:
import do_mpc


def mse_mean(mean, ref, indx=None):
        if indx is None:
            return torch.mean(torch.square(mean - ref))
        else:
            return torch.mean(torch.square((mean - ref)[indx]))

def constr_viol(mean, ub, lb, indx=None):
    if indx is None:
        return torch.mean(torch.relu(mean - ub) + torch.relu(lb - mean))
    else:
        return torch.mean(torch.relu((mean - ub)[indx]) + torch.relu((lb - mean)[indx]))



x_ref=[1.8807e-01, 1.1468e-01]
u_ref = 3e-5

x0 = np.array([0.08358817533129459, 0.05096839959225279]).reshape(-1,1) #2e-5

dt = 1
sim_length = 400

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
# dx = model.set_variable(var_type='_x', var_name='dx', shape=(2,1))

# Variables can also be vectors:
# dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))

# pump:
u = model.set_variable(var_type='_u', var_name='u')

x_meas = model.set_meas('x_meas', x, meas_noise=True)

# Input measurements
u_meas = model.set_meas('u_meas', u, meas_noise=False)



param = Parameter()

# A = model.set_variable('parameter', 'A')
# c12 = model.set_variable('parameter', 'c12')
# c2R = model.set_variable('parameter', 'c2R')
# g = model.set_variable('parameter', 'g')


x_next = vertcat(
    1/param.A*(param._u * u-param.c12*np.sqrt(2*param.g*(x[0]-x[1]))),
    1/param.A*(param.c12*np.sqrt(2*param.g*(x[0]-x[1]))-param.c2R*np.sqrt(2*param.g*(x[1])))
)

model.set_rhs('x', x_next, process_noise = False)

model.setup()

# CONTROLLER

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 10,
    't_step': dt,
    'n_robust': 0,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

mterm = (x_ref[0]-x[0])**2 + (x_ref[1]-x[1])**2
lterm = (x_ref[0]-x[0])**2 + (x_ref[1]-x[1])**2

# mterm = (x[0])**2 + (x[1])**2
# lterm = (x[0])**2 + (x[1])**2

mpc.set_objective(mterm=mterm, lterm=lterm)

# mpc.set_rterm(
#     phi_m_1_set=1e-2,
#     phi_m_2_set=1e-2
# )


mpc.bounds['lower','_u', 'u'] = 0
mpc.bounds['upper','_u', 'u'] = param.u * 2

mpc.bounds['lower','_x', 'x'] = [0,0]
mpc.bounds['upper','_x', 'x'] = [0.6,0.6]

_x = model.x

# mpc.set_rterm(u=10000)

# mpc.set_nl_cons('x', _x['x'], ub=np.array(x_ref), soft_constraint=True, penalty_term_cons=1e3)

# mpc.terminal_bounds['lower', 'x'] = x_ref
# mpc.terminal_bounds['upper', 'x'] = x_ref

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)

simulator.setup()

simulator.x0 = x0
mpc.x0 = x0



mpc.set_initial_guess()


import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
# mpl.rcParams['font.size'] = 18
# mpl.rcParams['lines.linewidth'] = 3
# mpl.rcParams['axes.grid'] = True

textWidth= 469.4704
textWidth_cm = 8.4
mpl.rcParams['text.usetex'] = True
mpl.style.use('seaborn-v0_8-paper')
mpl.style.use('tex')

xLabel='Time ($\mathrm{s})$'
yLabel='Water Level ($\mathrm{m}$)'


mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)


# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()


for g in [sim_graphics, mpc_graphics]:
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='x', axis=ax[0])
    # g.add_line(var_type='_x', var_name='phi_2', axis=ax[0])
    # g.add_line(var_type='_x', var_name='phi_3', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='u', axis=ax[1])


ax[0].set_ylabel('water level [m]')
ax[1].set_ylabel('volume flow')
ax[1].set_xlabel('time [s]')

u0 = np.zeros((1,1))

rise_time = None
settling_time = None

u_data = [u0]
x_data = [x0]
calc_time = []

for i in range(int(sim_length / dt)):
    if np.isclose(x0[0], x_ref[0], atol=5e-3) and rise_time is None:
        print(f"rise time: {i * dt}")
        rise_time = i * dt
        
    
    if np.isclose(x0[:,0], x_ref, atol=5e-3).all() and settling_time is None:
        # u0= u_ref
        settling_time = i * dt
        print(f"Settling time: {settling_time}")
    else:
        start = datetime.now()
        u0 = mpc.make_step(x0)
        calc_time.append((datetime.now() - start).microseconds)

    xu0 = simulator.make_step(u0)
    x0 =xu0[0:2]
    u_data.append(u0)
    x_data.append(x0)


print(np.mean(calc_time))
print(np.std(calc_time))

u_np = np.array(u_data)
x_np = np.array(x_data)
full_data = np.stack((x_np[:,0,:], x_np[:,1,:], u_np[:,0,:])).squeeze()
time = np.linspace(0,sim_length,int(sim_length / dt) + 1)

test_data = Data_Def(time, full_data, 2, 1)

peak_value = np.max(test_data.y)
peak_time = test_data.time[np.argmax(test_data.y[0,:])]   


control_err = mse_mean(
    torch.tensor(test_data.y[0:2,:]),
    torch.tile(torch.tensor(x_ref), (test_data.y.shape[1], 1)).t()
    #torch.zeros_like(torch.tensor(lode_data))
)

control_mean =  np.mean(test_data.y[2,:])

print(f"mean Control: {control_mean}")
print(f"Control error: {control_err}")



reference_data = {
    'time': test_data.time,
    'f1': test_data.y[0,:],
    'f2': test_data.y[1,:],
    'f3': test_data.y[2,:],
}

fig = create_mpc_plot(None, None, ['x1','x2', 'u'], 'Time ($\mathrm{s})$', 'Water Level ($\mathrm{m}$)', reference_data, x_e=[x_ref[0],x_ref[1],u_ref])



plt.show()

# save_plot_to_pdf(fig, f'mpc_plot_classic_3')


sim_graphics.plot_results()
# Reset the limits on all axes in graphic to show the data.
sim_graphics.reset_axes()
# Show the figure:
fig

plt.show()

sim_graphics.clear()

mpc_graphics.plot_predictions()
mpc_graphics.reset_axes()

fig
# Show the figure:
plt.show()