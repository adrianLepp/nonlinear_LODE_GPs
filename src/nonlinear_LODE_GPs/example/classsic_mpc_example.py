import numpy as np
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip.
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)

from nonlinear_LODE_GPs.systems.nonlinear_watertank import Parameter

# Import do_mpc package:
import do_mpc

x_ref=[0.188,0.115]

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
    'n_horizon': 100,
    't_step': 0.1,
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

mpc.set_rterm(u=10000)

# mpc.set_nl_cons('x', _x['x'], ub=np.array(x_ref), soft_constraint=True, penalty_term_cons=1e3)

# mpc.terminal_bounds['lower', 'x'] = x_ref
# mpc.terminal_bounds['upper', 'x'] = x_ref

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.1)

simulator.setup()

x0 = np.array([0.084, 0.051]).reshape(-1,1)

simulator.x0 = x0
mpc.x0 = x0



mpc.set_initial_guess()


import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

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
for i in range(250):
    u0 = mpc.make_step(x0)
    xu0 = simulator.make_step(u0)
    x0 =xu0[0:2]

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