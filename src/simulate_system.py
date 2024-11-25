from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var

import time
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.integrate import solve_ivp
# ----------------------------------------------------------------------------
from systems import Bipendulum, ThreeTank, System1, Inverted_Pendulum, Nonlinear_ThreeTank
from helpers import saveDataToCsv, collectMetaInformation, saveSettingsToJson

u_r_rel = 0.1
u_rel = 0.2

system = Nonlinear_ThreeTank(u_r_rel)

num_data = 30000
tStart = 0
tEnd = 3000 

ts = np.linspace(tStart, tEnd, num_data)
x0 = np.zeros(system.dimension)
x_r = np.array(system.equilibrium)
x0 = np.copy(x_r)
x0[3] = u_rel * system.param.u
sol = solve_ivp(system.stateTransition, [tStart, tEnd], x0, method='RK45', t_eval=ts,)#, max_step=dt ,  atol = 1, rtol = 1

x = sol.y.transpose()

x0 = x0 - x_r
sol_linear = solve_ivp(system.linear_stateTransition, [tStart, tEnd], x0, method='RK45', t_eval=ts,)#, max_step=dt ,  atol = 1, rtol = 1
x_linear = sol_linear.y.transpose()
x_linear = x_linear + x_r

plt.plot(ts, x[:,0], label="f1")
plt.plot(ts, x[:,1], label="f2")
plt.plot(ts, x[:,2], label="f3")
plt.plot(ts, x[:,3], label="u")

plt.plot(ts, x_linear[:,0], label="f1_lin", linestyle='dashed')
plt.plot(ts, x_linear[:,1], label="f2_lin", linestyle='dashed')
plt.plot(ts, x_linear[:,2], label="f3_lin", linestyle='dashed')

plt.xlabel('Time [s]')
plt.ylabel('y')
plt.title("Nonlinear Three Tank and linearization at u_e=0.1 from u_e=0.1 to u_e=0.2 with u=0.2")
plt.legend()
plt.show()

    