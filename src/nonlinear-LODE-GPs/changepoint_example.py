import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from  lodegp import LODEGP, Changepoint_LODEGP, optimize_gp
from helpers import *

#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
device = 'cpu'

system_name = "nonlinear_threetank"

num_data = 200
tStart = 0
tEnd = 20
u_r_rel = [ 0.1, 0.2]

u_rel = 0.3
test_start = 0
test_end = 20
test_count = 200
eval_step_size = (test_end-test_start)/test_count# 1e-4

system = load_system(system_name)

num_tasks = system.dimension

system_matrices = []
equilibriums = []

for i in range(len(u_r_rel)):
    A, equilibrium = system.get_ODEmatrix(u_r_rel[i])
    system_matrices.append(A)
    equilibriums.append(equilibrium)

changepoints = [100]

x0 = np.array(equilibriums[0])

train_x, train_y= simulate_system(system, x0, tStart, tEnd, num_data, u_rel)

train_y = equilibrium_base_change(train_x, train_y, equilibriums, changepoints, add=False)


# %% train

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = Changepoint_LODEGP(train_x, train_y, likelihood, num_tasks, system_matrices, changepoints)

optimize_gp(model,10)

# %% test

test_x = create_test_inputs(test_count, eval_step_size, test_start, test_end, 1)

model.eval()
likelihood.eval()

#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))

# %% recreate ODE from splines of prediction


train_x_np = train_x.numpy()

train_y_np = equilibrium_base_change(train_x, train_y, equilibriums, changepoints, add=True).numpy()
estimation = equilibrium_base_change(test_x, output.mean, equilibriums, changepoints, add=True).numpy()

#train_y_np = train_y.numpy() + x_r
test_x_np = test_x.numpy()
#estimation = output.mean.numpy() + x_r

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