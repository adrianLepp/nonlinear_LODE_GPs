import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import time
import torch
import matplotlib.pyplot as plt
from  lodegp import LODEGP 
from systems import Bipendulum, ThreeTank, System1

# %% config

torch.set_default_tensor_type(torch.DoubleTensor)
device = 'cpu'

SAVE_MODEL = False
LOAD_MODEL = True

if LOAD_MODEL:
    SAVE_MODEL = False

model_dir = "data/"
model_path = model_dir + "lodegp_bipendulum.pth"




# %% setup

num_data = 50 
train_x = torch.linspace(0, 15, num_data)

system = Bipendulum()
solution = system.get_ODEsolution(train_x)
num_tasks = system.dimension
system_matrix = system.get_ODEmatrix()



# %% train

train_y = torch.stack(solution, -1)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix)
#model(train_x)

if LOAD_MODEL:
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model.optimize()

# %% test

start = time.time()        
end = time.time()

test_start = 1
test_end = 5 
test_count = 1000
eval_step_size = 1e-4
second_derivative=True 
divider = 3 if second_derivative else 2
number_of_samples = int(test_count/divider)
test_x = torch.linspace(test_start, test_end, number_of_samples)
if second_derivative:
    test_x = torch.cat([test_x, test_x+torch.tensor(eval_step_size), test_x+torch.tensor(2*eval_step_size)])
else:
    test_x = torch.cat([test_x, test_x+torch.tensor(eval_step_size)])
test_x = test_x.sort()[0]
model.eval()
likelihood.eval()

#output = model(test_x)
with torch.no_grad():
    output = likelihood(model(test_x))

# %% recreate ODE from splines of prediction

fkt = list()
for dimension in range(model.kernelsize):
    output_channel = output.mean[:, dimension]
    fkt.append(spline([(t, y) for t, y in zip(test_x, output_channel)]))

ode = system.get_ODEfrom_spline(fkt)
ode_test_vals = test_x

ode_error_list = [[] for _ in range(model.ode_count)]
for val in ode_test_vals:
    for i in range(model.ode_count):
        #ode_error_list[i].append(np.abs(globals()[f"ode{i+1}"](val)))
        ode_error_list[i].append(np.abs(ode[i](val)))

print('ODE error', np.mean(ode_error_list))


train_x_np = train_x.numpy()
train_y_np = train_y.numpy()
test_x_np = test_x.numpy()


plt.figure()
plt.plot(train_x_np, train_y_np[:, 0], label="f1")
plt.plot(train_x_np, train_y_np[:, 1], label="f2")
plt.plot(train_x_np, train_y_np[:, 2], label="x''")
plt.legend()

plt.plot(test_x_np, output.mean[:, 0].numpy(), label="f1_est")
plt.plot(test_x_np, output.mean[:, 1].numpy(), label="f2_est")
plt.plot(test_x_np, output.mean[:, 2].numpy(), label="x'' _est")
plt.legend()
plt.show()


if SAVE_MODEL:
    torch.save(model.state_dict(), model_path)