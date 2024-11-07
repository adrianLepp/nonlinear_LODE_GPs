import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import time
import torch
import matplotlib.pyplot as plt
import argparse
import numpy as np
from scipy.integrate import solve_ivp
# ----------------------------------------------------------------------------
from  lodegp import LODEGP 
from systems import Bipendulum, ThreeTank, System1
from helpers import saveDataToCsv, collectMetaInformation, saveSettingsToJson

DATA_ID = 0 # TODO load DATA_ID from file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="example simulationfor lodegp"
    )
    parser.add_argument("--system", required=True, type=str, help="system to simulate")
    parser.add_argument("--save", required=True, type=bool, help="save data", default=False)
    parser.add_argument("--model", required=True, type=str, default="None", help="load or save model")
    #parser.add_argument("--num3", required=True, type=int)

    args = parser.parse_args()
    system_name = args.system
    SAVE_DATA = args.save
    if args.model == 'load':
        LOAD_MODEL = True
        SAVE_MODEL = False
    elif args.model == 'save':
        LOAD_MODEL = False
        SAVE_MODEL = True
    else:
        LOAD_MODEL = False
        SAVE_MODEL = False

    print(f"simulate {system_name}")
    # if LOAD_MODEL:
    #     print("load model with model id {MODEL_ID}")
    # if SAVE_MODEL:
    #     print(f"save model with model id {MODEL_ID}")
    if SAVE_DATA is True:
        print(f"save data with data id {DATA_ID}")

# %% config

#torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_default_dtype(torch.float64)
device = 'cpu'

model_name = "lodegp"
#system_name = "bipendulum"
name = str(DATA_ID) + '_' + model_name + "_" + system_name

model_dir = "data/"
model_path = model_dir + name + ".pth"

data_dir = "../data/"

# %% setup

num_data = 50
tStart = 0
tEnd = 1 
train_x = torch.linspace(tStart, tEnd, num_data)


match system_name:
    case "bipendulum":
        system = Bipendulum()
    case "threetank":
        system = ThreeTank()
    case "system1":
        system = System1()
    case "inverted_pendulum":
        system = Inverted_Pendulum()
    case _:
        raise ValueError(f"System {system_name} not found")

try:
    solution = system.get_ODEsolution(train_x)
except NotImplementedError:
    print("No analytical solution available. Use state transition function instead.")

    ts = np.linspace(tStart, tEnd, num_data)

    x0 = np.zeros(5)
    angle= 1
    x0[2] = angle/180*np.pi

    sol = solve_ivp(system.stateTransition, [tStart, tEnd], x0, method='RK45', t_eval=ts,)#, max_step=dt ,  atol = 1, rtol = 1
    x = sol.y.transpose()

    solution = (torch.tensor(x[:,0]), torch.tensor(x[:,2]), torch.tensor(x[:,4]))
except:
    print("Error in system")

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

test_start = 0
test_end = 1 
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
estimation = output.mean.numpy()


plt.figure()
plt.plot(train_x_np, train_y_np[:, 0], label="f1")
plt.plot(train_x_np, train_y_np[:, 1], label="f2")
plt.plot(train_x_np, train_y_np[:, 2], label="x''")
plt.legend()

plt.plot(test_x_np, estimation[:, 0], label="f1_est")
plt.plot(test_x_np, estimation[:, 1], label="f2_est")
plt.plot(test_x_np, estimation[:, 2], label="x'' _est")
plt.legend()
plt.show()


if SAVE_MODEL:
    torch.save(model.state_dict(), model_path)

if SAVE_DATA:
    data = {'time': test_x_np}
    for i in range(len(estimation[1])):
        data[f'f{i+1}'] = estimation[:, i]
    saveDataToCsv(data_dir, name, data, overwrite=True)

    info = collectMetaInformation(DATA_ID, model_name, system_name, model.named_parameters(), np.mean(ode_error_list))
    saveSettingsToJson(data_dir, name, info, overwrite=True)
    DATA_ID += 1

