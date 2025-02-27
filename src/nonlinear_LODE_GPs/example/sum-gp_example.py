


import gpytorch 
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from nonlinear_LODE_GPs.kernels import *
from nonlinear_LODE_GPs.mean_modules import *
import torch

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import *
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.weighting import Gaussian_Weight, KL_Divergence_Weight
from nonlinear_LODE_GPs.sum_gp import Local_GP_Sum

torch.set_default_dtype(torch.float64)
device = 'cpu'


local_predictions = False
SAVE = False
system_name = "nonlinear_watertank"

SIM_ID, MODEL_ID, model_path, config = get_config(system_name, save=SAVE)

optim_steps_single = 100
optim_steps = 10

equilibrium_controls = [
    0.1, 
    0.2, 
    0.3, 
    # 0.4, 
    # 0.5,
]

u_ctrl = 1

x0 = torch.tensor([0.0, 0.0])

t0 = 0.0
t1 = 50.0

downsample = 10
sim_time = Time_Def(t0, t1, step=0.1)
train_time = Time_Def(t0, t1, step=sim_time.step*downsample)
test_time = Time_Def(t0, t1, step=0.1)


system = load_system(system_name)
num_tasks = system.dimension

system_matrices = []
equilibriums = []
centers = []
for i in range(len(equilibrium_controls)):
    system_matrix , x_e = system.get_ODEmatrix(equilibrium_controls[i])
    system_matrices.append(system_matrix)
    equilibriums.append(torch.tensor(x_e))
    centers.append(torch.tensor([x_e]))

#l = 44194
w_func = Gaussian_Weight(centers[0])
d = w_func.covar_dist(centers[-1], w_func.center, square_dist=True)
l = d*torch.sqrt(torch.tensor(2))/8

# l = l*4

u = np.ones((sim_time.count,1)) * u_ctrl * system.param.u


_train_x, _train_y= simulate_system(system, x0, sim_time, u)
train_x, train_y = downsample_data(_train_x, _train_y, downsample)

likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
model = Local_GP_Sum(
    train_x, 
    train_y, 
    likelihood, 
    num_tasks, 
    system_matrices, 
    equilibriums, 
    centers,
    Gaussian_Weight, 
    weight_lengthscale=l)#, 
model._optimize(optim_steps_single)
# model.optimize(optim_steps, verbose=True)

test_x = test_time.linspace()
model.eval()
likelihood.eval()


#output = model(test_x)
# with torch.no_grad() and gpytorch.settings.debug(False):
#     output = likelihood(model(test_x))
#     estimate = output.mean
with torch.no_grad():
    estimate, cov, weights = model.predict(test_x)


train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension,train_time)
test_data = Data_Def(test_x.numpy(), estimate.detach().numpy(), system.state_dimension, system.control_dimension, test_time)

plot_results(train_data, test_data)
plot_weights(test_x, weights, title="Weighting Function")


if local_predictions:
    for i in range(len(model.models)):
        with torch.no_grad() and gpytorch.settings.debug(False):
            output = likelihood(model.models[i](test_x))
            estimate = output.mean
        train_data = Data_Def(train_x.numpy(), train_y.numpy(), system.state_dimension, system.control_dimension,train_time)
        test_data = Data_Def(test_x.numpy(), estimate.detach().numpy(), system.state_dimension, system.control_dimension, test_time)

        plot_results(train_data, test_data)


states = State_Description(
    equilibrium=equilibriums[-1],
    # equilibrium=torch.stack(equilibriums), 
    init=x0, 
    min=None, max=None)
# ----------------------------------------------------------------------------  


if SAVE:
    config['model_id'] = MODEL_ID
    config['simulation_id'] = SIM_ID
    save_everything(
        system_name, 
        model_path, 
        config, 
        train_data, 
        test_data, 
        sim_data=None, 
        init_state=states.init.numpy(), 
        system_param=states.equilibrium.numpy(), 
        model_dict=model.state_dict()
    )