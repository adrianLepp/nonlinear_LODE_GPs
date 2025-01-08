import gpytorch.constraints
import gpytorch.constraints
from helpers import *
import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from kernels import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from  lodegp import LODEGP, optimize_gp
from helpers import *
from likelihoods import *
from masking import *
from mean_modules import Equilibrium_Mean

def update_gp(model:LODEGP, train_x, train_y, noise, mask):
    return

def inference_mpc_gp(model:LODEGP, test_x, mask):
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        outputs = model(test_x)
        predictions = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=test_x, mask=mask)
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    return mean

def optimize_mpc_gp(gp:LODEGP, train_x, mask_stacked, training_iterations=100, verbose=True):
    gp.train()
    gp.likelihood.train()

    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1) 

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = gp(train_x)
        loss = -mll(output, gp.train_targets)

        loss.backward()
        if verbose is True:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

        # enforce constraints (heuristics)
        #FIXME: system specific
        # gp.covar_module.model_parameters.signal_variance_2 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.signal_variance_2))
        # max_signal_variance = 1
        # if gp.covar_module.model_parameters.signal_variance_2 > max_signal_variance:
        #     gp.covar_module.model_parameters.signal_variance_2 = torch.nn.Parameter(torch.tensor(max_signal_variance))
        # min_signal_variance = 1e-5
        # if gp.covar_module.model_parameters.signal_variance_2 < min_signal_variance:
        #     gp.covar_module.model_parameters.signal_variance_2 = torch.nn.Parameter(torch.tensor(min_signal_variance))

        # gp.covar_module.model_parameters.lengthscale_3 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.lengthscale_3))
        # min_lengthscale = 3.0
        # if gp.covar_module.model_parameters.lengthscale_2 < min_lengthscale:
        #     gp.covar_module.model_parameters.lengthscale_2 = torch.nn.Parameter(torch.tensor(min_lengthscale))

        # max_lengthscale = 3.0
        # if gp.covar_module.model_parameters.lengthscale_2 > max_lengthscale:
        #     gp.covar_module.model_parameters.lengthscale_2 = torch.nn.Parameter(torch.tensor(max_lengthscale))


    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))

def pretrain(system_matrix, num_tasks:int, time_obj:Time_Def, optim_steps:int, reference_strategie:dict, states:State_Description):
    
    trajectory, trajectory_time, train_noise = create_setpoints(reference_strategie, time_obj, states)

    train_x = torch.tensor(trajectory_time)
    train_y = torch.tensor(trajectory)# - torch.tensor(states.equilibrium)

    noise_constraint = gpytorch.constraints.Interval([1e-9,1e-9,1e-13],[1e-7,1e-7,1e-11])  #TODO
    likelihood = MultitaskGaussianLikelihoodWithMissingObs(num_tasks=num_tasks, original_shape=train_y.shape, noise_constraint=noise_constraint) #, has_task_noise=False
    mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
    model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)

    manual_noise = torch.tensor(train_noise)
    _, mask = create_mask(train_y)
    noise_strat = MaskedManualNoise(mask, manual_noise)
    model.likelihood.set_noise_strategy(noise_strat)

    model.mask = mask
    model.equilibrium = states.equilibrium

    optimize_mpc_gp(model, train_x, mask_stacked=torch.ones((train_x.shape[0], num_tasks)).bool(), training_iterations=optim_steps)

    return model, mask


def mpc_algorithm(system:ODE_System, model:LODEGP, states:State_Description,  t_end, dt_control, reference_strategy:dict, optim_steps=10, dt_step = 0.1, plot_single_steps=False ):
    # init time
    step_count = int(ceil(dt_control / dt_step))
    #dt_step =  dt_control /step_count
    control_count = int(t_end / dt_control)

    factor = 2

    sim_time = np.linspace(0, t_end * factor, factor * control_count * step_count + 1)
    gp_time = np.linspace(0, t_end , control_count * step_count + 1)

    # init states
    x_sim = np.zeros((factor * control_count * step_count + 1, system.dimension)) 
    x_sim[0] = states.init
    u_sim = np.zeros((factor * control_count * step_count + 1, system.control_dimension)) 
    u_sim[0] = states.init[system.state_dimension::]

    x_lode = np.zeros((control_count * step_count + 1, system.dimension))
    x_lode[0] = states.init

    # x_ref = np.zeros((control_count + 1, system.dimension))
    # t_ref = np.zeros(control_count + 1)
    x_ref = np.array([states.init, states.target])
    t_ref = np.array([0, t_end])
    num_tasks = system.dimension

    # control loop
    for i in range(control_count):
        # init time
        t_i = i * dt_control
        x_i = x_sim[i*step_count]
        control_time = Time_Def(t_i, t_end, step=dt_control)#* dt_step TODO: dt_step
        step_time = Time_Def(t_i, t_i + dt_control , step=dt_step)

        # generate training Data
        trajectory, trajectory_time, train_noise = create_setpoints(reference_strategy, control_time, states, x_i)
        manual_noise = torch.tensor(train_noise)

        # x_ref[i] = x_i
        # t_ref[i] = t_i

        # train gp model

        # v1
        #model.likelihood.set_noise(torch.tensor(train_noise))
        #model.set_train_data(torch.tensor(trajectory_time), torch.tensor(trajectory) - torch.tensor(states.equilibrium), strict=False)

        #v2
        train_y_masked, mask = create_mask(torch.tensor(trajectory))# - torch.tensor(states.equilibrium)
        model.mask = mask
        model.set_train_data(torch.tensor(trajectory_time), train_y_masked, strict=False)
        noise_strategy = MaskedManualNoise(mask, manual_noise)
        model.likelihood.set_noise_strategy(noise_strategy)

        optimize_mpc_gp(model,torch.tensor(trajectory_time), mask_stacked=torch.ones((torch.tensor(trajectory_time).shape[0], num_tasks)).bool(), training_iterations=optim_steps, verbose=False)

        #print(f'Iter {i}, time: {t_i}')
        # named_parameters = list(model.named_parameters())
        # for j in range(len(named_parameters)):
        #     print(named_parameters[j][0], named_parameters[j][1].data) #.item()


        # prediction
        
        #model.likelihood.eval()
        reference_time = torch.linspace(step_time.start, step_time.end, step_time.count+1)
        reference = inference_mpc_gp(model, reference_time, mask).numpy()
         
        # outputs = model(reference_time)
        # reference = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=reference_time, mask=mask).mean.numpy()# + states.equilibrium

        x_lode[i*step_count+1:(i+1)*step_count+1] = reference[1::]

        u_ref = reference[:,system.state_dimension:system.state_dimension+system.control_dimension]#.flatten()
        #u_ref = x_lode[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()

        # simulate system
        #u_sim[i*step_count+1:(i+1)*step_count+1] = u_ref[0:-1]
        u_sim[i*step_count+1:(i+1)*step_count+1] = u_ref[1::]


        #t_step , x_sim_step= simulate_system(system, x_i[0:system.state_dimension], step_time.start, step_time.end, step_time.count, u_ref , linear=False)
        sol = solve_ivp(system.stateTransition, [step_time.start, step_time.end], x_i[0:system.state_dimension], method='RK45', t_eval=reference_time.numpy(), args=(u_sim, step_time.step ))#, max_step=dt ,  atol = 1, rtol = 1
        x_sim_current = np.concatenate([sol.y.transpose()[1::], u_ref[1::]], axis=1)


        #x_sim[i*step_count+1:(i+1)*step_count+1] = x_sim_step.numpy()
        x_sim[i*step_count+1:(i+1)*step_count+1] =    x_sim_current

        if plot_single_steps:
            train_data = Data_Def(trajectory_time, trajectory, system.state_dimension, system.control_dimension)
            test_data = Data_Def(reference_time.numpy(), reference, system.state_dimension, system.control_dimension)
            sim_data = Data_Def(reference_time.numpy(), x_sim_current, system.state_dimension, system.control_dimension)
            plot_results(train_data, test_data, sim_data)# 

    
    
    # x_ref[-1] = states.target
    # t_ref[-1] = t_end

    u_sim[(i+1)*step_count+1::] = states.equilibrium[system.state_dimension::]
    x_i = x_sim[(i+1)*step_count]
    reference_time = np.linspace(t_end, t_end*factor, int((t_end*factor - t_end)/dt_step+1))


    sol = solve_ivp(system.stateTransition, [t_end, t_end * factor], x_i[0:system.state_dimension], method='RK45', t_eval=reference_time, args=(u_sim, step_time.step ))#, max_step=dt ,  atol = 1, rtol = 1
    x_sim_current = np.concatenate([sol.y.transpose()[1::], u_sim[(i+1)*step_count:-1]], axis=1)

    x_sim[(i+1)*step_count+1::] =    x_sim_current
    
    sim_data = Data_Def(sim_time, x_sim, system.state_dimension, system.control_dimension)
    lode_data = Data_Def(gp_time, x_lode, system.state_dimension, system.control_dimension)
    train_data = Data_Def(t_ref, x_ref, system.state_dimension, system.control_dimension)

    return sim_data, train_data, lode_data

def create_setpoints(reference_strategy:dict, time_obj:Time_Def, states:State_Description, x_0=None):
    a = 1

    constraint_points = reference_strategy['constraints']

    if x_0 is not  None:
        states.init = x_0

    if reference_strategy['target'] is True:
        a += 1

    start_noise = (states.max-states.min) * reference_strategy['start_noise']
    end_noise = (states.max-states.min) * reference_strategy['end_noise']

    x_mean = (states.max + states.min) / 2
    x_noise = ((states.max - states.min) / 4) #np.sqrt

    end_time = time_obj.start + constraint_points * time_obj.step

    if end_time > time_obj.end:
        end_time = time_obj.end - time_obj.step
        constraint_points = int((end_time - time_obj.start) / time_obj.step)
        

    trajectory_time = np.linspace(time_obj.start, end_time, constraint_points+a).flatten()
    trajectory = np.tile(x_mean, (constraint_points+a, 1))
    noise = np.tile(x_noise, (constraint_points+a,))
    
    trajectory[0,:] = states.init
    noise[0:x_noise.shape[0]] = start_noise
    

    if reference_strategy['target'] is True:
        trajectory_time[-1] = time_obj.end
        trajectory[-1,:] = states.target
        noise[-x_noise.shape[0]::] = end_noise

    return trajectory, trajectory_time, noise