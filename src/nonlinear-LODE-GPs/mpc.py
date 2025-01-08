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

    optimizer = torch.optim.Adam(
        #params=list(set(gp.parameters()) - {gp.mean_module.parameters(),gp.likelihood.parameters(),  }),
        gp.parameters(), 
        lr=0.1
    ) 

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
        gp.covar_module.model_parameters.signal_variance_2 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.signal_variance_2))
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

def pretrain(system_matrix, num_tasks:int, time_obj:Time_Def, optim_steps:int, reference_strategie:dict, states:State_Description, hyperparameters:dict=None):
    
    trajectory, t_trajectory, train_noise = create_setpoints(reference_strategie, time_obj, states)

    train_x = torch.tensor(t_trajectory)
    train_y = torch.tensor(trajectory)# - torch.tensor(states.equilibrium)

    #noise_constraint = gpytorch.constraints.Interval([1e-9,1e-9,1e-13],[1e-7,1e-7,1e-11])
    likelihood = MultitaskGaussianLikelihoodWithMissingObs(
        num_tasks=num_tasks, 
        original_shape=train_y.shape, 
        #noise_constraint=noise_constraint
        ) #, has_task_noise=False
    mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
    model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)

    manual_noise = torch.tensor(train_noise)
    _, mask = create_mask(train_y)
    noise_strat = MaskedManualNoise(mask, manual_noise)
    model.likelihood.set_noise_strategy(noise_strat)

    model.mask = mask
    model.equilibrium = states.equilibrium

    # hyperparameter constraints
    if hyperparameters is not None:
        for key, value in hyperparameters.items():
            if hasattr(model.covar_module.model_parameters, key):
                setattr(model.covar_module.model_parameters, key, torch.nn.Parameter(torch.tensor(value), requires_grad=False))
            else:
                print(f'Hyperparameter {key} not found in model')

    optimize_mpc_gp(model, train_x, mask_stacked=torch.ones((train_x.shape[0], num_tasks)).bool(), training_iterations=optim_steps)

    print("\n----------------------------------------------------------------------------------\n")
    print('Trained model parameters:')
    named_parameters = list(model.named_parameters())
    for j in range(len(named_parameters)):
        print(named_parameters[j][0], named_parameters[j][1].data) #.item()
    print("\n----------------------------------------------------------------------------------\n")


    return model, mask


def mpc_algorithm(system:ODE_System, model:LODEGP, states:State_Description, reference_strategy:dict, control_time:Time_Def, sim_time:Time_Def, optim_steps=10, plot_single_steps=False ):
    # init time
    step_count = int(ceil(control_time.step / sim_time.step))
    #dt_step =  dt_control /step_count
    #control_count = int(control_time.end / control_time.step)
    

    #factor = 1.5
    gp_steps =   int(control_time.count * step_count + 1)
    sim_steps = int(sim_time.count + 1)

    t_sim = np.linspace(0, sim_time.end, sim_steps)
    t_gp = np.linspace(0, control_time.end , gp_steps)

    # init states
    x_sim = np.zeros((sim_steps, system.dimension)) 
    x_sim[0] = states.init
    u_sim = np.zeros((sim_steps, system.control_dimension)) 
    u_sim[0] = states.init[system.state_dimension::]

    x_lode = np.zeros((gp_steps, system.dimension))
    x_lode[0] = states.init

    # x_ref = np.zeros((control_count + 1, system.dimension))
    # t_setpoint = np.zeros(control_count + 1)
    num_tasks = system.dimension

    x_ref = np.array([states.init, states.target])
    t_ref = np.array([0, control_time.end])

    # control loop
    for i in range(control_time.count):
        # init time
        t_i = i * control_time.step
        x_i = x_sim[i*step_count]
        ref_time = Time_Def(t_i, control_time.end, step=control_time.step)#* dt_step TODO: dt_step
        step_time = Time_Def(t_i, t_i + control_time.step , step=sim_time.step)

        # generate training Data
        trajectory, t_trajectory, train_noise = create_setpoints(reference_strategy, ref_time, states, x_i)
        manual_noise = torch.tensor(train_noise)

        # x_ref[i] = x_i
        # t_setpoint[i] = t_i

        # train gp model

        # v1
        #model.likelihood.set_noise(torch.tensor(train_noise))
        #model.set_train_data(torch.tensor(t_trajectory), torch.tensor(trajectory) - torch.tensor(states.equilibrium), strict=False)

        #v2
        train_y_masked, mask = create_mask(torch.tensor(trajectory))# - torch.tensor(states.equilibrium)
        model.mask = mask
        model.set_train_data(torch.tensor(t_trajectory), train_y_masked, strict=False)
        noise_strategy = MaskedManualNoise(mask, manual_noise)
        model.likelihood.set_noise_strategy(noise_strategy)

        optimize_mpc_gp(model,torch.tensor(t_trajectory), mask_stacked=torch.ones((torch.tensor(t_trajectory).shape[0], num_tasks)).bool(), training_iterations=optim_steps, verbose=False)

        #print(f'Iter {i}, time: {t_i}')
        # prediction
        
        #model.likelihood.eval()
        t_setpoint = torch.linspace(step_time.start, step_time.end, step_time.count+1)
        setpoint = inference_mpc_gp(model, t_setpoint, mask).numpy()
         
        # outputs = model(t_setpoint)
        # setpoint = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=t_setpoint, mask=mask).mean.numpy()# + states.equilibrium

        x_lode[i*step_count+1:(i+1)*step_count+1] = setpoint[1::]

        u_ref = setpoint[:,system.state_dimension:system.state_dimension+system.control_dimension]#.flatten()
        #u_ref = x_lode[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()

        # simulate system
        #u_sim[i*step_count+1:(i+1)*step_count+1] = u_ref[0:-1]
        u_sim[i*step_count+1:(i+1)*step_count+1] = u_ref[1::]


        #t_step , x_sim_step= simulate_system(system, x_i[0:system.state_dimension], step_time.start, step_time.end, step_time.count, u_ref , linear=False)
        sol = solve_ivp(system.stateTransition, [step_time.start, step_time.end], x_i[0:system.state_dimension], method='RK45', t_eval=t_setpoint.numpy(), args=(u_sim, step_time.step ))#, max_step=dt ,  atol = 1, rtol = 1
        x_sim_current = np.concatenate([sol.y.transpose()[1::], u_ref[1::]], axis=1)


        #x_sim[i*step_count+1:(i+1)*step_count+1] = x_sim_step.numpy()
        x_sim[i*step_count+1:(i+1)*step_count+1] =    x_sim_current

        if plot_single_steps:
            train_data = Data_Def(t_trajectory, trajectory, system.state_dimension, system.control_dimension)
            test_data = Data_Def(t_setpoint.numpy(), setpoint, system.state_dimension, system.control_dimension)
            sim_data = Data_Def(t_setpoint.numpy(), x_sim_current, system.state_dimension, system.control_dimension)
            plot_results(train_data, test_data, sim_data)# 

    
    
    # x_ref[-1] = states.target
    # t_setpoint[-1] = t_end

    u_sim[(i+1)*step_count+1::] = states.equilibrium[system.state_dimension::]
    x_i = x_sim[(i+1)*step_count]
    t_setpoint = np.linspace(control_time.end, sim_time.end, int((sim_time.end - control_time.end)/sim_time.step+1))

    sol = solve_ivp(system.stateTransition, [control_time.end, sim_time.end], x_i[0:system.state_dimension], method='RK45', t_eval=t_setpoint, args=(u_sim, step_time.step ))#, max_step=dt ,  atol = 1, rtol = 1
    x_sim_current = np.concatenate([sol.y.transpose()[1::], u_sim[(i+1)*step_count:-1]], axis=1)

    x_sim[(i+1)*step_count+1::] =    x_sim_current
    
    sim_data = Data_Def(t_sim, x_sim, system.state_dimension, system.control_dimension)
    lode_data = Data_Def(t_gp, x_lode, system.state_dimension, system.control_dimension)
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
    x_noise = ((states.max - states.min) / 8) #np.sqrt

    end_time = time_obj.start + constraint_points * time_obj.step

    if end_time > time_obj.end:
        end_time = time_obj.end - time_obj.step
        constraint_points = int((end_time - time_obj.start) / time_obj.step)
        

    t_trajectory = np.linspace(time_obj.start, end_time + time_obj.step* (a-1) , constraint_points+a).flatten()
    trajectory = np.tile(x_mean, (constraint_points+a, 1))
    noise = np.tile(x_noise, (constraint_points+a,))
    
    trajectory[0,:] = states.init
    noise[0:x_noise.shape[0]] = start_noise
    

    if reference_strategy['target'] is True:
        t_trajectory[-1] = time_obj.end
        trajectory[-1,:] = states.target
        noise[-x_noise.shape[0]::] = end_noise

        # trajectory[-1,2] = x_mean[2]
        # noise[-1] = x_noise[2]

    return trajectory, t_trajectory, noise