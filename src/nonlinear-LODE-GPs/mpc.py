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
        gp.covar_module.model_parameters.signal_variance_2 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.signal_variance_2))
        max_signal_variance = 1
        if gp.covar_module.model_parameters.signal_variance_2 > max_signal_variance:
            gp.covar_module.model_parameters.signal_variance_2 = torch.nn.Parameter(torch.tensor(max_signal_variance))

        # gp.covar_module.model_parameters.lengthscale_3 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.lengthscale_3))
        min_lengthscale = 3.0
        if gp.covar_module.model_parameters.lengthscale_2 < min_lengthscale:
            gp.covar_module.model_parameters.lengthscale_2 = torch.nn.Parameter(torch.tensor(min_lengthscale))


    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))

def pretrain(system_matrix, num_tasks:int, time_obj:Time_Def, optim_steps:int, reference_strategie:int, states:State_Description):
    
    trajectory, trajectory_time, train_noise = create_reference(reference_strategie, time_obj, states)

    train_x = torch.tensor(trajectory_time)
    train_y = torch.tensor(trajectory)# - torch.tensor(states.equilibrium)

    likelihood = MultitaskGaussianLikelihoodWithMissingObs(num_tasks=num_tasks, original_shape=train_y.shape)
    mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
    model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)

    manual_noise = torch.tensor(train_noise)
    _, mask = create_mask(train_y)
    noise_strat = MaskedManualNoise(mask, manual_noise)
    likelihood.set_noise_strategy(noise_strat)

    model.mask = mask
    model.equilibrium = states.equilibrium

    optimize_mpc_gp(model, train_x, mask_stacked=torch.ones((train_x.shape[0], num_tasks)).bool(), training_iterations=optim_steps)

    return model, mask


def mpc_algorithm(system:ODE_System, model:LODEGP, states:State_Description,  t_end, dt_control, reference_strategy=1, optim_steps=10, dt_step = 0.1, plot_single_steps=False ):
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

    x_ref = np.zeros((control_count + 1, system.dimension))
    t_ref = np.zeros(control_count + 1)
    num_tasks = system.dimension

    # control loop
    for i in range(control_count):
        # init time
        t_i = i * dt_control
        x_i = x_sim[i*step_count]
        control_time = Time_Def(t_i, t_end, step=dt_control)#* dt_step
        step_time = Time_Def(t_i, t_i + dt_control , step=dt_step)

        # generate training Data
        trajectory, trajectory_time, train_noise = create_reference(reference_strategy, control_time, states, x_i)
        manual_noise = torch.tensor(train_noise)

        x_ref[i] = x_i
        t_ref[i] = t_i

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

        print(f'Iter {i}, time: {t_i}')
        named_parameters = list(model.named_parameters())
        for j in range(len(named_parameters)):
            print(named_parameters[j][0], named_parameters[j][1].data) #.item()


        # prediction
        model.eval()
        model.likelihood.eval()
        with torch.no_grad():
            reference_time = torch.linspace(step_time.start, step_time.end, step_time.count+1)
            outputs = model(reference_time)
            reference = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=reference_time, mask=mask).mean.numpy()# + states.equilibrium

        x_lode[i*step_count:(i+1)*step_count+1] = reference

        u_ref = reference[:,system.state_dimension:system.state_dimension+system.control_dimension]#.flatten()
        #u_ref = x_lode[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()

        # simulate system
        u_sim[i*step_count+1:(i+1)*step_count+1] = u_ref[0:-1]
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

    
    
    x_ref[-1] = states.target
    t_ref[-1] = t_end

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

def mpc_feed_forward(test_time:Time_Def, x_0, x_e, model:LODEGP, likelihood, system, SIM_ID:int, MODEL_ID:int, model_path, model_dir, optim_steps:int, train_x, train_y):
    cnt = 0
    while cnt < 1:
        optimize_gp(model,optim_steps)
        # create reference and control input
        test_x = create_test_inputs(test_time.count, test_time.step, test_time.start, test_time.end, 1)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            #output = likelihood(model(test_x), train_data=model.train_inputs[0], current_data=test_x, mask=mask)
            output = likelihood(model(test_x))

        train_data = Data_Def(train_x.numpy(), train_y.numpy() + x_e, system.state_dimension, system.control_dimension)
        test_data = Data_Def(test_x.numpy(), output.mean.numpy() + x_e, system.state_dimension, system.control_dimension)

        u_ref = test_data.y[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()
        ref_x, ref_y= simulate_system(system, x_0[0:system.state_dimension], test_time.start, test_time.end, test_time.count, u_ref, linear=False)
        ref_data = Data_Def(ref_x.numpy(), ref_y.numpy(), system.state_dimension, system.control_dimension)

        plot_results(train_data, test_data, ref_data)

        # if SAVE:
        #     save_results(model, equilibrium, x0, SIM_ID, MODEL_ID, CONFIG_FILE, system_name, config, model_path, train_data, test_data, train_time, test_time, ref_data=ref_data, linear=False) 
        #     MODEL_ID += 1
        #     SIM_ID += 1
        #     model_path = f'{model_dir}/{str(MODEL_ID)}{name}.pth'

        cnt += 1  
        
        lengthscale = model.covar_module.model_parameters.lengthscale_3
        model.covar_module.model_parameters.lengthscale_3 = torch.nn.Parameter(lengthscale *0.9, requires_grad=False)

        # signal_variance = model.covar_module.model_parameters.signal_variance_3
        # model.covar_module.model_parameters.signal_variance_3 = torch.nn.Parameter(abs(signal_variance), requires_grad=False)

def create_reference(strategy:int, time_obj:Time_Def, states:State_Description, x_0=None):
    '''
    strategy:
    - 1: one start and one target point
    - 2: smooth transition from start to target point
    - 3: soft constraints at (max + min)/2 with variance (max - min)/2
    - 4: one start point
    '''
    # TODO: noise is system specific and needs to given as parameter
    start_noise = states.max * 1e-8
    end_noise = start_noise
    #start_noise = [1e-8, 1e-8, 1e-8, 1e-10]
    #end_noise = [1e-8, 1e-8, 1e-8, 1e-10]

    if x_0 is not  None:
        states.init = x_0

    if strategy == 1:
        trajectory = [states.init, states.target]
        trajectory_time = [time_obj.start, time_obj.end]

        noise = [start_noise, end_noise]
    elif strategy == 2:
        t_factor = 10
        gain_factor = 10

        trajectory = [states.init, states.init]
        trajectory_time = [time_obj.start - t_factor*2, time_obj.start - t_factor]
        noise = [1, 1]

        for i in range(10):
            trajectory_time.append(time_obj.start + i * t_factor)
            trajectory.append(states.init + (states.target - states.init) * i / gain_factor)
            #trajectory[i][system.state_dimension:] = [0.5 * system.param.u]
            noise.append(1e5)

        for i in range(5):
            trajectory_time.append(time_obj.end + i * t_factor)
            trajectory.append(states.target)
            noise.append(1e4)

    elif strategy == 3:
        if states.max is None or states.min is None:
            raise ValueError("Max and min values are required for reference strategy 3")
        
        x_mean = (states.max + states.min) / 2
        x_noise = (states.max - states.min) / 2

        trajectory_time = np.linspace(time_obj.start, time_obj.end, time_obj.count+1).flatten()

        trajectory = np.tile(x_mean, (time_obj.count+1, 1))
        #trajectory[:, control_dim] = np.nan #FIXME
        trajectory[0,:] = states.init
        trajectory[-1,:] = states.target

        # noise variant 1
        # noise = np.ones((time_obj.count+1,)) * x_noise.mean()
        # noise[0] = start_noise
        # noise[-1] = end_noise

        # noise variant 2
        noise = np.tile(x_noise, (time_obj.count+1,))
        noise[0:x_noise.shape[0]] = start_noise
        noise[-x_noise.shape[0]::] = end_noise


        
    
    elif strategy == 4:
        trajectory = [states.init]
        trajectory_time = [time_obj.start]

        noise = [start_noise]
    else:
        raise ValueError(f"strategy {str(strategy)} not supported")
    
    ## end_noise = start_noise * exp((log(1/start_noise))/(t_end-dt_control)*t_i)
    return trajectory, trajectory_time, noise