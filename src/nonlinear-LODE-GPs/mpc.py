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

def update_gp(model:LODEGP, train_x, train_y, noise, mask):
    return

def optimize_mpc_gp(gp:LODEGP, train_x, mask_stacked, training_iterations=100, verbose=True):
    # Find optimal model hyperparameters
    gp.train()
    gp.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    #print(list(self.named_parameters()))
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = gp(train_x)#FIXME: 
        loss = -mll(output, gp.train_targets)

        gp.eval()
        #list(model.named_parameters())[2][1].requires_grad = True
        #list(model.named_parameters())[3][1].requires_grad = True
        output_eval = gp(train_x)
        mean_eval = output_eval.mean

        loss.backward()
        if verbose is True:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

        gp.train()
        #losses[i] = loss.item()
        gp.covar_module.model_parameters.signal_variance_3 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.signal_variance_3))

    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))

def pretrain(system_matrix, num_tasks:int, time_obj:Time_Def, optim_steps:int, reference_strategie:int, states:State_Description):
    
    trajectory, trajectory_time, train_noise = create_reference(reference_strategie, time_obj, states)

    train_x = torch.tensor(trajectory_time)
    train_y = torch.tensor(trajectory) - torch.tensor(states.equilibrium)

    manual_noise = torch.tensor(train_noise)


    task_noise = [1e-8, 1e-8, 1e-8, 1e-7]
    # likelihood2 = FixedTaskNoiseMultitaskLikelihood(num_tasks=num_tasks ,noise=torch.tensor(train_noise)*1e-8,  rank=0)
    # likelihood3 = FixedTaskNoiseMultitaskLikelihood2(num_tasks=num_tasks ,data_noise=torch.tensor(train_noise), task_noise=task_noise, rank=0)
    likelihood4 = FixedTaskNoiseMultitaskLikelihood_LinOP(num_tasks=num_tasks ,noise=torch.tensor(train_noise), rank=0)
    #likelihood5 = FixedTaskNoiseMultitaskLikelihood_LinOP(num_tasks=num_tasks ,noise=torch.tensor(train_noise),task_noise=task_noise, rank=0)
    likelihood6 = MultitaskGaussianLikelihoodWithMissingObs(num_tasks=num_tasks, original_shape=train_y.shape)

    # manual_noise = 1e-8 * torch.ones(len(train_x)*num_tasks)

    # manual_noise[3] = 1e-10
    # manual_noise[7] = 1e-10
    # # Model 1 - 3
    # manual_noise[5::3] = 2.5
    # manual_noise[4::3] = 1.
    # manual_noise[3::3] = 1.

    _, mask = create_mask(train_y)
    noise_strat = MaskedManualNoise(mask, manual_noise)
    likelihood6.set_noise_strategy(noise_strat)

    model = LODEGP(train_x, train_y, likelihood6, num_tasks, system_matrix)

    optimize_mpc_gp(model, train_x, mask_stacked=torch.ones((train_x.shape[0], num_tasks)).bool(), training_iterations=optim_steps)

    
    #optimize_gp(model,optim_steps)

    return model, mask
    

def mpc_algorithm_2(system:ODE_System, model:LODEGP, predict_ll:gpytorch.likelihoods.Likelihood, states:State_Description,  t_end, dt_control, reference_strategy=1, optim_steps=10, dt_step = 0.1 ):
    # init time
    step_count = int(ceil(dt_control / dt_step))
    #dt_step =  dt_control /step_count
    control_count = int(t_end / dt_control)

    # init states
    x_sim = np.zeros((control_count * step_count + 1, system.dimension))
    x_sim[0] = states.init

    x_lode = np.zeros((control_count * step_count + 1, system.dimension))
    x_lode[0] = states.init

    x_ref = np.zeros((control_count + 1, system.dimension))
    t_ref = np.zeros(control_count + 1)
    num_tasks = system.dimension
    task_noise = [1e-8, 1e-8, 1e-8, 1e-7]

    # misc

    #trajectory, trajectory_time, train_noise = create_reference(1, Time_Def(0,t_end,count=1), x_0, x_target)


    for i in range(control_count):
        # init time
        t_i = i * dt_control
        x_i = x_sim[i*step_count]
        control_time = Time_Def(t_i, t_end, step=dt_control/dt_step)
        step_time = Time_Def(t_i, t_i + dt_control , step=dt_step)

        # generate training Data
        trajectory, trajectory_time, train_noise = create_reference(reference_strategy, control_time, states, x_i)
        manual_noise = torch.tensor(train_noise)
        #trajectory, trajectory_time, train_noise = create_reference(reference_strategy, step_time, x_i, states.target)

        # trajectory = [x_0, x_target]
        # trajectory_time = [step_time.start, step_time.end]

        # start_noise = 1e5
        # end_noise = start_noise * exp((log(1/start_noise))/(t_end-dt_control)*t_i)
        # train_noise = [1, end_noise]
        # print(f"end_noise: {end_noise}")

        x_ref[i] = x_i
        t_ref[i] = t_i

        # train gp model

        # v1
        #model.likelihood.set_noise(torch.tensor(train_noise))
        #model.set_train_data(torch.tensor(trajectory_time), torch.tensor(trajectory) - torch.tensor(states.equilibrium), strict=False)

        #v2
        train_y_masked, mask = create_mask(torch.tensor(trajectory) - torch.tensor(states.equilibrium))
        model.mask = mask
        model.set_train_data(torch.tensor(trajectory_time), train_y_masked, strict=False)
        noise_strategy = MaskedManualNoise(mask, manual_noise)
        model.likelihood.set_noise_strategy(noise_strategy)

        optimize_mpc_gp(model,torch.tensor(trajectory_time), mask_stacked=torch.ones((torch.tensor(trajectory_time).shape[0], num_tasks)).bool(), training_iterations=optim_steps, verbose=False)
        #optimize_gp(model, optim_steps, verbose=False)

        # prediction
        model.eval()
        #predict_ll.eval()
        model.likelihood.eval()
        with torch.no_grad():
            #reference = predict_ll(model(torch.linspace(step_time.start, step_time.end, step_time.count))) 
            current_time = torch.linspace(step_time.start, step_time.end, step_time.count)
            outputs = model(current_time)
            reference = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=current_time, mask=mask)

        x_ref_current = reference.mean.numpy() + states.equilibrium #should be 10x4

        u_ref = x_ref_current[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten() # should be 10x1 

        # simulate system

        t_step , x_sim_step= simulate_system(system, x_i[0:system.state_dimension], 0, step_time.end, step_time.count, u_ref, linear=False) # check t_step

        x_sim[i*step_count+1:(i+1)*step_count+1] = x_sim_step.numpy()
        x_lode[i*step_count+1:(i+1)*step_count+1] = x_ref_current

    sim_time = np.linspace(0, t_end, control_count * step_count + 1)
    sim_data = Data_Def(sim_time, x_sim, system.state_dimension, system.control_dimension)
    lode_data = Data_Def(sim_time, x_lode, system.state_dimension, system.control_dimension)

    #train_data = Data_Def(np.array(trajectory_time), np.array(trajectory), system.state_dimension, system.control_dimension)

    x_ref[-1] = states.target
    t_ref[-1] = t_end
    train_data = Data_Def(t_ref, x_ref, system.state_dimension, system.control_dimension)

    return sim_data, train_data, lode_data


def mpc_algorithm(test_time:Time_Def, x_0, x_target, task_noise, model:LODEGP, system, likelihood, num_tasks:int, optim_steps:int):
    x_sim = np.zeros((test_time.count, system.dimension))
    x_sim[0] = x_0
    x_ref = x_sim.copy()

    for i in range(test_time.count - 1):
        t_i = test_time.start + i * test_time.step
        # generate training Data
        train_time = Time_Def(t_i, test_time.end, step=1)
        trajectory, trajectory_time, train_noise = create_reference(1, train_time, x_sim[i], x_target)

        train_x = torch.tensor(trajectory_time)
        train_y = torch.tensor(trajectory) - torch.tensor(x_target)

        # update GP model
        model.likelihood = FixedTaskNoiseMultitaskLikelihood2(num_tasks=num_tasks ,data_noise=torch.tensor(train_noise), task_noise=task_noise, rank=0)
        model.set_train_data(train_x, train_y, strict=False)
        optimize_gp(model,optim_steps)

        # prediction
        test_x = create_test_inputs(test_time.count-i, test_time.step, t_i, test_time.end, 1)
        model.eval()
        likelihood.eval()
        with torch.no_grad():
            output = likelihood(model(test_x))

        x_ref[i+1] = output.mean[0].numpy() + x_target
        u_ref = x_ref[i+1][system.state_dimension:system.state_dimension+system.control_dimension]
        #u_ref = x_ref[:,system.state_dimension:system.state_dimension+system.control_dimension].flatten()
        
        
        ref_x, ref_y= simulate_system(system, x_sim[i][0:system.state_dimension], 0, test_time.step, 1, u_ref, linear=False)
        x_sim[i+1] = ref_y[0].numpy()

    time = np.linspace(test_time.start, test_time.end, test_time.count)

    ref_data = Data_Def(time, x_sim, system.state_dimension, system.control_dimension)
    test_data = Data_Def(time, x_ref, system.state_dimension, system.control_dimension)
    train_data = Data_Def(np.array(trajectory_time), np.array(trajectory), system.state_dimension, system.control_dimension)

    plot_results(train_data, test_data, ref_data)

    return x_sim, x_ref

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
    start_noise = [1e-8, 1e-8, 1e-8, 1e-10]
    end_noise = [1e-8, 1e-8, 1e-8, 1e-10]
    control_dim = 3

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
    
    return trajectory, trajectory_time, noise