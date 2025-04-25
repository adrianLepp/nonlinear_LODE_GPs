import gpytorch.constraints
import gpytorch.constraints
from nonlinear_LODE_GPs.helpers import *
import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from nonlinear_LODE_GPs.kernels import *
import torch
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import LODEGP, optimize_gp
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.likelihoods import *
from nonlinear_LODE_GPs.masking import *
from nonlinear_LODE_GPs.mean_modules import Equilibrium_Mean

def update_gp(model:LODEGP, train_x:torch.Tensor, train_y:torch.Tensor, manual_noise:torch.Tensor, optim_steps:int=0):
    if isinstance(model.likelihood, MultitaskGaussianLikelihoodWithMissingObs):
        train_y_masked, mask = create_mask(train_y)
        model.mask = mask
        model.set_train_data(train_x, train_y_masked, strict=False)
        noise_strategy = MaskedManualNoise(mask, manual_noise)
        model.likelihood.set_noise_strategy(noise_strategy)
    else:
        #model.likelihood.set_noise(torch.tensor(train_noise))
        model.set_train_data(train_x, train_y, strict=False)

    if isinstance(model.likelihood, FixedTaskNoiseMultitaskLikelihood):
        model.likelihood.set_task_noise(manual_noise)

    if optim_steps > 0:
        optimize_mpc_gp(model,train_x, training_iterations=optim_steps, verbose=False)

def inference_mpc_gp(model:LODEGP, test_x:torch.Tensor):
    model.eval()
    model.likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        noise  = torch.ones(torch.Size((model.num_tasks,test_x.shape[0]))).flatten() * 1e-4
        outputs = model(test_x) #, noise
        if isinstance(model.likelihood, MultitaskGaussianLikelihoodWithMissingObs):
            predictions = model.likelihood(outputs, train_data=model.train_inputs[0], current_data=test_x, mask=model.mask)
        else:
            predictions = model.likelihood(outputs, noise = noise) #noise = torch.ones_like(test_x) * 1e-4
        mean = predictions.mean
        #lower, upper = predictions.confidence_region()

    return mean

def optimize_mpc_gp(gp:LODEGP, train_x, training_iterations=100, verbose=True):
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

    print("\n----------------------------------------------------------------------------------\n")
    print('Trained model parameters:')
    named_parameters = list(gp.named_parameters())
    param_conversion = torch.nn.Softplus()

    raw = False
    for j in range(len(named_parameters)):
        if raw is True:
            print(named_parameters[j][0], named_parameters[j][1].data) #.item()
        else :
            print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()

    
    optimizer.zero_grad()
    output = gp(train_x)
    loss = -mll(output, gp.train_targets)
    print(f"loss: {loss.item()}")
    print("\n----------------------------------------------------------------------------------\n")

        # enforce constraints (heuristics)
        #gp.covar_module.model_parameters.signal_variance_2 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.signal_variance_2))
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
    
    train_y, train_x, manual_noise = create_setpoints(reference_strategie, time_obj, states)

    #noise_constraint = gpytorch.constraints.Interval([1e-9,1e-9,1e-13],[1e-7,1e-7,1e-11])
    likelihood = MultitaskGaussianLikelihoodWithMissingObs(
        num_tasks=num_tasks, 
        original_shape=train_y.shape, 
        #noise_constraint=noise_constraint
        ) #, has_task_noise=False
    mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
    model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)

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

    optimize_mpc_gp(model, train_x, training_iterations=optim_steps)


    return model, mask


def predict_reference(model:LODEGP, step_time:Time_Def, states:State_Description, x_i:np.ndarray, t_i:float, convergence:bool, EARLY_CONVERGENCE:bool=False):
    t_reference = step_time.linspace()

    if EARLY_CONVERGENCE and np.isclose(x_i[0:2], states.target[0:2], atol=5e-3).all(): #, atol=1e-3
        if convergence is False:
            convergence = True
            print(f'Target reached at time {t_i}')

        reference = np.tile(states.target, (step_time.count, 1)) #FIXME
    
    else:
        _reference = inference_mpc_gp(model, t_reference)
        if isinstance(model.likelihood,MultitaskGaussianLikelihoodWithMissingObs) and any(model.mask) is True:
            reference = stack_plot_tensors(_reference, model.num_tasks).numpy()
        else:
            reference = _reference.numpy()
    
    return t_reference, reference, convergence
        

def mpc_algorithm(system:ODE_System, model:LODEGP, states:State_Description, reference_strategy:dict, control_time:Time_Def, sim_time:Time_Def, optim_steps=10, plot_single_steps=False ):
    convergence  = False
    EARLY_CONVERGENCE = False
    settling_time = None
    # init time
    step_count = int(ceil(control_time.step / sim_time.step))
    
    gp_steps =   int((control_time.count-1) * step_count + 1) #TODO +1
    sim_steps = int(sim_time.count ) #TODO + 1

    t_sim = np.linspace(0, sim_time.end, sim_steps)
    t_gp = np.linspace(0, control_time.end , gp_steps)

    # init states
    x_sim = np.zeros((sim_steps, system.dimension)) 
    x_sim[0] = states.init
    u_sim = np.zeros((sim_steps, system.control_dimension)) 
    u_sim[0] = states.init[system.state_dimension::]

    x_lode = np.zeros((gp_steps, system.dimension))
    x_lode[0] = states.init

    num_tasks = system.dimension

    x_ref = np.array([states.init, states.target])
    t_ref = np.array([0, control_time.end])

    # control loop
    for i in range(control_time.count - 1):
        # init time
        t_i = i * control_time.step
        x_i = x_sim[i*step_count]
        ref_time = Time_Def(t_i, control_time.end, step=control_time.step)#* dt_step TODO: dt_step
        step_time = Time_Def(t_i, t_i + control_time.step , step=sim_time.step)
        #print(f'Iter {i}, time: {t_i}')
        start_idx = i*step_count+1
        end_idx = (i+1)*step_count+1

        t_past = torch.linspace(t_i - reference_strategy['past-values'] * control_time.step, t_i-control_time.step, reference_strategy['past-values'])
        t_past = t_past[t_past >= 0]
        past = torch.tensor(np.array([x_sim[np.where(t_sim == t)].flatten() for t in t_past]))
        
        noise = torch.tensor([1e-5, 1e-5, 1e-7])
        x_i_noise = torch.tensor(x_i)+ torch.randn(x_i.shape) * noise
        setpoint, t_setpoint, manual_noise = create_setpoints(reference_strategy, ref_time, states, x_i_noise, t_past, past)
        update_gp(model, t_setpoint, setpoint, manual_noise, optim_steps)
        t_reference, reference,  convergence = predict_reference(model, step_time, states, x_i, t_i, convergence, EARLY_CONVERGENCE)

        if convergence is True and settling_time is None:
            settling_time = t_i
            # print(f'Settling time: {settling_time}')
        
        x_lode[start_idx:end_idx] = reference[1::]
        u_ref = reference[:,system.state_dimension:system.state_dimension+system.control_dimension]#.flatten()

        # simulate system
        u_sim[start_idx:end_idx] = u_ref[1::]
        sol = solve_ivp(system.stateTransition, [step_time.start, step_time.end], x_i[0:system.state_dimension], method='RK45', t_eval=t_reference.numpy(), args=(u_sim, step_time.step ), max_step=step_time.step)#, max_step=dt ,  atol = 1, rtol = 1
        x_sim_current = np.concatenate([sol.y.transpose()[1::], u_ref[1::]], axis=1)
        x_sim[start_idx:end_idx] = x_sim_current

        if plot_single_steps and t_i > 1000:
            t_reference_2 = torch.linspace(step_time.start, control_time.end, control_time.count +1)
            reference_2 = inference_mpc_gp(model, t_reference_2, model.mask).numpy()

            train_data = Data_Def(t_setpoint.numpy(), setpoint.numpy(), system.state_dimension, system.control_dimension)
            test_data = Data_Def(t_reference_2.numpy(), reference_2, system.state_dimension, system.control_dimension)
            sim_data = Data_Def(t_reference.numpy()[1::], x_sim_current, system.state_dimension, system.control_dimension)
            plot_results(train_data, test_data, sim_data)# 

    # simulate the remainding time with constant control input 
    #Shouldnt happen anymore:
    # if sim_time.end > control_time.end:

    #     if convergence is True and settling_time is None:
    #         settling_time = t_i

    #     u_sim[(i+1)*step_count+1::] = states.equilibrium[system.state_dimension::]
    #     x_i = x_sim[(i+1)*step_count]
    #     t_reference = np.linspace(control_time.end, sim_time.end, int((sim_time.end - control_time.end)/sim_time.step)) #TODO + 1

    #     sol = solve_ivp(system.stateTransition, [control_time.end, sim_time.end], x_i[0:system.state_dimension], method='RK45', t_eval=t_reference, args=(u_sim, step_time.step ))#, max_step=dt ,  atol = 1, rtol = 1
    #     #x_sim_current = np.concatenate([sol.y.transpose()[1::], u_sim[(i+1)*step_count:-1]], axis=1)
    #     x_sim_current = np.concatenate([sol.y.transpose(), u_sim[(i+1)*step_count:-1]], axis=1)

    #     x_sim[(i+1)*step_count+1::] =    x_sim_current
    
    # x_lode[-1] = x_lode[-2] #FIXME  
    sim_data = Data_Def(t_sim, x_sim, system.state_dimension, system.control_dimension)
    lode_data = Data_Def(t_gp, x_lode, system.state_dimension, system.control_dimension)
    train_data = Data_Def(t_ref, x_ref, system.state_dimension, system.control_dimension)

    return sim_data, train_data, lode_data, settling_time

def create_setpoints(reference_strategy:dict, time_obj:Time_Def, states:State_Description, x_0=None, t_past:torch.Tensor=None, past:torch.Tensor=None):
    a = 1

    constraint_points = reference_strategy['constraints']

    if x_0 is not  None:
        states.init = x_0

    if reference_strategy['target'] is True:
        a += 1

    # init_noise = (states.max-states.min) * reference_strategy['init_noise']
    # target_noise = (states.max-states.min) * reference_strategy['target_noise']
    init_noise = torch.tensor(reference_strategy['init_noise'])
    target_noise = torch.tensor(reference_strategy['target_noise'])

    x_mean = (states.max + states.min) / 2
    x_noise = ((states.max - states.min) / 4) #np.sqrt

    # if reference_strategy['soft_constraints'] == 'equilibrium':
    #     x_mean[2] = torch.nan

    end_time = time_obj.start + constraint_points * time_obj.step

    if end_time > time_obj.end:
        end_time = time_obj.end  #FIXME: - time_obj.step
        constraint_points = int((end_time - time_obj.start) / time_obj.step)
        

    t_setpoint = torch.linspace(time_obj.start, end_time + time_obj.step* (a-1) , constraint_points+a).flatten()
    setpoint = torch.tile(x_mean, (constraint_points+a, 1))
    noise = torch.tile(x_noise, (constraint_points+a,))
    
    setpoint[0,:] = states.init
    #setpoint[0,2] = torch.nan

    noise[0:x_noise.shape[0]] = init_noise
    

    if reference_strategy['target'] is True:
        t_setpoint[-1] = 100 # time_obj.end 
        setpoint[-1,:] = states.target
        # setpoint[-1,2] = torch.nan
        #noise[-x_noise.shape[0]::] = target_noise

        noise[-x_noise.shape[0]::] = target_noise
        #noise[-1] = x_noise[2] / 4
        #noise[2] = x_noise[2] / 8

        # setpoint[-1,2] = x_mean[2]
        # noise[-1] = x_noise[2]
    
    if t_past is not None and past is not None:
        t_setpoint = torch.cat((t_past, t_setpoint))
        setpoint = torch.cat((past, setpoint))
        noise = torch.cat((torch.tile(init_noise*1e5,(t_past.shape[0],)), noise))

    return setpoint, t_setpoint, noise