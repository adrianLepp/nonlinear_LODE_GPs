
# from sage.all import *
# import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from nonlinear_LODE_GPs.kernels import *
import torch

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.likelihoods import *
from nonlinear_LODE_GPs.masking import *
from nonlinear_LODE_GPs.mpc import mpc_algorithm, pretrain, optimize_mpc_gp, create_setpoints
from nonlinear_LODE_GPs.lodegp import LODEGP
from nonlinear_LODE_GPs.mean_modules import Equilibrium_Mean

from result_reporter.latex_exporter import save_plot_to_pdf, create_mpc_plot

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False
CONFIG_FILE = 'config.json'

system_name = "nonlinear_watertank"







def run_sim(lengthscale=None, signal_variance=None):
    print("\n----------------------------------------------------------------------------------\n")
    try:
        with open(CONFIG_FILE,"r") as f:
            config = json.load(f)
            model_dir=config['model_dir']
            data_dir=config['data_dir']
            model_name = config['model_name']

            if SAVE:
                global SIM_ID, MODEL_ID
                SIM_ID = config['simulation_id'] + 1
                MODEL_ID = config['model_id'] + 1

                name =  '_' + model_name + "_" + 'mpc' + "_" + system_name
                model_path = f'{model_dir}/{str(MODEL_ID)}{name}.pth'
            else: 
                SIM_ID = -1
                MODEL_ID = -1
                model_path = f'{model_dir}/{model_name}.pth'

        
        print(f"simulate {system_name}")

        if SAVE:
            print(f"save model with model id {MODEL_ID}")
            print(f"save data with data id {SIM_ID}")
        
    except:
        print("No config file found. Data and model will not be saved.")
    print("\n----------------------------------------------------------------------------------\n")

    init_noise = [1e-8, 1e-8, 1e-12]
    target_noise = [1e-7, 1e-7, 1e-11]# [1e-8, 1e-8, 1e-12]#


    # Reference
    reference_strategie = {
        'target': True,
        'constraints' : 10,
        'past-values' : 0,
        'init_noise' : init_noise,
        'target_noise' : target_noise,
        'soft_constraints' : 'state_limit' # 'state_limit' or 'equilibrium'
    }

    # Equilibrium values for the system
    u_1 = 0.2   # control input to find equilibrium where we start
    u_2 = 0.3 # control input to find equilibrium where we want to end and linearize around

    # TIME
    t = 200

    control_time = Time_Def(
        0, 
        t, 
        step=1
    )#* dt_step

    sim_time = Time_Def(
        0, 
        t, 
        step=0.1
    )

    # GP settings
    optim_steps = 0
    pretrain_steps = 300

    # lengthscale = 5.5
    hyperparameters = {
        # 'lengthscale_2': torch.log(torch.exp(torch.tensor(lengthscale))-1), # 5.77
        # 'signal_variance_2': torch.log(torch.exp(torch.tensor(signal_variance))-1), #
    }


    system = load_system(system_name)
    num_tasks = system.dimension

    _ , x0 = system.get_ODEmatrix(u_1)
    x_0 = torch.tensor(x0)
    system_matrix , equilibrium = system.get_ODEmatrix(u_2)
    x_e = torch.tensor(equilibrium)

    # soft constraints for states
    #x_min = torch.tensor([system.x_min[0],system.x_min[1], x_e[2]])

    if reference_strategie['soft_constraints'] == 'state_limit':
        x_min = torch.tensor(system.x_min)
        x_max = torch.tensor(system.x_max)
    elif reference_strategie['soft_constraints'] == 'equilibrium':
        #constraint_factor = 1.1
        x_min = x_e * 0.9
        #x_min =torch.cat((x_e[0:system.state_dimension] * 0.9, x_e[system.state_dimension::] * 0.5),0)
        x_max = x_e * 1.1
        #x_max =torch.cat((x_e[0:system.state_dimension] * 1.1, x_e[system.state_dimension::] * 2),0)

    #x_min[2] = x_e[2]
    states = State_Description(x_e, x_0, min=x_min, max=x_max)

    with gpytorch.settings.observation_nan_policy('mask'):


        train_y, train_x, task_noise = create_setpoints(reference_strategie, control_time, states)
        likelihood = FixedTaskNoiseMultitaskLikelihood(num_tasks=num_tasks, noise=torch.tensor([1e-8,1e-8]), rank=num_tasks, has_task_noise=True, task_noise=task_noise)


        mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
        model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)

        if hyperparameters is not None:
            for key, value in hyperparameters.items():
                if hasattr(model.covar_module.model_parameters, key):
                    setattr(model.covar_module.model_parameters, key, torch.nn.Parameter(torch.tensor(value), requires_grad=False))
                else:
                    print(f'Hyperparameter {key} not found in model')

        optimize_mpc_gp(model, train_x, training_iterations=pretrain_steps)

        # model, mask = pretrain(system_matrix, num_tasks, control_time, pretrain_steps, reference_strategie, states, hyperparameters)# pretrain the system and generate gp model. eventually not necessary
        sim_data, ref_data, lode_data, _, _ = mpc_algorithm(system, model, states, reference_strategie,  control_time, sim_time, optim_steps)#, plot_single_steps=True


    # calc error
    def mse_mean(mean, ref, indx=None):
        if indx is None:
            return torch.mean(torch.square(mean - ref))
        else:
            return torch.mean(torch.square((mean - ref)[indx]))

    def constr_viol(mean, ub, lb, indx=None):
        if indx is None:
            return torch.mean(torch.relu(mean - ub) + torch.relu(lb - mean))
        else:
            return torch.mean(torch.relu((mean - ub)[indx]) + torch.relu((lb - mean)[indx]))
        

    # halfCount = int(t_end/dt_step)+1

    # fullCount = sim_data.time.shape[0]


    constraint_viol = constr_viol(
        torch.tensor(sim_data.y), 
        x_max.clone().detach().reshape(1, -1), 
        x_min.clone().detach().reshape(1, -1)
    )
    control_err = mse_mean(
        torch.tensor(sim_data.y[:,0:system.control_dimension]),
        torch.tile(states.target[0:system.control_dimension].clone().detach(), (sim_time.count, 1))
        #torch.zeros_like(torch.tensor(lode_data))
    )

    control_mean =  mean(sim_data.y[:,2])#0:control_time.count+1

    print(f"mean Control: {control_mean}")
    print(f"Control error: {control_err}")
    print(f"Constraint violation: {constraint_viol}")

    reference_data = {
        'time': sim_data.time,
        'f1': sim_data.y[:,0],
        'f2': sim_data.y[:,1],
        'f3': sim_data.y[:,2],
    }
    fig = create_mpc_plot(None, None, ['x1','x2', 'u'], 'Time ($\mathrm{s})$', 'Water Level ($\mathrm{m}$)', reference_data, x_e=[states.target[0],states.target[1],states.target[2]], close_constraint=False)
    plt.show()

    save_plot_to_pdf(fig, f'mpc_plot_endpoint_constraint')

    # plot_results(ref_data, lode_data, sim_data)
    # plt.show()

    if SAVE:
        # torch.save(model.state_dict(), model_path)
        with open(CONFIG_FILE,"w") as f:
            config['model_id'] = MODEL_ID
            config['simulation_id'] = SIM_ID
            json.dump(config, f)
        add_modelConfig(MODEL_ID, system_name,  x0, equilibrium, control_time.start, control_time.end, control_time.step)

        add_training_data(MODEL_ID, ref_data.time, ref_data.y)

        add_simulationConfig(SIM_ID, MODEL_ID, system_name, x0, equilibrium, sim_time.start, sim_time.end, sim_time.step, [control_err, constraint_viol])

        add_simulation_data(SIM_ID, lode_data.time, lode_data.y)

        add_reference_data(SIM_ID, 'nonlinear', sim_data.time, sim_data.y)

        print(f"save model with model id {MODEL_ID}")
        print(f"save data with data id {SIM_ID}")


run_sim()

# lengthscales = [
#     # 8, 
#     5, 
#     # 3.5
#     ]
# s_variances = [
#     0.01, 
#     0.05, 
#     # 0.1
#     ]

# for i in range(len(lengthscales)):
#     for j in range(len(s_variances)):
#         print(f"lengthscale: {lengthscales[i]}, signal_variance: {s_variances[j]}")
#         run_sim(lengthscales[i], s_variances[j])
#         print("\n----------------------------------------------------------------------------------\n")