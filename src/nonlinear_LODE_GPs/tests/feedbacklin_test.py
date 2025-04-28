import torch
from result_reporter.latex_exporter import plot_states, surface_plot, plot_trajectory, save_plot_to_pdf, plot_single_states
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from pylatex import Document, Tabular, Table

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import get_config, load_system, Data_Def, Time_Def, save_everything
from nonlinear_LODE_GPs.feedback_linearization import Simulation_Config, learn_system_nonlinearities, Controller
from nonlinear_LODE_GPs.gp import Linearizing_Control_2, Linearizing_Control_4, Linearizing_Control_5, CompositeModel
from scipy.integrate import solve_ivp

torch.set_default_dtype(torch.float64)
device = 'cpu'

SAVE = False
PLOT = False

system_name = "inverted_pendulum"

def create_controller(Control_Class, system, controller_0, t:float, a0:float, a1:float, v:float, b_prior:float):
    optim_steps = 100
    downsample = 20
    sim_time = Time_Def(0, t, step=0.01)

    
    noise = torch.tensor([1e-3, 1e-3, 0], dtype=torch.float64)

    sim_configs = [
        Simulation_Config(sim_time, [np.pi/2  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
        Simulation_Config(sim_time, [-np.pi/2 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),

        Simulation_Config(sim_time, [np.pi/4  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
        Simulation_Config(sim_time, [-np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),

        Simulation_Config(sim_time, [3* np.pi/4  , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
        Simulation_Config(sim_time, [-3 * np.pi/4 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
        
        Simulation_Config(sim_time, [np.pi , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
        Simulation_Config(sim_time, [0 , 0 ,0], np.zeros((sim_time.count,1)), downsample, 'u=0'),
    ]


    control_gp_kwargs = {
        #Linearizing_Control_2
        'consecutive_training':False,
        #Linearizing_Control_4
        'b' : b_prior,
        'a' : torch.tensor([[a0],[a1]], dtype=torch.float64),
        'v' : torch.tensor([v], dtype=torch.float64),
        'controller':controller_0, # controller_0 # None  
        'noise': noise,
    }

    model_config = {
    'device': '',
    'model_path': '',
    'load': False,
    'save': False,
}

    alpha, beta, control_gp = learn_system_nonlinearities(
        system, 
        sim_configs, 
        optim_steps, 
        ControlGP_Class = Control_Class,
        controlGP_kwargs = control_gp_kwargs,
        plot=False, 
        model_config=model_config,
        )
    return alpha, beta, control_gp
    

def test_controller(controller:Controller, system, time_def:Time_Def, position):

    x_0 = np.array([position , 0 ,0])

    y_ref_control = np.zeros((time_def.count))
    ts = time_def.linspace()

    with gpytorch.settings.observation_nan_policy('mask'):
        with torch.no_grad():
            u_control = np.zeros_like(ts)
            sol = solve_ivp(
                system.stateTransition_2, 
                [time_def.start, time_def.end], 
                x_0[0:system.state_dimension], 
                method='RK45', 
                t_eval=ts, args=(time_def.step, controller, u_control, y_ref_control),
                max_step=0.01
            )
            x = sol.y.transpose()

            solution = []
            for i in range (x.shape[1]):
                solution.append(x[:,i])

            solution.append(u_control)
            control_y = np.stack(solution, -1)

    control_data = Data_Def(ts.numpy(), control_y, system.state_dimension, system.control_dimension, time_def)
    return control_data

def analyze_data(control_data):
    x0 = control_data.y[0,:]
    xEnd = control_data.y[-1,:]

    # if xEnd[1] > x0[1]:
    #     xPeak = np.max(control_data.y[:,1])
    #     peak_idx = np.argmax(control_data.y[:,1])
    # else:
    #     xPeak = np.min(control_data.y[:,1])
    #     peak_idx = np.argmin(control_data.y[:,1])
    peak = np.max(np.abs(control_data.y[:,1]))
    peak_idx = np.argmax(np.abs(control_data.y[:,1]))

    peakTime = control_data.time[peak_idx]

    energy = np.sum(control_data.y[:,1]**2) * (control_data.time[1] - control_data.time[0])
   
    steady_state_error = np.abs(xEnd)[0]

    settling_idx = np.argmax(np.isclose(control_data.y[:, 0], xEnd[0], atol=1e-3))
    # settling_idx = np.argmin(np.linalg.norm(control_data.y - xEnd, axis=1))
    settling_time = control_data.time[settling_idx]

    return settling_time , steady_state_error, peak, peakTime, energy


def control_metric_table(control_metric, table_name):
    metric_names = ['steady_state_error','peak','peakTime', 'settling_time', 'energy', 'rmse_alpha', 'rmse_beta']
    geometry_options = {"margin": "2.54cm", "includeheadfoot": True}
    doc = Document(page_numbers=True, geometry_options=geometry_options)
    
    with doc.create(Table()) as tab:
        with doc.create(Tabular('c | c c c c c c c')) as table:
            table.add_hline()
            table.add_row(['controller'] + metric_names)
            table.add_hline()

            for key in control_metric.keys():
                row =  [key]
                for metric in metric_names:
                    row.append(f"{control_metric[key][metric]['mean']:.4f} \pm {control_metric[key][metric]['std']:.4f}")
                table.add_row(row)
            table.add_hline()

    doc.generate_pdf(table_name, clean_tex=False)


def test_standard_controller():
    t_train  = 5
    t_test = 10

    test_time = Time_Def(0, t_test, step=0.01)

    system = load_system(system_name, a0=0, a1=0, v=1)

    controll_param_variations = [
        # {
        # 'a0': 10,
        # 'a1': 15,
        # 'v': 0,
        # },
        # {
        # 'a0': 20,
        # 'a1': 25,
        # 'v': 0,
        # },
        {
        'a0': 3,
        'a1': 4,
        'v': 0,
        },
    
    ]


    random_repeats = 20
    seed_variants = range(random_repeats)



    

    for seed in seed_variants:
        for controll_param in controll_param_variations:
            a0 = controll_param['a0']
            a1 = controll_param['a1']
            v = controll_param['v']
            exact_controller = Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]), alpha=system.alpha, beta=system.beta)
            controller_0 = Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]))
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            position = (np.pi - 2* rng.random() * np.pi)
            exact_data = test_controller(exact_controller, system, test_time, position)
            prior_data = test_controller(controller_0, system, test_time, position)

            if PLOT:
                settling_time , por, offset, peakTime = analyze_data(prior_data)
                print(f"Settling time: {settling_time:.2f}, Peak time: {peakTime:.2f}, Overshoot ratio: {por:.2f}, Offset: {offset:.2f}")
                figure = plot_single_states(
                    [exact_data, prior_data],
                    ["exact feedback", r'$u_0$'],
                    header= [r'$x_1$', r'$x_2$', r'$u$'], 
                    yLabel=['angle (rad)', 'angular velocity (rad/s) ', 'force (N) '],
                )


                trajectory_plot = plot_trajectory([exact_data, prior_data], {}, ax_labels=['angle (rad)', 'angular velocity (rad/s)'], labels = [ "exact feedback", r'$u_0$'])

                plt.show()
                

def get_gp_param(model):
    param_conversion = torch.nn.Softplus()
    named_parameters = list(model.named_parameters())
    parameters = {}
    for j in range(len(named_parameters)):
        parameters[named_parameters[j][0]] = param_conversion(named_parameters[j][1].data).tolist()
    
    return {
        'alpha': {
            'lengthscale': parameters['_alpha.covar_module.raw_outputscale'],
            'signal_variance': parameters['_alpha.covar_module.base_kernel.raw_lengthscale'],
            'mean' : parameters['_alpha.variational_strategy._variational_distribution.variational_mean'],
            'covar' : parameters['_alpha.variational_strategy._variational_distribution.chol_variational_covar']
        },
        'beta': {
            'lengthscale': parameters['_log_beta.covar_module.raw_outputscale'],
            'signal_variance': parameters['_log_beta.covar_module.base_kernel.raw_lengthscale'],
            'mean' : parameters['_log_beta.variational_strategy._variational_distribution.variational_mean'],
            'covar' : parameters['_log_beta.variational_strategy._variational_distribution.chol_variational_covar']
        },
        'noise': {
            'noise' : parameters['likelihood.noise_covar.raw_noise']
        }
    }

    return {
        'alpha': {
            'lengthscale': parameters['covar_module.cov_alpha.base_kernel.raw_lengthscale'],
            'signal_variance' : ['covar_module.cov_alpha.raw_outputscale'],
            'mean': parameters['mean_module.base_means.1.raw_constant'],
        },
        'noise': {
            'task_noise': parameters['likelihood.raw_task_noises'],
            'noise': parameters['likelihood.raw_noise'],
        },
        'beta': {
            'lengthscale': parameters['covar_module.cov_beta.base_kernel.raw_lengthscale'],
            'signal_variance' : ['covar_module.cov_beta.raw_outputscale'],
            'mean': parameters['mean_module.base_means.1.raw_constant'],
        }
    }

def evaluate_nonlinearities(gp_model, system, alpha, beta):
    l = 100
    val = 3* torch.pi / 4
    x_min = [-val, -val]
    x_max = [torch.pi, val ]

    test_points1, test_points2 = torch.meshgrid(
                torch.linspace(x_min[0], x_max[0], l),
                torch.linspace(x_min[1], x_max[1], l)
            )
    test_points = torch.stack([test_points1.flatten(), test_points2.flatten()], dim=-1)

    beta_system = torch.zeros(gp_model.train_targets.shape[0])
    alpha_system = torch.zeros(gp_model.train_targets.shape[0])
    for i in range(gp_model.train_targets.shape[0]):
        alpha_system[i] = system.alpha(gp_model.train_inputs[0][i].numpy())
        beta_system[i] = system.beta(gp_model.train_inputs[0][i].numpy())

    with gpytorch.settings.observation_nan_policy('mask'):
        with torch.no_grad():
            test_alpha = alpha(test_points).squeeze()
            test_beta = beta(test_points, 0).squeeze()

    alpha_system = torch.zeros_like(test_alpha)
    beta_system = torch.zeros_like(test_beta)
    for i in range(test_points.shape[0]):
        alpha_system[i] = system.alpha(test_points[i].numpy())
        beta_system[i] = system.beta(test_points[i].numpy())

    rmse_alpha = torch.sqrt(torch.mean((alpha_system - test_alpha) ** 2))
    rmse_beta = torch.sqrt(torch.mean((beta_system - test_beta) ** 2))
    return rmse_alpha, rmse_beta

    

def main():
    t_train  = 5
    t_test = 10

    test_time = Time_Def(0, t_test, step=0.01)

    system = load_system(system_name, a0=0, a1=0, v=1)

    a0 = 1
    a1 = 2
    v = 0

    Controllers = [CompositeModel, Linearizing_Control_5, ]
    random_repeats = 20
    seed_variants = range(random_repeats)

    exact_controller = Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]), alpha=system.alpha, beta=system.beta)
    controller_0 = Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]))

    control_metric = {
    }
    for Control_Class in Controllers:
        control_metrics = {
                'settling_time': [],
                'steady_state_error': [],
                'peak': [],
                'peakTime': [],
                'energy': [],
                'rmse_alpha': [],
                'rmse_beta': [],
        }

        for seed in seed_variants:
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            torch.manual_seed(seed)
            
            alpha, beta, control_gp = create_controller(Control_Class, system, controller_0,  t_train, a0, a1, v, b_prior=10)
            controller = Controller(system.state_dimension, system.control_dimension, a=np.array([a0, a1]), v=np.array([v]), alpha=alpha, beta=beta)
            
            position = (np.pi - 2* rng.random() * np.pi)
            control_data = test_controller(controller, system, test_time, position)
            exact_data = test_controller(exact_controller, system, test_time, position)
            prior_data = test_controller(controller_0, system, test_time, position)

            settling_time , steady_state_error, peak, peakTime, energy = analyze_data(control_data)

            rmse_alpha, rmse_beta = evaluate_nonlinearities(control_gp, system, alpha, beta)

            control_metrics['settling_time'].append(settling_time)
            control_metrics['steady_state_error'].append(steady_state_error)
            control_metrics['peak'].append(peak)
            control_metrics['peakTime'].append(peakTime)
            control_metrics['energy'].append(energy)
            control_metrics['rmse_alpha'].append(rmse_alpha)
            control_metrics['rmse_beta'].append(rmse_beta)

            

            # print(f"Settling time: {settling_time:.2f}, Peak time: {peakTime:.2f}, Overshoot ratio: {por:.2f}, Offset: {offset:.2f}")

            if PLOT:
                figure = plot_single_states(
                    [control_data, exact_data, prior_data],
                    ['GP', "exact feedback", r'$u_0$'],
                    header= [r'$x_1$', r'$x_2$', r'$u$'], 
                    yLabel=['angle (rad)', 'angular velocity (rad/s) ', 'force (N) '],
                )

                trajectory_plot = plot_trajectory([control_data, exact_data, prior_data], {}, ax_labels=['angle (rad)', 'angular velocity (rad/s)'], labels = ['GP', "exact feedback", r'$u_0$'])

                plt.show()
                
        avg_metrics = {key: {'mean': np.mean(value), 'std': np.std(value)} for key, value in control_metrics.items()}

        control_metric[Control_Class.__name__] = avg_metrics
        print(f"Control Class: {Control_Class.__name__}")

    control_metric_table(control_metric, '../data/tables/control_metrics')

            

    
main()