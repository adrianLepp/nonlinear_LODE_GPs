import gpytorch 
# from nonlinear_LODE_GPs.kernels import *
import torch
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import itertools
from pylatex import Document, Tabular, Table
# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.lodegp import  optimize_gp, LODEGP
from nonlinear_LODE_GPs.helpers import *
from nonlinear_LODE_GPs.mean_modules import Equilibrium_Mean

from result_reporter.latex_exporter import plot_loss, plot_error, plot_states, save_plot_to_pdf, plot_single_states

torch.set_default_dtype(torch.float64)

device = 'cpu'
system_name = "nonlinear_watertank"

SAVE_TABLES = False
LOSS_PLOTS = True
STATE_PLOTS = False
loss_file = '../data/losses/e-lodegp'
table_file = '../data/tables/e-lodegp'

def simulation_loop(downsample:int, u_e:float, u:float, additive_se:bool):

    system_matrix , equilibrium = system.get_ODEmatrix(u_e)
    x_0 = np.array(equilibrium)
    u_trajectory = np.ones((sim_time.count,1)) * u * system.param.u


    _train_x, _train_y= simulate_system(system, x_0[0:system.state_dimension], sim_time, u_trajectory)
    sim_data = Data_Def(_train_x, _train_y,system.state_dimension, system.control_dimension, sim_time)
    train_data = sim_data.downsample(downsample).add_noise(noise)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks, 
        has_global_noise=False, 
        noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
    )
    # likelihood.initialize(task_noises=noise)
    # likelihood.raw_task_noises.requires_grad = False

    mean_module = Equilibrium_Mean(equilibrium, num_tasks)
    model = LODEGP(train_data.time, train_data.y, likelihood, num_tasks, system_matrix, mean_module, additive_se=additive_se) #system.state_var, system.control_var

    training_loss, hyperparam = optimize_gp(model,optim_steps, verbose=False)

    if additive_se is True:
        label = 'LODE-GP + SE'
    else:
        label = 'LODE-GP'
    loss_tracker.add_loss(label, training_loss)

    # %% test
    test_x = sim_time.linspace()
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        output = model(test_x)
        lower, upper = output.confidence_region()

    uncertainty = {
        'lower': lower.numpy(),
        'upper': upper.numpy()
    }

    gp_data = Data_Def(test_x.numpy(), output.mean.numpy(), system.state_dimension, system.control_dimension, sim_time, uncertainty)

    x0_e = x_0 - np.array(equilibrium)
    ref_x, ref_y= simulate_system(system, x0_e[0:system.state_dimension], sim_time, u_trajectory-equilibrium[-1], linear=True)
    ref_data = Data_Def(ref_x.numpy(), ref_y.numpy() + np.array(equilibrium), system.state_dimension, system.control_dimension, sim_time)

    data.append(gp_data)
    data.append(train_data)
    data.append(ref_data)

    _, ode_error_de = get_ode_from_spline(system, ref_data.y, ref_data.time, verbose=False)
    _, ode_error_gp = get_ode_from_spline(system, gp_data.y, gp_data.time, verbose=False)

    rmse_gp = root_mean_squared_error(_train_y.numpy(), gp_data.y)
    rmse_de = root_mean_squared_error(_train_y.numpy(), ref_data.y)

    error_gp = Data_Def(gp_data.time, abs(gp_data.y - _train_y.numpy()), system.state_dimension, system.control_dimension) 
    error_de = Data_Def(ref_data.time, abs(ref_data.y - _train_y.numpy()), system.state_dimension, system.control_dimension)

    err_data.append(error_gp)
    err_data.append(error_de)

    # print(f"RMSE GP: {rmse_gp:.2e}")
    # fig_results = plot_states([gp_data, ref_data, train_data])
    # plt.show()

    if additive_se is True:
        model_result = {
            'training_loss': training_loss[-1],
            'lengthscale': hyperparam['covar_module.kernels.1.base_kernel.model_parameters.lengthscale_2'],
            'signal_variance': hyperparam['covar_module.kernels.1.base_kernel.model_parameters.signal_variance_2'],
            'noise': hyperparam['likelihood.raw_task_noises'],
            'gp_outputscale' : hyperparam['covar_module.kernels.1.raw_outputscale'],
            'se_signal_variance': hyperparam['covar_module.kernels.0.base_kernel.task_covar_module.raw_var'],
            'se_lengthscale': hyperparam['covar_module.kernels.0.base_kernel.data_covar_module.raw_lengthscale'],
            'se_outputscale': hyperparam['covar_module.kernels.0.raw_outputscale'],
        }
    else:
        model_result = {
            'training_loss': training_loss[-1],
            'lengthscale': hyperparam['covar_module.model_parameters.lengthscale_2'],
            'signal_variance': hyperparam['covar_module.model_parameters.signal_variance_2'],
            'noise': hyperparam['likelihood.raw_task_noises'],
        }

        
    err_result = {
        'rmse_de': rmse_de,
        'rmse_gp': rmse_gp,
        'ode_error_de': np.mean(ode_error_de),
        'ode_error_gp': np.mean(ode_error_gp)
    }
    return model_result, err_result

    # return training_loss, hyperparam, rmse_de, rmse_gp, np.mean(ode_error_de), np.mean(ode_error_gp)

if __name__ == "__main__":
    # Fixed parameters
    simulation_time = 100
    simulation_step = 0.1
    optim_steps = 300

    sim_time = Time_Def(0, simulation_time, step=simulation_step)

    # variable parameters / variations
    data_size_variants = [10, 50, 100] #[10, 25, 50, 75, 100]#10, 20, 50, 100 [10] # 

    random_repeats = 1
    u_e_variants =  [0.2]# [0.1, 0.2, 0.3, 0.4] #[0.2] # 
    u_variants = [0.3] #[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    seed_variants = range(random_repeats)

    system = load_system(system_name)
    num_tasks = system.dimension
    noise = torch.tensor([1e-5, 1e-5, 1e-7])

    avg_model = {}
    avg_err = {}

    se_variants = [False, True]

    for data_size in data_size_variants:
        model_results = {}
        err_results = {}

        loss_tracker = LossTracker(f'{loss_file}_{data_size}.csv')
        data = []
        err_data = []

        
        downsample = int(simulation_time / simulation_step / data_size)
        for u_e in u_e_variants:
            for additive_se in se_variants:
                u = u_e + 0.1
                if u <= u_e:
                        continue
                for seed in seed_variants:
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    # try:
                    model_result, err_result = simulation_loop(downsample, u_e, u, additive_se)
                    # except Exception as e:
                        # print(f"Error during simulation with data size {data_size}, u_e {u_e}, u {u}, seed {seed}: {e}")
                        # continue

                    for key in model_result.keys():
                        if key not in model_results:
                            model_results[key] = []
                        model_results[key].append(model_result[key])

                    for key in err_result.keys():
                        if key not in err_results:
                            err_results[key] = []
                        err_results[key].append(err_result[key])


        for key in model_results.keys():
            if key not in avg_model:
                avg_model[key] = []

            if key == 'noise' or key == 'se_signal_variance':
                mean = np.array(model_results[key]).mean(axis=0)
                std = np.array(model_results[key]).std(axis=0)
            else: 
                mean = np.mean(model_results[key])
                std = np.std(model_results[key])
            
            avg_model[key].append({
                'mean': mean,
                'std': std
            })
        
        for key in err_results.keys():        
            if key not in avg_err:
                avg_err[key] = []
            avg_err[key].append({
                'mean': np.mean(err_results[key]),
                'std': np.std(err_results[key])
            })
        
        if LOSS_PLOTS:
            loss_tracker.to_csv()
            fig_loss = loss_tracker.plot_losses()
            plt.show()
            save_plot_to_pdf(fig_loss, f'loss_plot_e-lodegp_{data_size}')

        if STATE_PLOTS:
            state_figure = plot_single_states(
                [data[0], data[3], data[2], data[1]],
                ['LODE-GP', 'LODE-GP+SE', 'lin. ODE', 'training' ],
                header= ['$x_1$', '$x_2$', '$u_1$'], 
                yLabel=['$x_1$ - fill level ($m$)', '$x_2$ - fill level ($m$)', '$u_1$ - flow rate ($m^3/s$) '],
                line_styles = ['-', '-.', '--', '.']               
            )
            
            err_figure = plot_single_states(
                [err_data[0], err_data[2], err_data[1]],
                ['LODE-GP', 'LODE-GP+SE', 'lin. ODE', 'training' ],
                header= ['$x_1$', '$x_2$', '$u_1$'], 
                yLabel=['$x_1$ - absolute error ($m$)', '$x_2$ - absolute error ($m$)', '$u_1$ - absolute error ($m^3/s$) '],
                line_styles = ['-', '-.', '--']
            )

            plt.show()
            save_plot_to_pdf(state_figure, f'state_plot_e-lodegp_{data_size}')
            save_plot_to_pdf(err_figure, f'error_plot_e-lodegp_{data_size}')


    if SAVE_TABLES:
        geometry_options = {"margin": "2.54cm", "includeheadfoot": True}
        doc = Document(page_numbers=True, geometry_options=geometry_options)
        
        with doc.create(Table()) as tab2:
            with doc.create(Tabular('c r | l l')) as table:
                # table.add_caption("Error")
                table.add_hline()
                table.add_row(["Model", "D", "RMSE", "ODE Error"])
                table.add_hline()

                for i in range(len(data_size_variants)):
                    rmse = avg_err['rmse_gp'][i]
                    ode_error = avg_err['ode_error_gp'][i]
                    
                    table.add_row([
                        '',
                        f"{data_size_variants[i]}",
                        f"{rmse['mean']:.1e} \pm {rmse['std']:.1e}",
                        f"{ode_error['mean']:.1e} \pm {ode_error['std']:.1e}"
                        ])
                table.add_hline()

                rmse = avg_err['rmse_de'][0]
                ode_error = avg_err['ode_error_de'][0]
                table.add_row([
                    'linear ODE',
                    '',
                    f"{rmse['mean']:.1e} \pm {rmse['std']:.1e}",
                    f"{ode_error['mean']:.1e} \pm {ode_error['std']:.1e}"
                    ])

        with doc.create(Table()) as tab3:
            with doc.create(Tabular('c r | l l l l')) as table:
                # table.add_caption("Error")
                table.add_hline()
                table.add_row(["Model", "D", "Loss", "signal variance", "lengthscale", 'noise'])
                table.add_hline()

                for i in range(len(data_size_variants)):
                    loss = avg_model['training_loss'][i]
                    lengthscale = avg_model['lengthscale'][i]
                    signal_variance = avg_model['signal_variance'][i]
                    noise = avg_model['noise'][i]
                    
                    table.add_row([
                        '',
                        f"{data_size_variants[i]}",
                        f"{loss['mean']:.2f} \pm {loss['std']:.2f}",
                        f"{lengthscale['mean']:.2f} \pm {lengthscale['std']:.2f}",
                        f"{signal_variance['mean']:.4f} \pm {signal_variance['std']:.4f}",
                        ", ".join([f"{val:.1e}" for val in noise['mean']]) + " \pm " + ", ".join([f"{val:.1e}" for val in noise['std']])
                        # f"{noise['mean']:.1e} \pm {noise['std']:.1e}"
                        ])
                table.add_hline()

        if additive_se is True:
            with doc.create(Table()) as tab:
                with doc.create(Tabular('c r | l l l l')) as table:
                    table.add_hline()
                    table.add_row(["Model", "D", "gp out", "se out", 'se lengthscale', "se signal variance"])
                    table.add_hline()

                    for i in range(len(data_size_variants)):
                        gp_outputscale = avg_model['gp_outputscale'][i]
                        se_outputscale = avg_model['se_outputscale'][i]
                        se_lengthscale = avg_model['se_lengthscale'][i]
                        se_signal_variance = avg_model['se_signal_variance'][i]
                        
                        
                        table.add_row([
                            '',
                            f"{data_size_variants[i]}",
                            f"{gp_outputscale['mean']:.1e} \pm {gp_outputscale['std']:.1e}",
                            f"{se_outputscale['mean']:.1e} \pm {se_outputscale['std']:.1e}",
                            f"{se_lengthscale['mean']:.2f} \pm {se_lengthscale['std']:.2f}",
                            # f"{se_signal_variance['mean']:.4f} \pm {se_signal_variance['std']:.4f}",
                            ", ".join([f"{val:.1e}" for val in se_signal_variance['mean']]) + " \pm " + ", ".join([f"{val:.1e}" for val in se_signal_variance['std']])
                            ])
                    table.add_hline()

        doc.generate_pdf(table_file, clean_tex=False)
            
