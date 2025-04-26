
import gpytorch 
import torch
import numpy as np
from sklearn.metrics import root_mean_squared_error

from pylatex import Document, Tabular, Table
import matplotlib.pyplot as plt

from result_reporter.latex_exporter import plot_states, plot_weights, plot_trajectory, plot_error, save_plot_to_pdf, plot_single_states

# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import get_config, Time_Def, load_system, simulate_system, downsample_data, save_everything, plot_results,  Data_Def, State_Description, get_ode_from_spline
from nonlinear_LODE_GPs.weighting import Gaussian_Weight , Mahalanobis_Distance
from nonlinear_LODE_GPs.combined_posterior import CombinedPosterior_ELODEGP

def simulation_loop(system, Weight_fct, output_weights, datapoints):
    optim_steps_single = 300
    optim_steps =500
    learning_rate = 3

    equilibrium_controls = [
        0.1, # [2.0897e-02, 1.2742e-02, 1.0000e-05]
        0.2, # [8.3588e-02, 5.0968e-02, 2.0000e-05]
        0.3, # [1.8807e-01, 1.1468e-01, 3.0000e-05]
        0.4, # [3.3435e-01, 2.0387e-01, 4.0000e-05]
        0.5, # [5.2243e-01, 3.1855e-01, 5.0000e-05]
    ]

    u_ctrl = 1

    x0 = torch.tensor([0.0, 0.0])
    dt = 0.1
    t0 = 0.0
    t1 = 200.0
    downsample = int(t1 / dt / datapoints)
    # downsample =50
    sim_time = Time_Def(t0, t1, step=dt)
    train_time = Time_Def(t0, t1, step=sim_time.step*downsample)
    test_time = Time_Def(t0, t1, step=dt)

    num_tasks = system.dimension

    system_matrices = []
    equilibriums = []
    centers = []
    for i in range(len(equilibrium_controls)):
        system_matrix , x_e = system.get_ODEmatrix(equilibrium_controls[i])
        system_matrices.append(system_matrix)
        equilibriums.append(torch.tensor(x_e))
        centers.append(torch.tensor([x_e]))

    
    u = np.ones((sim_time.count,1)) * u_ctrl * system.param.u


    _train_x, _train_y= simulate_system(system, x0, sim_time, u)
    sim_data = Data_Def(_train_x, _train_y, system.state_dimension, system.control_dimension,train_time)
    noise = torch.tensor([1e-5, 1e-5, 1e-7])
    train_data = sim_data.downsample(downsample).add_noise(noise)

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
    model = CombinedPosterior_ELODEGP(
        train_data.time, 
        train_data.y, 
        likelihood, 
        num_tasks, 
        system_matrices, 
        equilibriums, 
        centers,
        Weight_fct, 
        shared_weightscale=False,
        additive_se=False,
        clustering=True,
        output_weights=output_weights,
        )#, 
    model._optimize(optim_steps_single)
    loss, parameters = model.optimize(optim_steps, verbose=False, learning_rate=learning_rate)

    test_x = test_time.linspace()
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        estimate, cov, weights = model.predict(test_x)
        output = gpytorch.distributions.MultitaskMultivariateNormal(estimate, cov)
        lower, upper = output.confidence_region()

        test_data = Data_Def(test_x.numpy(), estimate.detach().numpy(), system.state_dimension, system.control_dimension, test_time, uncertainty={
                    'variance': output.variance,
                    'lower': lower.detach().numpy(),
                    'upper': upper.detach().numpy(),
                    }, )
        
    weight_sum = torch.stack(weights).sum(dim=0)

    model_result = {
        'training_loss': loss[-1],
    }
    
    for i in range(len(equilibrium_controls)):
        model_result[f'l{i}'] = parameters[f'w_fcts.{i}.raw_scale']


    rmse_gp = root_mean_squared_error(sim_data.y.numpy(), test_data.y)
    _, ode_error_gp = get_ode_from_spline(system, test_data.y, test_data.time, verbose=False)

    error_gp = Data_Def(test_data.time, abs(test_data.y - _train_y.numpy()), system.state_dimension, system.control_dimension) 
    err_data.append(error_gp)

    err_result = {
        'rmse_gp': rmse_gp,
        'ode_error_gp': np.mean(ode_error_gp),
        'mean_weight': weight_sum.mean(),
        'min_weight': weight_sum.min(),
        'max_weight': weight_sum.max(),
    }
    # if STATE_PLOTS:
    #     if Weight_fct.__name__ == 'Gaussian_Weight':
    #         img_count = 0
    #     else:
    #         img_count = 1
    #     state_figure = plot_single_states(
    #         [test_data, train_data],
    #         [Weight_fct.__name__, 'training' ],
    #         header= ['$x_1$', '$x_2$', '$u_1$'], 
    #         yLabel=['$x_1$ - fill level ($m$)', '$x_2$ - fill level ($m$)', '$u_1$ - flow rate ($m^3/s$) '],
    #         line_styles = ['-',  '.'],
    #         colors = [img_count,2],               
    #     )
    #     save_plot_to_pdf(state_figure, f'state_plot_mixture-lodegp_{Weight_fct.__name__}')

    return model_result, err_result


if __name__ == "__main__":
    system_name = "nonlinear_watertank"
    system = load_system(system_name)

    N = 50
    SAVE_TABLES = False
    STATE_PLOTS = True
    table_file = '../data/tables/mixture-lodegp'

    random_repeats = 1
    seed_variants = range(random_repeats)

    Weight_fcts = [
        Gaussian_Weight,
        Mahalanobis_Distance,
    ]
    Weight_fct_names = [
        'Gaussian',
        'Mahalanobis',
    ]

    avg_model = {}
    avg_err = {}

    err_data = []

    for i in range(len(Weight_fcts)):
        model_results = {}
        err_results = {}

        Weight_fct = Weight_fcts[i]
        if Weight_fct == Gaussian_Weight:
            output_weights = False
        else:
            output_weights = True
        for seed in seed_variants:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_result, err_result = simulation_loop(system, Weight_fct, output_weights, N)
            print(f"Model result: {model_result}")
            print(f"Error result: {err_result}")

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

    if STATE_PLOTS:
        err_figure = plot_single_states(
            [err_data[0], err_data[1]],
            ['Gaussian weight', 'Mahalanobis distance'],
            header= ['$x_1$', '$x_2$', '$u_1$'], 
            yLabel=['$x_1$ - absolute error ($m$)', '$x_2$ - absolute error ($m$)', '$u_1$ - absolute error ($m^3/s$) '],
            line_styles = ['-', '-.']
        )

        plt.show()
        save_plot_to_pdf(err_figure, f'error_plot_mixture-lodegp')

    
    if SAVE_TABLES:
        geometry_options = {"margin": "2.54cm", "includeheadfoot": True}
        doc = Document(page_numbers=True, geometry_options=geometry_options)
        
        with doc.create(Table()) as tab:
            with doc.create(Tabular('c | c c c c c')) as table:
                table.add_hline()
                table.add_row(["Weight Function", "RMSE", "ODE Error", "Mean Weight", "Min Weight", "Max Weight"])
                table.add_hline()

                for i in range(len(Weight_fct_names)):
                    rmse = avg_err['rmse_gp'][i]
                    ode_error = avg_err['ode_error_gp'][i]
                    mean_weight = avg_err['mean_weight'][i]
                    min_weight = avg_err['min_weight'][i]
                    max_weight = avg_err['max_weight'][i]
                    
                    table.add_row([
                        f"{Weight_fct_names[i]}",
                        f"{rmse['mean']:.1e} \pm {rmse['std']:.1e}",
                        f"{ode_error['mean']:.1e} \pm {ode_error['std']:.1e}",
                        f"{mean_weight['mean']:.4f} \pm {mean_weight['std']:.4f}",
                        f"{min_weight['mean']:.4f} \pm {min_weight['std']:.4f}",
                        f"{max_weight['mean']:.4f} \pm {max_weight['std']:.4f}",
                        ])
                table.add_hline()

        with doc.create(Table()) as tab2:
            with doc.create(Tabular('c | c c c c c c')) as table:
                table.add_hline()
                table.add_row(["Weight Function", 'loss', "e-LODEGP 1", "e-LODEGP 2", "e-LODEGP 3", "e-LODEGP 4", "e-LODEGP 5"])
                table.add_hline()

                
                for i in range(len(Weight_fct_names)):
                    row = [f"{Weight_fct_names[i]}"]
                    for key in avg_model.keys():
                        value = avg_model[key][i]
                        row.append(f"{value['mean']:.2f} \pm {value['std']:.2f}")
                    
                    table.add_row(row)
                table.add_hline()
                
        doc.generate_pdf(table_file, clean_tex=False)

