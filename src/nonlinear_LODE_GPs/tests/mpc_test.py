import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from pylatex import Document, Tabular, Table

from result_reporter.latex_exporter import create_mpc_plot, save_plot_to_pdf


# ----------------------------------------------------------------------------
from nonlinear_LODE_GPs.helpers import State_Description, load_system, Time_Def
from nonlinear_LODE_GPs.likelihoods import FixedTaskNoiseMultitaskLikelihood
# from nonlinear_LODE_GPs.masking import *
from nonlinear_LODE_GPs.mpc import mpc_algorithm, pretrain, optimize_mpc_gp, create_setpoints
from nonlinear_LODE_GPs.lodegp import LODEGP, optimize_gp
from nonlinear_LODE_GPs.mean_modules import Equilibrium_Mean


PLOT = False


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

def mpc_loop(system, u_1, u_2, reference_strategie, control_time, sim_time, optim_steps):
    num_tasks = system.dimension
    _ , x0 = system.get_ODEmatrix(u_1)
    x_0 = torch.tensor(x0)
    system_matrix , equilibrium = system.get_ODEmatrix(u_2)
    x_e = torch.tensor(equilibrium)

    if reference_strategie['soft_constraints'] == 'state_limit':
        x_min = torch.tensor(system.x_min)
        x_max = torch.tensor(system.x_max)
    elif reference_strategie['soft_constraints'] == 'equilibrium':
        x_min = x_e * 0.9
        x_max = x_e * 1.1

    states = State_Description(x_e, x_0, min=x_min, max=x_max)

    with gpytorch.settings.observation_nan_policy('mask'):
        train_y, train_x, task_noise = create_setpoints(reference_strategie, control_time, states)
        likelihood = FixedTaskNoiseMultitaskLikelihood(num_tasks=num_tasks, noise=torch.tensor([1e-8,1e-8]), rank=num_tasks, has_task_noise=True, task_noise=task_noise)


        mean_module = Equilibrium_Mean(states.equilibrium, num_tasks)
        model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)


        training_loss, hyperparam = optimize_gp(model, training_iterations=optim_steps, verbose=False)

        # model, mask = pretrain(system_matrix, num_tasks, control_time, pretrain_steps, reference_strategie, states, hyperparameters)# pretrain the system and generate gp model. eventually not necessary
        sim_data, ref_data, lode_data, settling_time, calc_time = mpc_algorithm(system, model, states, reference_strategie,  control_time, sim_time, 0)#, plot_single_steps=True

        constraint_viol = constr_viol(
            torch.tensor(sim_data.y), 
            x_max.clone().detach().reshape(1, -1), 
            x_min.clone().detach().reshape(1, -1)
        )
        control_err = mse_mean(
            torch.tensor(sim_data.y[:,0:2]),
            torch.tile(states.target[0:2].clone().detach(), (sim_time.count, 1))
            #torch.zeros_like(torch.tensor(lode_data))
        )

        control_mean =  np.mean(sim_data.y[:,2])

        reference_data = {
            'time': sim_data.time,
            'f1': sim_data.y[:,0],
            'f2': sim_data.y[:,1],
            'f3': sim_data.y[:,2],

        }
        peak_value = np.max(sim_data.y)
        peak_time = sim_data.time[np.argmax(sim_data.y[:,0])]
        if peak_value <= states.target[0]:
            peak_value = None
            peak_time = None
        
        if peak_value is not None:
            peak = peak_value - states.target[0]
            target  = states.target[0] - states.init[0]
            p_o_ratio = peak / target
        else:
            p_o_ratio = None

        rise_time = None
        for t, value in zip(sim_data.time, sim_data.y[:, 0]):
            if np.isclose(value, states.target[0], atol=1e-3):
                rise_time = t
                break



        if PLOT:
            fig = create_mpc_plot(None, None, ['x1','x2', 'u'], 'Time ($\mathrm{s})$', 'Water Level ($\mathrm{m}$)', reference_data, x_e=[states.target[0],states.target[1],states.target[2]])
            plt.show()
            save_plot_to_pdf(fig, f'mpc_plot_{reference_strategie['name']}')

        return {
            'constraint_viol': constraint_viol.item(),
            'control_err': control_err.item(),
            'control_mean': control_mean,
            'settling_time': settling_time,
            'rise_time': rise_time,
            'peak_value': peak_value,
            'peak_time': peak_time,
            "p_o_ratio": p_o_ratio,
            'calc_time': calc_time,
        }, {
            'lengthscale': hyperparam['covar_module.model_parameters.lengthscale_2'],
            'signal_variance': hyperparam['covar_module.model_parameters.signal_variance_2'],
            'loss': training_loss[-1],
        }


def main():
    table_file = '../data/tables/mpc_test_table_200'
    random_repeats = 10
    u_e_variants =  [0.2] #[0.1, 0.2, 0.3, 0.4] # [0.1, 0.2, 0.3, 0.4] #[0.2] # 
    seed_variants = range(random_repeats)

    init_noise = [1e-8, 1e-8, 1e-12]
    target_noise = [1e-6, 1e-6, 1e-12]# [1e-8, 1e-8, 1e-12]#

    optim_steps = 300

    system = load_system(system_name)

    # Reference
    reference_strategie_variants = [ 
        {
        'target': False,
        'constraints' : 10,
        'past-values' : 0,
        'init_noise' : init_noise,
        'target_noise' : target_noise,
        'soft_constraints' : 'state_limit',
        'time' : 200,
        'name' : 'prior'
    },
    # {
    #     'target': False,
    #     'constraints' : 10,
    #     'past-values' : 0,
    #     'init_noise' : init_noise,
    #     'target_noise' : target_noise,
    #     'soft_constraints' : 'equilibrium',
    #     'time' : 200,
    #     'name': 'close_constraint'
    # },
    # {
    #     'target': True,
    #     'constraints' : 10,
    #     'past-values' : 0,
    #     'init_noise' : init_noise,
    #     'target_noise' : target_noise,
    #     'soft_constraints' : 'state_limit',
    #     'time' : 200,
    #     'name': 'target'
    # }, 
    ]

    error_stats = {}
    param_stats = {}
    for reference_strategie in reference_strategie_variants:
        if reference_strategie['target'] == True:
            control_time = Time_Def(0, 200, step=1)#* dt_step
        else:
            control_time = Time_Def(0, reference_strategie['time'], step=1)
        sim_time = Time_Def(0, reference_strategie['time'], step=0.1)

        error_lists = {}
        param_lists = {}
        for u_e in u_e_variants:
            u_1 = u_e
            u_2 = u_1 + 0.1
            for seed in seed_variants:
                np.random.seed(seed)
                torch.manual_seed(seed)
                error, param = mpc_loop(system, u_1, u_2, reference_strategie, control_time, sim_time, optim_steps)

                for key in error.keys():
                    if key not in error_lists:
                        error_lists[key] = []
                    error_lists[key].append(error[key])

                for key in param.keys():
                    if key not in param_lists:
                        param_lists[key] = []
                    param_lists[key].append(param[key])
    

        for key in error_lists.keys():        
            if key not in error_stats:
                error_stats[key] = []
            
            try:
                error_stats[key].append({
                    'mean': np.mean(error_lists[key]),
                    'std': np.std(error_lists[key])
                })
            except:
                print(f"Error in {key}: {error_lists[key]}")
                error_stats[key].append({
                    'mean': np.nan,
                    'std': np.nan
                })

        for key in param_lists.keys():        
            if key not in param_stats:
                param_stats[key] = []
            param_stats[key].append({
                'mean': np.mean(param_lists[key]),
                'std': np.std(param_lists[key])
            })

    model_number = len(reference_strategie_variants)

    geometry_options = {"margin": "2.54cm", "includeheadfoot": True}
    doc = Document(page_numbers=True, geometry_options=geometry_options)
    
    with doc.create(Table()) as tab:
        with doc.create(Tabular('l | r r r r')) as table:
            # table.add_caption("Error")
            table.add_hline()
            table.add_row(["Metric", "Model (a)", "Model (b)", "Model (c)", "Model (d)"])
            table.add_hline()

            for key in error_stats.keys():
                row = [key, '']
                for i in range(model_number):
                    row.append(f"{error_stats[key][i]['mean']:.1e} \pm {error_stats[key][i]['std']:.1e}")
                table.add_row(row)
                # table.add_row([
                #     key,
                #     # f"{error_stats[key][0]['mean']:.1e} \pm {error_stats[key][0]['std']:.1e}",
                #     '',
                #     ,
                #     f"{error_stats[key][1]['mean']:.1e} \pm {error_stats[key][1]['std']:.1e}",
                #     f"{error_stats[key][2]['mean']:.1e} \pm {error_stats[key][2]['std']:.1e}"
                # ])

            table.add_hline()

    model_number = len(reference_strategie_variants)
    with doc.create(Table()) as tab2:
        with doc.create(Tabular('l | r r r ')) as table:
            # table.add_caption("Error")
            table.add_hline()
            table.add_row(["", "Model (b)", "Model (c)", "Model (d)"])
            table.add_hline()

            for key in param_stats.keys():

                row = [key]
                for i in range(model_number):
                    row.append(f"{param_stats[key][i]['mean']:.4f} \pm {param_stats[key][i]['std']:.4f}")
                table.add_row(row)

            table.add_hline()

    doc.generate_pdf(table_file, clean_tex=False)


if __name__ == "__main__":
    system_name = "nonlinear_watertank"
    main()


# TODO save plots
# TODO save error to table

# create_mpc_plot(training_data, simulation_data, system_description['states'], reference_data=reference_data, x_e=model_config['system_param'], close_constraint=close_constraint))
# save_plot_to_pdf(figures[-1], f'mpc_plot_{sim_id}_3')