import os

import numpy as np
from matplotlib import pyplot as plt

from matplotlib_utils import set_matplotlib_properties
from tensorboard_utils import get_scalar_lists, get_x_y_values


def subcategorybar(X, vals, x_label, legends, width=0.8):
    # https://stackoverflow.com/questions/48157735/plot-multiple-bars-for-categorical-data
    n = len(vals)
    _X = np.arange(len(X))

    fig, ax = plt.subplots()

    for i in range(n):
        ax.barh(_X - width / 2. + i / float(n) * width, vals[i],
                width / float(n), align="edge")

    ax.set_yticks(_X, X)
    ax.legend(legends, frameon=True)
    ax.grid(True, which='both', axis='x')
    ax.set_xlabel(x_label)
    # ax.set_title("Meta Testing - Success Rate")
    fig.savefig(fname=os.path.join(out_path, "_".join(["comparison", x_label.replace(" ", "_"), "bar"]) + ".svg"),
                bbox_inches='tight')
    fig.show()


def max_values_bar_plot(test_list, algo_names, x_label, use_env_steps_as_x_axis):
    max_y = list()
    for i, (scalars, experiment_name) in enumerate(zip(test_list, algo_names)):
        max_y_per_task = dict()
        for idx, (key, value) in enumerate(scalars.items()):
            if key == 'Average':  # do not plot average
                continue
            x, y = get_x_y_values(value,
                                  smoothing_factor=0.6,
                                  override_steps_to_plot=True,
                                  steps_to_plot=300,
                                  use_env_steps_as_x_axis=use_env_steps_as_x_axis)

            max_y_per_task[key] = np.max(y)
        max_y.append(max_y_per_task)
    # dict_per_task = {dict_key: [dic[dict_key] for dic in max_y] for dict_key in max_y[0]}
    subcategorybar(list(max_y[0].keys()), [list(i.values()) for i in max_y], x_label=x_label, legends=algo_names)


def single_tasks_plot(test_list, algo_names, use_env_steps_as_x_axis, y_label, ):
    for i, (scalars, experiment_name) in enumerate(zip(test_list, algo_names)):
        fig, ax = plt.subplots()
        for idx, (key, value) in enumerate(scalars.items()):
            if key == 'Average':  # do not plot average
                continue
            x, y = get_x_y_values(value,
                                  smoothing_factor=0.6,
                                  override_steps_to_plot=True,
                                  steps_to_plot=300,
                                  use_env_steps_as_x_axis=use_env_steps_as_x_axis)
            ax.plot(x, y, label=key)

        if use_env_steps_as_x_axis:
            ax.set_xlabel("Training Environment Steps")
        else:
            ax.set_xlabel("Epoch")

        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend(frameon=True, prop={'size': 14})
        ax.set_title(experiment_name)
        fig.savefig(fname=os.path.join(out_path, experiment_name + "_" + y_label.replace(" ", "_") + ".svg"), bbox_inches='tight')
        fig.show()


if __name__ == '__main__':
    set_matplotlib_properties()

    algo_names = ["MAML",
                  "PEARL",
                  "MACAW"]

    x_axis_nr = [0,
                 0,
                 1]

    paths = ["logs/MAML/ML10/outer_lr/outer-lr=0.001",
             "logs/PEARL_new/ML10/lr/lr=3e-4",
             "logs/MACAW/ML10/lr/pol-lr=0.001_val-lr=0.1"]

    out_path = "./comparison"

    use_env_steps_as_x_axis = True

    ##################################################################################################
    tensorboard_file_paths = list()
    for cur_path in paths:
        tensorboard_file_paths.append([os.path.join(cur_path, file) for file in os.listdir(cur_path) if "events" in file][0])

    min_nr_steps, test_avg_return_list, test_success_rate_list = get_scalar_lists(tensorboard_file_paths)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()

    ax = [ax1, ax2]
    lines = list()
    for i, (avg_returns, legend, ax_nr) in enumerate(zip(test_avg_return_list, algo_names, x_axis_nr)):
        avg_return = avg_returns['Average']
        x, y = get_x_y_values(avg_return,
                              smoothing_factor=0.6,
                              override_steps_to_plot=True,
                              steps_to_plot=300,
                              use_env_steps_as_x_axis=use_env_steps_as_x_axis)
        lines += ax[ax_nr].plot(x, y, label=legend, color=f"C{i}")
        if ax_nr != 0:
            ax[ax_nr].tick_params(axis='x', colors=f"C{i}")
            ax2.set_xlabel("Training Steps", color=f"C{i}")

    if use_env_steps_as_x_axis:
        ax1.set_xlabel("Training Environment Steps")
    else:
        ax1.set_xlabel("Epoch")

    ax1.set_ylabel("Average Test Return")
    ax1.grid(True)

    # ax1.set_title("Meta Testing")

    ax1.legend(lines, algo_names, frameon=True)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    fig.savefig(fname=os.path.join(out_path, "comparison_Average_Return" + ".svg"), bbox_inches='tight')
    fig.show()

    ###############
    max_values_bar_plot(test_list=test_success_rate_list,
                        x_label="Maximum Success Rate",
                        algo_names=algo_names,
                        use_env_steps_as_x_axis=use_env_steps_as_x_axis)

    max_values_bar_plot(test_list=test_avg_return_list,
                        x_label="Maximum Average Test Return",
                        algo_names=algo_names,
                        use_env_steps_as_x_axis=use_env_steps_as_x_axis)

    single_tasks_plot(test_list=test_avg_return_list,
                      y_label="Average Test Return",
                      algo_names=algo_names,
                      use_env_steps_as_x_axis=use_env_steps_as_x_axis)

    single_tasks_plot(test_list=test_success_rate_list,
                      y_label="Success Rate",
                      algo_names=algo_names,
                      use_env_steps_as_x_axis=use_env_steps_as_x_axis)
