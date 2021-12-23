# based on https://gist.github.com/tomrunia/1e1d383fb21841e8f144
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import ScalarEvent

from matplotlib_utils import set_matplotlib_properties
from tensorboard_utils import get_scalar_lists, get_x_y_values


def plot_per_test_task(scalar_event_list: List[ScalarEvent], env: str, fname_prefix: str, exp_name: str, fname_postfix: str, legend_names: str,
                       out_path: str, override_steps_to_plot: bool, smoothing_factor: float, steps_to_plot: int, x_axis_env_steps: bool,
                       y_label: str, annotate=False):
    for i, experiment_name in enumerate(legend_names):
        scalars = scalar_event_list[i]
        fig, ax = plt.subplots()
        fig_bar, ax_bar = plt.subplots()
        for idx, (key, value) in enumerate(scalars.items()):
            if key == 'Average':  # do not plot average
                continue
            x, y = get_x_y_values(value, override_steps_to_plot, smoothing_factor, steps_to_plot, x_axis_env_steps)

            ax.plot(x, y, label=key)

            max_y = np.max(y)
            ax_bar.barh(y=key, width=max_y, height=0.6)

            if annotate:
                ax_bar.annotate(np.round(max_y, decimals=2),
                                xy=(max_y, idx),
                                ha='left', va='center')

        if annotate:
            ax_bar.set_xlim(tuple(1.05 * limit for limit in ax_bar.get_xlim()))
        ax_bar.grid(True, which='both', axis='x')
        ax_bar.set_xlabel("Maximum " + y_label)
        ax_bar.set_title(env + " Meta Testing")
        fig_bar.savefig(fname=os.path.join(out_path, "_".join([fname_prefix, exp_name, experiment_name, fname_postfix, "bar"]) + ".svg"),
                        bbox_inches='tight')
        fig_bar.show()

        if x_axis_env_steps:
            ax.set_xlabel("Training Environment Steps")
        else:
            ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        ax.grid()
        ax.set_title(env + " Meta Testing")
        ax.legend(frameon=True, prop={'size': 14})
        fig.savefig(fname=os.path.join(out_path, "_".join([fname_prefix, exp_name, experiment_name, fname_postfix]) + ".svg"), bbox_inches='tight')
        fig.show()


def plot_tensorflow_log(log_file_path_list, legend_names, env, algo, exp_name,
                        max_epochs_to_plot=300,
                        out_path="./figures",
                        x_axis_env_steps=True,
                        smoothing_factor=0.6,
                        override_steps_to_plot=True):
    min_nr_steps, test_avg_return_list, test_success_rate_list = get_scalar_lists(log_file_path_list)

    steps_to_plot = min(min_nr_steps, max_epochs_to_plot)

    log_dir = os.path.join(out_path, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    output_file_name = "_".join([env, algo, exp_name])
    log_file = open(os.path.join(log_dir, output_file_name + ".log"), "w")
    log_str = str(out_file_name) + "\n"
    log_file.write(log_str)

    # plot average
    for avg_returns, legend in zip(test_avg_return_list, legend_names):
        avg_return = avg_returns['Average']
        x, y = get_x_y_values(avg_return, override_steps_to_plot, smoothing_factor, steps_to_plot, x_axis_env_steps)

        plt.plot(x, y, label=legend)

        log_str = "legend: " + str(legend) + " | max y " + str(np.max(y)) + " at epoch " + str(np.argmax(y)) + "\n"
        print(log_str, end='')
        log_file.write(log_str)

    log_file.close()

    if x_axis_env_steps:
        plt.xlabel("Training Environment Steps")
    else:
        plt.xlabel("Epoch")
    plt.ylabel("Average Test Return")
    plt.grid()
    plt.title(env + " Meta Testing")
    plt.legend(frameon=True, prop={'size': 14})

    out_path = os.path.join(out_path, env)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    plt.savefig(fname=os.path.join(out_path, output_file_name + ".svg"), bbox_inches='tight')
    plt.show()

    output_file_name = "_".join([env, algo])

    # plot success rates per experiment
    plot_per_test_task(scalar_event_list=test_success_rate_list,
                       env=env,
                       fname_prefix=output_file_name,
                       fname_postfix="SuccessRate",
                       exp_name=exp_name,
                       legend_names=legend_names,
                       out_path=out_path,
                       override_steps_to_plot=override_steps_to_plot,
                       smoothing_factor=smoothing_factor,
                       steps_to_plot=steps_to_plot,
                       x_axis_env_steps=x_axis_env_steps,
                       y_label="Success Rate",
                       annotate=True)

    # plot avg per experiment
    plot_per_test_task(scalar_event_list=test_avg_return_list,
                       env=env,
                       fname_prefix=output_file_name,
                       fname_postfix="AverageReturn",
                       exp_name=exp_name,
                       legend_names=legend_names,
                       out_path=out_path,
                       override_steps_to_plot=override_steps_to_plot,
                       smoothing_factor=smoothing_factor,
                       steps_to_plot=steps_to_plot,
                       x_axis_env_steps=x_axis_env_steps,
                       y_label="Average Test Return")


def gather_all_log_paths(main_log_path, env_dirs):
    log_paths = list()
    for env_dir in env_dirs:
        env_log_path = os.path.join(main_log_path, env_dir)

        exp_dirs = [directory for directory in os.listdir(env_log_path) if os.path.isdir(os.path.join(env_log_path, directory))]

        for exp_dir in exp_dirs:
            log_paths.append(os.path.join(env_log_path, exp_dir))

    return log_paths


if __name__ == '__main__':
    set_matplotlib_properties()

    main_log_path = "./logs/MACAW".replace("/", os.path.sep)
    env_dirs = [dir for dir in os.listdir(main_log_path)]

    log_paths = gather_all_log_paths(main_log_path=main_log_path, env_dirs=env_dirs)

    # Overwrite log paths with specific experiment to only plot specific experiments
    # log_paths = ["./logs/Tensorboard_MAML45/discount-f"]

    for log_path in log_paths:
        print("---------------")
        print(log_path)
        print("---------------")

        out_file_name = log_path.replace(os.path.sep, "_")[2:]
        dirs = [directory for directory in os.listdir(log_path) if os.path.isdir(os.path.join(log_path, directory))]
        dirs.sort()

        log_file_path_list = list()
        folder_names = list()
        for log_dir in dirs:
            log_dir_path = os.path.join(log_path, log_dir)
            # filter for tensor board logs
            log_file = [file for file in os.listdir(log_dir_path) if "events" in file][0]
            log_file_path = os.path.join(log_dir_path, log_file)
            folder_names.append(log_dir)
            log_file_path_list.append(log_file_path)

        plot_tensorflow_log(log_file_path_list, legend_names=folder_names,
                            env=log_path.split(os.path.sep)[-2],
                            algo=log_path.split(os.path.sep)[-3],
                            exp_name=log_path.split(os.path.sep)[-1])
