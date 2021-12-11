# based on https://gist.github.com/tomrunia/1e1d383fb21841e8f144
import os
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent

# TODO not sure if this is the correct font?
# use latex font
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 18
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['figure.figsize'] = (8, 5)
mpl.rcParams['legend.handletextpad'] = 0.3
mpl.rcParams['legend.handlelength'] = 1.0
mpl.rcParams['legend.handleheight'] = 0.2
mpl.rcParams['legend.labelspacing'] = 0.1


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    # https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def get_scalars(event_acc, tags):
    scalars = dict()
    for tag in tags:
        task_name = tag.split('/')[-2]
        scalars[task_name] = event_acc.Scalars(tag)

    return scalars


def get_all_test_scalars(input_log_file_path: str):
    event_acc = EventAccumulator(input_log_file_path)
    event_acc.Reload()

    # get all scalar tags in the log file
    tags_scalars = event_acc.Tags()['scalars']

    # filter MetaTest tags
    meta_test_tags = [tag for tag in tags_scalars if "MetaTest" in tag]

    meta_test_tags_average_return = [tag for tag in meta_test_tags if "AverageReturn" in tag]
    meta_test_tags_success_rate = [tag for tag in meta_test_tags if "SuccessRate" in tag]

    test_avg_return = get_scalars(event_acc, meta_test_tags_average_return)
    test_success_rate = get_scalars(event_acc, meta_test_tags_success_rate)

    return test_avg_return, test_success_rate


def get_x_y_values(scalar_event: ScalarEvent,
                   override_steps_to_plot: bool,
                   smoothing_factor: float,
                   steps_to_plot: int,
                   use_env_steps_as_x_axis: bool):
    scalar_event = np.asarray(scalar_event)  # dimension 1: 0 = time, 1 = steps, 2 = value

    if override_steps_to_plot:
        steps_to_plot = scalar_event.shape[0]

    if use_env_steps_as_x_axis:
        x = scalar_event[:steps_to_plot, 1]
    else:
        x = np.arange(steps_to_plot)

    y = scalar_event[:steps_to_plot, 2]

    if smoothing_factor != 0.0:
        y = smooth(y, smoothing_factor)

    return x, y


def plot_per_test_task(scalar_event_list: List[ScalarEvent], env: str, fname_prefix: str, fname_postfix: str, legend_names: str,
                       out_path: str, override_steps_to_plot: bool, smoothing_factor: float, steps_to_plot: int, x_axis_env_steps: bool,
                       y_label: str):
    for i, experiment_name in enumerate(legend_names):
        scalars = scalar_event_list[i]
        for key, value in scalars.items():
            if key == 'Average':  # do not plot average
                continue
            x, y = get_x_y_values(value, override_steps_to_plot, smoothing_factor, steps_to_plot, x_axis_env_steps)

            plt.plot(x, y, label=key)

        if x_axis_env_steps:
            plt.xlabel("Training Environment Steps")
        else:
            plt.xlabel("Epoch")
        plt.ylabel(y_label)
        plt.grid()
        plt.title(env + " Meta Testing")
        plt.legend(frameon=True, prop={'size': 14})
        plt.savefig(fname=os.path.join(out_path, "_".join([fname_prefix, experiment_name, fname_postfix]) + ".svg"), bbox_inches='tight')
        plt.show()


def plot_tensorflow_log(log_file_path_list, legend_names, env, algo, exp_name,
                        max_epochs_to_plot=300,
                        out_path="./figures",
                        x_axis_env_steps=True,
                        smoothing_factor=0.6,
                        override_steps_to_plot=True):
    test_avg_return_list = list()
    test_success_rate_list = list()

    steps_to_plot = max_epochs_to_plot
    for log_path in log_file_path_list:
        test_avg_return, test_success_rate = get_all_test_scalars(log_path)

        test_avg_return_list.append(test_avg_return)
        test_success_rate_list.append(test_success_rate)

        nr_entries = len(list(test_avg_return.values())[0])
        steps_to_plot = min(steps_to_plot, nr_entries)

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
                       legend_names=legend_names,
                       out_path=out_path,
                       override_steps_to_plot=override_steps_to_plot,
                       smoothing_factor=smoothing_factor,
                       steps_to_plot=steps_to_plot,
                       x_axis_env_steps=x_axis_env_steps,
                       y_label="Success Rate")

    # plot avg per experiment
    plot_per_test_task(scalar_event_list=test_avg_return_list,
                       env=env,
                       fname_prefix=output_file_name,
                       fname_postfix="AverageReturn",
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

    main_log_path = "./logs/MAML"
    env_dirs = [dir for dir in os.listdir(main_log_path)]

    log_paths = gather_all_log_paths(main_log_path=main_log_path, env_dirs=env_dirs)

    # Overwrite log paths with specific experiment to only plot specific experiments
    # log_paths = ["./logs/Tensorboard_MAML45/discount-f"]

    for log_path in log_paths:
        print("---------------")
        print(log_path)
        print("---------------")

        out_file_name = log_path.replace("/", "_")[2:]
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
                            env=log_path.split("/")[-2],
                            algo=log_path.split("/")[-3],
                            exp_name=log_path.split("/")[-1])
