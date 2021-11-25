# based on https://gist.github.com/tomrunia/1e1d383fb21841e8f144
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# TODO not sure if this is the correct font?
# use latex font
mpl.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 18
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['figure.figsize'] = (8, 5)


def plot_tensorflow_log(log_file_path_list, legend_names, fname,
                        max_epochs_to_plot=300,
                        out_path="./figures",
                        x_axis_env_steps=False):
    test_avg_return_list = list()

    steps_to_plot = max_epochs_to_plot
    for log_path in log_file_path_list:
        event_acc = EventAccumulator(log_path)
        event_acc.Reload()

        # Show all tags in the log file
        # print(event_acc.Tags())

        test_avg_return = event_acc.Scalars('MetaTest/Average/AverageReturn')
        steps_to_plot = min(steps_to_plot, len(test_avg_return))
        test_avg_return_list.append(np.array(test_avg_return))

    log_dir = os.path.join(out_path, 'logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file = open(os.path.join(log_dir, fname + ".log"), "w")
    log_str = str(out_file_name) + "\n"
    log_file.write(log_str)

    for avg_return, legend in zip(test_avg_return_list, legend_names):
        x = np.arange(steps_to_plot)
        if x_axis_env_steps:
            x = avg_return[:steps_to_plot, 1]
        y = avg_return[:steps_to_plot, 2]  # 0 = time, 1 = steps, 2 = value

        plt.plot(x, y, label=legend)

        log_str = "legend: " + str(legend) + " | max y " + str(np.max(y)) + " at epoch " + str(np.argmax(y)) + "\n"
        print(log_str, end='')
        log_file.write(log_str)

    log_file.close()

    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.grid()
    plt.title("Meta Test Average Return")
    plt.legend(frameon=True, prop={'size': 14})

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    plt.savefig(fname=os.path.join(out_path, fname + ".svg"), bbox_inches='tight')
    plt.show()


def gather_all_log_paths(main_log_path, env_dirs):
    log_paths = list()
    for env_dir in env_dirs:
        env_log_path = os.path.join(main_log_path, env_dir)

        exp_dirs = [directory for directory in os.listdir(env_log_path) if os.path.isdir(os.path.join(env_log_path, directory))]

        for exp_dir in exp_dirs:
            log_paths.append(os.path.join(env_log_path, exp_dir))

    return log_paths


if __name__ == '__main__':

    main_log_path = "./logs/"
    env_dirs = ['Tensorbaord_PEARL10',
                'Tensorboard_MAML10',
                'Tensorboard_MAML1_Basketball',
                'Tensorboard_MAML45',
                'Tensorboard_PEARL1_Basketball']

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

        plot_tensorflow_log(log_file_path_list, legend_names=folder_names, fname=out_file_name)
