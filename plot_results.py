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

    for avg_return, legend in zip(test_avg_return_list, legend_names):
        x = np.arange(steps_to_plot)
        if x_axis_env_steps:
            x = avg_return[:steps_to_plot, 1]
        y = avg_return[:steps_to_plot, 2]  # 0 = time, 1 = steps, 2 = value

        plt.plot(x, y, label=legend)

        print("legend:", legend, "| max y", np.max(y), "at epoch", np.argmax(y))

    plt.xlabel("Epoch")
    plt.ylabel("Average Return")
    plt.grid()
    plt.title("Meta Test Average Return")
    plt.legend(frameon=True, prop={'size': 14})

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    plt.savefig(fname=os.path.join(out_path, fname + ".svg"), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    log_path = "./logs/Tensorboard_MAML45/discount-f"
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
