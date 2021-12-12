from typing import List

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator, ScalarEvent


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


def get_scalar_lists(log_file_path_list: List[str]):
    test_avg_return_list = list()
    test_success_rate_list = list()
    min_nr_steps = 10000

    for log_path in log_file_path_list:
        test_avg_return, test_success_rate = get_all_test_scalars(log_path)

        test_avg_return_list.append(test_avg_return)
        test_success_rate_list.append(test_success_rate)

        nr_entries = len(list(test_avg_return.values())[0])
        min_nr_steps = min(min_nr_steps, nr_entries)

    return min_nr_steps, test_avg_return_list, test_success_rate_list


def smooth(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    # https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


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
