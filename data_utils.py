from typing import List

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


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