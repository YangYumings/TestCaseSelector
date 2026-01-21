import math


def reportDatasetInfo(test_case_data: list):
    """
    控制台输出测试用例列表的基本信息
    :param test_case_data:
    :return:
    """
    cycle_cnt = 0
    failed_test_case_cnt = 0
    test_case_cnt = 0
    failed_cycle = 0
    for cycle in test_case_data:
        if cycle.get_test_cases_count() > 5:
            cycle_cnt = cycle_cnt + 1
            test_case_cnt = test_case_cnt + cycle.get_test_cases_count()
            failed_test_case_cnt = failed_test_case_cnt + cycle.get_failed_test_cases_count()
            if cycle.get_failed_test_cases_count() > 0:
                failed_cycle = failed_cycle + 1
    print(f"# of cycle: {cycle_cnt}, # of test case: {test_case_cnt}, # of failed test case: {failed_test_case_cnt}, "
          f" failure rate:{failed_test_case_cnt / test_case_cnt}, # failed test cycle: {failed_cycle}")


def millis_interval(start, end):
    """
    计算时间差，毫秒
    :param start:
    :param end:
    :return:
    """
    diff = end - start
    millis = diff.days * 24 * 60 * 60 * 1000
    millis += diff.seconds * 1000
    millis += diff.microseconds / 1000
    return millis


def get_steps(N, episodes):
    return int(episodes * (N * (math.log(N, 2) + 1)))
