import numpy as np
import copy

from sklearn import preprocessing


class CycleTestCases:
    """
    用于输出每个 CI 周期的记录
    """
    test_cases = {}
    cycle_id = 0

    def __init__(self, cycle_id: int, test_cases=None):
        self.cycle_id = cycle_id
        self.test_cases = test_cases if test_cases is not None else []

    def add_test_case(self, cycle_id, test_id, test_suite, avg_exec_time: int, last_exec_time: int, verdict: int,
                      failure_history: list, duration_group, time_group, exec_time_history: list):
        test_cases = {'test_id': test_id, 'test_suite': test_suite, 'avg_exec_time': avg_exec_time, 'verdict': verdict,
                      'duration_group': duration_group, 'time_group': time_group, 'cycle_id': cycle_id,
                      'last_exec_time': last_exec_time, 'prob': 0.0}
        if failure_history:
            test_cases['failure_history'] = failure_history
            test_cases['age'] = len(failure_history)
        else:
            test_cases['failure_history'] = []
            test_cases['age'] = 0
        if exec_time_history:
            test_cases['exec_time_history'] = exec_time_history
        self.test_cases.append(test_cases)

    def delete_test_case(self, test_id):
        if self.test_cases[test_id]:
            del self.test_cases[test_id]

    def get_test_cases_count(self):
        return len(self.test_cases)

    def get_failed_test_cases_count(self):
        cnt = 0
        for test_case in self.test_cases:
            if test_case['verdict'] == 1:
                cnt = cnt + 1
        return cnt

    def get_optimal_order(self):
        """
        生成最优排序
        :return:
        """
        optimal_order_by_verdict = copy.deepcopy(sorted(self.test_cases, key=lambda x: x['verdict'], reverse=True))
        optimal_order = []
        optimal_order.extend(
            sorted(optimal_order_by_verdict[0:self.get_failed_test_cases_count()], key=lambda x: x['last_exec_time']))
        optimal_order.extend(
            sorted(optimal_order_by_verdict[self.get_failed_test_cases_count():], key=lambda x: x['last_exec_time']))
        return optimal_order

    def get_test_case_vector_length(self, test_case, win_size):
        """
        测试用例向量长度 = 额外特征长度 + 历史窗口长度
        :param test_case:
        :param win_size:
        :return:
        """
        extra_length = 4
        if 'complexity_metrics' in test_case.keys():
            extra_length = extra_length + len(test_case['complexity_metrics'])
        if 'other_metrics' in test_case.keys():
            extra_length = extra_length + len(test_case['other_metrics'])

        return win_size + extra_length

    def export_test_cases(self, option: str, pad_digit=9, max_test_cases_count=0, winsize=4, test_case_vector_size=7):
        """
        将测试用例集合转换为标准化的数值数组表示，用于机器学习模型的输入。
        :param option:
        :param pad_digit:
        :param max_test_cases_count:
        :param winsize:
        :param test_case_vector_size:
        :return:
        """
        if option == "list_avg_exec_with_failed_history":
            # assume param1 refers to the number of test cases,
            # params 2 refers to the history windows size, and param3 refers to pa
            test_cases_array = np.zeros((max_test_cases_count, test_case_vector_size))
            i = 0
            for test_case in self.test_cases:
                test_cases_array[i] = self.export_test_case(test_case,
                                                            "list_avg_exec_with_failed_history", win_size=winsize)
                i = i + 1
            for i in range(len(self.test_cases), max_test_cases_count):
                test_cases_array[i] = np.repeat(pad_digit, test_case_vector_size)
            test_cases_array = preprocessing.normalize(test_cases_array, axis=0, norm='max')
            # test_cases_array[:, 1] = preprocessing.normalize(test_cases_array[:, 1])
            return test_cases_array
        else:
            return None

    def export_test_case(self, test_case: dict, option: str, pad_digit=9, win_size=4):
        """
        将单个测试用例转换为数值化特征向量：将测试用例的字典数据转换为固定长度的数值向量
        向量： failure_history、other metrics、avg_exec_time 、 age、 time_group、duration_group
        :param test_case:
        :param option:
        :param pad_digit:
        :param win_size:
        :return:
        """
        if option == "list_avg_exec_with_failed_history":
            # assume param1 refers to the number of test cases,
            # params 2 refers to the history windows size, and param3 refers to pa
            # 特征向量长度计算
            extra_length = 4
            if 'complexity_metrics' in test_case.keys():
                extra_length = extra_length + len(test_case['complexity_metrics'])
            if 'other_metrics' in test_case.keys():
                extra_length = extra_length + len(test_case['other_metrics'])

            test_case_vector = np.zeros((win_size + extra_length))

            # 特征向量填充
            index_1 = 0
            for j in range(0, len(test_case['failure_history'])):
                if j >= win_size:
                    break
                test_case_vector[j] = test_case['failure_history'][j]
                index_1 = index_1 + 1
            for j in range(len(test_case['failure_history']), win_size):
                test_case_vector[j] = pad_digit
                index_1 = index_1 + 1
            if 'complexity_metrics' in test_case.keys():
                index_2 = index_1
                for j in range(index_2, index_2 + len(test_case['complexity_metrics'])):
                    test_case_vector[j] = test_case['complexity_metrics'][j - index_2]
                    index_1 = index_1 + 1
            if 'other_metrics' in test_case.keys():
                index_2 = index_1
                for j in range(index_2, index_2 + len(test_case['other_metrics'])):
                    test_case_vector[j] = test_case['other_metrics'][j - index_2]
                    index_1 = index_1 + 1

            test_case_vector[index_1] = test_case['avg_exec_time']
            test_case_vector[index_1 + 1] = test_case['age']
            if 'time_group' in test_case.keys():
                test_case_vector[index_1 + 2] = test_case['time_group']
            else:
                test_case_vector[index_1 + 2] = 0
            if 'duration_group' in test_case.keys():
                test_case_vector[index_1 + 3] = test_case['duration_group']
            else:
                test_case_vector[index_1 + 3] = 0
            return test_case_vector
        else:
            return None

    # 移动， 方是完整测试用例的 CICycle
    def calc_NAPFD_ordered_vector(self, test_case_vector: list):
        # 原测试用例长度
        ts_len = len(self.test_cases)
        # 子集的长度
        sub_ts_len = len(test_case_vector)
        # 子集失败的个数
        sub_ts_fail = len([tc for tc in test_case_vector if tc['verdict'] == 1])
        # 所有失败的个数
        ts_fail = self.get_failed_test_cases_count()

        print(f"测试用例统计 - 总数: {ts_len}, 子集: {sub_ts_len}, 子集失败: {sub_ts_fail}, 总失败: {ts_fail}")

        # 缺陷数
        p = sub_ts_fail / ts_fail

        rank_sum = 0.0
        for i, test_case in enumerate(test_case_vector):
            if test_case['verdict'] == 1:  # 如果测试用例失败
                rank_sum += (i + 1)

        if sub_ts_len > 0:
            napfd = p - rank_sum / (ts_len * sub_ts_len) + p / (2 * ts_len)
        else:
            napfd = 0

        return napfd

    def get_failed_test_cases_count_part(self, test_case: list):
        total = 0
        for ts in test_case:
            if ts['verdict'] == 1:
                total = total + 1
        return total
