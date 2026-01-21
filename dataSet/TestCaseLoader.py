import pandas as pd

from dataSet.CiCycle import CycleTestCases


class TestCaseLoader:

    def __init__(self, data_path, dataset_type):
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.test_cases = None

    def load(self):
        """
        数据加载，扫描 CSV 文件，存储其周期内每个测试用例的所有特征
        返回：列表，文件中所有测试用例
        """
        last_results = []  # last_results 列
        if self.dataset_type == "simple":
            df = pd.read_csv(self.data_path, error_bad_lines=False, sep=",")  # pandas 读取CSV数据
            for i in range(df.shape[0]):  # 遍历 csv 文件每一行
                last_result_str: str = df["LastResults"][i]  # 提出 LastResults 列，去除中括号，转为整数列表
                temp_list = (last_result_str.strip("[").strip("]").split(","))
                if temp_list[0] != '':  # 若为空，转为空列表
                    last_results.append(list(map(int, temp_list)))
                else:
                    last_results.append([])
            df["LastResults"] = last_results
            self.test_cases = df
        return self.test_cases

    def preProcess(self):
        """
        1.根据 Cycle 列确定 CI 周期数范围；
        2.遍历每个 CI，搜集每轮 CI 执行的测试用例；
            2.1 保存在一个 CICycleLog 中；维护 该周期数cycle_id  和 存放该周期测试用例特征的字典test_cases
            2.2 将 pd 表格中的数据转换成为 字典
            test_id = id  不是测试用例 id
            avg_exec_time = Duration
            last_exec_time = Duration 由于数据集只有一个平均值，此外真实场景下同一测试用例每次的执行时间也差不多
            verdict = Verdict 本次执行结果
            failure_history = LastResults 历史记录，不包括本次  失败的历史记录是头插法，因此预处理数据时要注意
            age = failure_history 的长度，代表测试用例的历史深度，也就是这个测试用例再当前 CI 前活跃了多久
            cycle_id = Cycle
            duration_group = DurationGroup
            time_group = TimeGroup
        3.返回 ci_cycle_logs
        """
        min_cycle = min(self.test_cases["Cycle"])  # 查找数据中最小和最大的构建周期(Cycle)
        max_cycle = max(self.test_cases["Cycle"])
        cycle_list = []  # 初始化空列表，用于存储处理后的CI周期日志
        if self.dataset_type == 'simple':
            # 遍历所有周期ID，为每个周期创建一个 CICycleLog 实例，并将所有测试用例添加到该实例中
            for i in range(min_cycle, max_cycle + 1):
                cycle_test_cases = CycleTestCases(i)
                raw = self.test_cases.loc[self.test_cases['Cycle'] == i]
                # ID、名称、执行时间、执行结果、失败历史、周期ID、执行时间分组、时间分组、执行时间历史
                for index, test_case in raw.iterrows():
                    cycle_test_cases.add_test_case(test_id=test_case["Id"], test_suite=test_case["Name"],
                                                   avg_exec_time=test_case["Duration"],
                                                   last_exec_time=test_case["Duration"],
                                                   verdict=test_case["Verdict"],
                                                   failure_history=test_case["LastResults"],
                                                   cycle_id=test_case["Cycle"],
                                                   duration_group=test_case["DurationGroup"],
                                                   time_group=test_case["TimeGroup"],
                                                   exec_time_history=None)
                cycle_list.append(cycle_test_cases)  # 将处理完的周期日志添加到列表中
        return cycle_list
