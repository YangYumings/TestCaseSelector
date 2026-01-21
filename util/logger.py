import os
from openpyxl import Workbook
from datetime import datetime
from pathlib import Path

from config.Config import Config


class LogManager:
    def __init__(self, config: Config):
        self.config = config
        self.wb = None
        self.ws = None
        self.log_file, self.csv_file = self.initialize_files()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.log_file:
            self.log_file.close()
        if self.csv_file:
            self.csv_file.close()
        if self.wb:
            self.wb.save(self.config.log_dir + "/log_data.xlsx")

    def initialize_files(self):
        """初始化日志文件"""
        if not os.path.exists(self.config.log_dir):
            os.makedirs(self.config.log_dir)

        self.log_file = open(str(Path(self.config.log_dir) / self.config.log_filename), "a")
        self.csv_file = open(str(self.config.log_dir + "/sorted_test_case.csv"), "a")

        # 写入日志文件表头
        self.log_file.write(
            "timestamp,mode,algo,model_name,episodes,steps,cycle_id,training_time,testing_time,"
            "winsize,test_cases_count,failed_test_cases_count,time_optimal,time,verdict_list,"
            "verdict_optimal_list,select_test_case_id,optimal_select_test_case_id,napfd,napfd_optimal,dc,dc_optimal"
            + os.linesep)

        # 初始化Excel工作簿
        self.wb = Workbook()
        self.ws = self.wb.active
        self.ws.title = "Log Data"
        header = [
            "timestamp", "mode", "algo", "model_name", "episodes", "steps", "cycle_id",
            "training_time", "testing_time", "winsize", "test_cases_count", "failed_test_cases_count",
            "time_optimal", "time", "verdict_list", "verdict_optimal_list", "select_test_case_id",
            "optimal_select_test_case_id", "napfd", "napfd_optimal", "dc", "dc_optimal"
        ]
        self.ws.append(header)

        return self.log_file, self.csv_file

    def write_log_entry(self, config, model_save_path, steps, j,
                        training_time, test_time, test_case_data,
                        time_optimal, time, test_case_verdict_vector, test_case_verdict_vector_optimal,
                        test_case_id_vector, test_case_id_vector_optimal,
                        napfd, napfd_optimal, dc, dc_optimal):
        """写入单条日志记录"""
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        model_name = Path(model_save_path).stem

        # 写入文本日志文件
        self.log_file.write(
            f"{timestamp},{config.mode},{config.algo},{model_name},{config.episodes},{steps},{j},"
            f"{training_time},{test_time},{config.win_size},"
            f"{test_case_data[j].get_test_cases_count()},{test_case_data[j].get_failed_test_cases_count()},"
            f"{time_optimal},{time},{'|'.join(test_case_verdict_vector)},"
            f"{'|'.join(test_case_verdict_vector_optimal)},{'|'.join(test_case_id_vector)},"
            f"{'|'.join(test_case_id_vector_optimal)},{napfd},{napfd_optimal},{dc},{dc_optimal}"
            + os.linesep
        )
        self.log_file.flush()

    def write_csv_entry(self, config, model_save_path, steps, j,
                        training_time, test_time, test_case_data,
                        time_optimal, time, test_case_verdict_vector, test_case_verdict_vector_optimal,
                        test_case_id_vector, test_case_id_vector_optimal,
                        napfd, napfd_optimal, dc, dc_optimal):
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        model_name = Path(model_save_path).stem
        # 写入测试用例文件
        self.csv_file.write(
            f"{timestamp},{config.mode},{config.algo},{model_name},{config.episodes},{steps},{j},"
            f"{training_time},{test_time},{config.win_size},{dc},{'|'.join(test_case_id_vector)}"
            + os.linesep
        )

        self.csv_file.flush()

        # 添加到Excel工作表
        row = [
            timestamp, config.mode, config.algo, model_name, str(config.episodes), str(steps), str(j),
            str(training_time), str(test_time), str(config.win_size),
            str(test_case_data[j].get_test_cases_count()),
            str(test_case_data[j].get_failed_test_cases_count()),
            str(time_optimal), str(time), "|".join(test_case_verdict_vector),
            "|".join(test_case_verdict_vector_optimal),
            "|".join(test_case_id_vector), "|".join(test_case_id_vector_optimal),
            str(napfd), str(napfd_optimal), str(dc), str(dc_optimal)
        ]
        self.ws.append(row)

    """打印测试结果到控制台"""

    def print_test_results(self, j, napfd, napfd_optimal, dc, dc_optimal, test_case_data,
                           time_optimal, time, test_case_verdict_vector, test_case_verdict_vector_optimal,
                           test_case_id_vector, test_case_id_vector_optimal):
        print(
            f"Testing agent on cycle {j}, "
            f"NAPFD: {napfd}, NAPFD_optimal: {napfd_optimal}, "
            f"Defect Coverage: {dc}, Optimal Defect Coverage: {dc_optimal}, "
            f"test cases num: {test_case_data[j].get_test_cases_count()}, "
            f"failed test num: {test_case_data[j].get_failed_test_cases_count()}, "
            f"time: {time_optimal}, Optimal time: {time}, "
            f"verdict list: {'|'.join(test_case_verdict_vector)}, "
            f"Optimal verdict list: {'|'.join(test_case_verdict_vector_optimal)}, "
            f"selected_test_case: {'|'.join(test_case_id_vector)}, "
            f"selected_test_case_optimal: {'|'.join(test_case_id_vector_optimal)}",
            flush=True
        )
