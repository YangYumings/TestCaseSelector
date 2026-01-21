import os
import warnings
import sys
import argparse

warnings.filterwarnings("ignore", category=FutureWarning)  # 全局忽略 FutureWarning
os.environ["PYTHONWARNINGS"] = "ignore"  # 对子进程也生效

from config.Config import ConfigBuilder
from pathlib import Path
from datetime import datetime
from stable_baselines.bench import Monitor
from dataSet.CiCycle import CycleTestCases
from dataSet.TestCaseLoader import TestCaseLoader
from select.CustomCallback import CustomCallback
from select.agent.Agent import Agent
from select.env.envFactory.FactoryRegistry import FactoryRegistry
from util.Util import reportDatasetInfo, millis_interval, get_steps
from util.logger import LogManager

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # 只报 Error


def experiment(config, cycle_list, verbos=False):
    """
    实验
    1.初始周期验证
        model_path ： ../experiments/pointwise/A2C/paintcontrol-additional-features_4/
        conf.output_path  =  model_path = '../experiments/pointwise/A2C/paintcontrol-additional-features_4/'
        model_save_path = ../experiments/pointwise/A2C/paintcontrol-additional-features_4/pointwise_A2C_0_7
        log_dir = '../experiments/pointwise/A2C/paintcontrol-additional-features_4/pointwise_A2C_paintcontrol-additional-features_10_4_log.txt'
    """
    first_round = config.start_cycle <= 0
    model_save_path = None
    """
    2.日志管理器初始化
    """
    logManager = LogManager(config)
    """
    3.每周期处理
    """
    for i in range(config.start_cycle, config.end_cycle - 1):  # 使用 [start_cycle, end_cycle] 训练数据
        if (cycle_list[i].get_test_cases_count() < 6) or \
                ((config.dataset_type == "simple") and
                 (cycle_list[i].get_failed_test_cases_count() < 1)):
            continue
        """
        2.1 创造环境
        """
        factory = FactoryRegistry.getFactory(config.mode)
        env = factory.create_environment(config, cycle_list[i])
        N = cycle_list[i].get_test_cases_count()
        steps = get_steps(N, config.episodes)
        print(f"Training agent with replaying of cycle {i} with steps {steps}")

        previous_model_path = model_save_path  # 保存上一个模型路径，如果是首次训练则为 None
        model_save_path = f"{config.log_dir}/{config.mode}_{config.algo}_{config.data_filename}_{config.start_cycle}_{i}"  # 生成当前模型保存路径

        '''
        记录三大指标：
        累计奖励、episode长度、耗时
        设置监视器和自定义回调函数（自动保存更好的模型参数）
        '''
        env = Monitor(env, model_save_path + "_monitor.csv")
        callback_class = CustomCallback(save_path=model_save_path,
                                        check_freq=int(steps / config.episodes), log_dir=config.log_dir, verbose=verbos)

        """
        2.2 创建代理 + 一轮训练
        """
        agent_instance = Agent()
        if first_round:  # 第一次训练，根据算法与环境创建代理
            tp_agent = agent_instance.create_model(config.algo, env)
            training_start_time = datetime.now()
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)  # 训练
            training_end_time = datetime.now()
            first_round = False
        else:
            tp_agent = agent_instance.load_model(algo=config.algo, env=env,
                                                 path=previous_model_path + ".zip")  # 使用上一次训练后的模型参数
            training_start_time = datetime.now()
            tp_agent.learn(total_timesteps=steps, reset_num_timesteps=True, callback=callback_class)
            training_end_time = datetime.now()
        print("Training agent with replaying of cycle " + str(i) + " is finished")

        """
        2.3 预测下一轮结果
        """
        j = i + 1  # 下一个测试用例大于6个的周期
        while (((cycle_list[j].get_test_cases_count() < 6)
                or ((config.dataset_type == "simple") and (cycle_list[j].get_failed_test_cases_count() == 0)))
               and (j < config.end_cycle)):
            j = j + 1
        if j >= config.end_cycle - 1:
            break

        """
        2.4 为预测创建环境
        """
        factory_test = FactoryRegistry.getFactory(config.mode.upper())
        env_test = factory_test.create_environment(config, cycle_list[i])
        test_time_start = datetime.now()
        test_case_vector = agent_instance.test_agent(env=env_test, algo=config.algo,
                                                     model_path=model_save_path + ".zip",
                                                     mode=config.mode)  # 预测 sorted_test_case.csv 保存预测的顺序  此处返回所有测试用例的序列
        test_time_end = datetime.now()

        """
        2.5 数据记录
        """
        test_case_id_vector = []
        test_case_id_vector_optimal = []
        test_case_verdict_vector = []  # 存储排序后测试用例的实际通过情况
        test_case_verdict_vector_optimal = []

        selected_test_case_optimal = CycleTestCases(cycle_id=j)
        selected_test_case = CycleTestCases(cycle_id=j)
        time_optimal = 0.0
        time = 0.0
        revenue_threshold = float()

        # 就是指标计算
        if cycle_list[j].get_failed_test_cases_count() != 0:
            order = cycle_list[j].get_optimal_order()
            for test_case in order:
                if test_case['verdict'] == 0:
                    break
                else:
                    selected_test_case_optimal.test_cases.append(test_case)
                    time_optimal += test_case['avg_exec_time']

            """
            指标计算 + 记录计算
            """
            napfd_optimal = cycle_list[j].calc_NAPFD_ordered_vector(selected_test_case_optimal.test_cases)
            failed_num = cycle_list[j].get_failed_test_cases_count_part(selected_test_case_optimal.test_cases)
            dc_optimal = failed_num / len(selected_test_case_optimal.test_cases)

            for test_case in selected_test_case_optimal.test_cases:
                test_case_id_vector_optimal.append(str(test_case['test_id']))
                test_case_verdict_vector_optimal.append(str(test_case['verdict']))

        else:
            time_optimal = 0.0
            napfd_optimal = 1.0
            dc_optimal = 1.0
            test_case_id_vector_optimal = []
            test_case_verdict_vector_optimal = []

        # 预测结果，test_case_vector 是模型输出的每个测试用例的失败概率。需要进一步处理，以添加概率
        order = test_case_vector
        for test_case in order:
            unit_time_revenue = test_case['prob'] / test_case['avg_exec_time']
            if unit_time_revenue > revenue_threshold:
                selected_test_case.test_cases.append(test_case)

        # 时间阈值
        # for test_case in order:
        #     if time + test_case['avg_exec_time'] > T_budget:
        #         break
        #     else:
        #         selected_test_case.test_cases.append(test_case)
        #         time += test_case['avg_exec_time']

        napfd = cycle_list[j].calc_NAPFD_ordered_vector(selected_test_case.test_cases)
        failed_num = cycle_list[j].get_failed_test_cases_count_part(selected_test_case.test_cases)
        dc = failed_num / len(selected_test_case.test_cases)

        for test_case in selected_test_case.test_cases:
            test_case_id_vector.append(str(test_case['test_id']))
            test_case_verdict_vector.append(str(test_case['verdict']))

        test_time = millis_interval(test_time_start, test_time_end)
        training_time = millis_interval(training_start_time, training_end_time)
        """
        2.6 控制台打印
        """
        logManager.print_test_results(j, napfd, napfd_optimal, dc, dc_optimal, cycle_list,
                                      time_optimal, time, test_case_verdict_vector, test_case_verdict_vector_optimal,
                                      test_case_id_vector, test_case_id_vector_optimal)

        """
        2.7 日志记录
        """
        logManager.write_log_entry(config, model_save_path, steps, j,
                                   training_time, test_time, cycle_list,
                                   time_optimal, time, test_case_verdict_vector, test_case_verdict_vector_optimal,
                                   test_case_id_vector, test_case_id_vector_optimal,
                                   napfd, napfd_optimal, dc, dc_optimal)

        logManager.write_csv_entry(config, model_save_path, steps, j,
                                   training_time, test_time, cycle_list,
                                   time_optimal, time, test_case_verdict_vector, test_case_verdict_vector_optimal,
                                   test_case_id_vector, test_case_id_vector_optimal,
                                   napfd, napfd_optimal, dc, dc_optimal)


''' 
    方法入口
'''
if __name__ == '__main__':
    ''' 
    1.参数解析
    '''
    parser = argparse.ArgumentParser(description='Test case ')
    recursion_limit = sys.getrecursionlimit()
    print("Recursion limit:" + str(recursion_limit))
    sys.setrecursionlimit(1000000)

    parser.add_argument('-m', '--mode', help='[pairwise,pointwise,listwise] ', required=True)
    parser.add_argument('-a', '--algo', help='[a2c,dqn,..]', required=True)
    parser.add_argument('-d', '--dataset_type', help='simple, enriched', required=False, default="simple")
    parser.add_argument('-e', '--episodes', help='Training episodes ', required=True)
    parser.add_argument('-w', '--win_size', help='Windows size of the history', required=False)
    parser.add_argument('-t', '--train_data', help='Train set folder', required=True)
    parser.add_argument('-f', '--first_cycle', help='first cycle used for training', required=False)
    parser.add_argument('-c', '--cycle_count', help='Number of cycle used for training', required=False)
    parser.add_argument('-l', '--list_size', help='Maximum number of test case per cycle', required=False)
    parser.add_argument('-o', '--output_path', help='Output path of the agent model', required=False)
    parser.add_argument('-n', '--notes', help='Statement of Purpose for the Experiment',
                        required=False)  # 用于指明每个实验的目的，免于所有实验记录都放到同一个目录中，当前存在问题。
    parser.add_argument('-s', '--time_threshold', help='time threshold', required=False)

    supported_formalization = ['PAIRWISE', 'POINTWISE', 'LISTWISE', 'LISTWISE2']
    supported_algo = ['DQN', 'PPO2', "A2C", "ACKTR", "DDPG", "ACER", "GAIL", "HER", "PPO1", "SAC", "TD3", "TRPO"]
    args = parser.parse_args()

    '''
    2.观察所有参数的值，字典格式
    '''
    print(vars(args))
    assert supported_formalization.count(args.mode.upper()) == 1, "The formalization mode is not set correctly"
    assert supported_algo.count(args.algo.upper()) == 1, "The formalization mode is not set correctly"

    '''
    3.参数初始化，配置对象赋值
    '''
    config = (ConfigBuilder()  # 构建器模式
              .with_win_size(int(args.win_size))  # 时间窗口大小
              .with_start_cycle(int(args.first_cycle))  # 初始周期数
              .with_dataset_type(args.dataset_type)  # 数据集类型
              .with_log_dir()  # 日志文件输出文件夹
              .with_log_filename()  # 日志文件名
              .with_mode(args.mode)  # 模式
              .with_algo(args.algo)  # 算法
              .with_episodes(args.episodes)  # 训练步数
              .with_train_data_path(args.train_data)  # 训练数据路径
              .with_data_filename(Path(args.train_data).stem)  # 数据集文件名
              .with_time_threshold(args.time_threshold)  # 时间阈值
              .build())

    print(config.log_dir)
    print("a")
    print(config.log_filename)
    '''
    4.数据加载，预处理  
    TODO 验证前面 CI 周期号是否能指定训练集大小
    '''
    loader = TestCaseLoader(config.train_data_path, config.dataset_type)  # 数据加载对象实例
    test_cases = loader.load()  # 数据加载，扫描CSV文件
    cycle_list = loader.preProcess()  # 数据预处理，列表转实例
    config.end_cycle = config.start_cycle + config.cycle_count - 1

    reportDatasetInfo(test_case_data=cycle_list)  # 控制台打印数据集信息， 每轮 CI 的相关测试用例数，失败率等

    '''
    5.实验环节
    执行模式、算法、数据集、训练回合数、起始周期、结束周期、模型输出路径、config
    '''
    experiment(config, cycle_list=cycle_list, verbos=False)
