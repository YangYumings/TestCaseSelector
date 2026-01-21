from pathlib import Path


class Config:
    def __init__(self):
        self.fill_value = -1  # 填补数据，数据预处理阶段填充缺失值
        self.win_size = 10  # 时间窗口大小，后续默认为 10
        self.dataset_type = "simple"  # 数据集类型。根据特征分为 simple、 complex
        self.max_test_case_cnt = 400  # 最大测试用例数量
        self.train_steps = 10000  # 训练步数
        self.discount_factor = 0.9  # 强化学习中的折扣因子，用于未来奖励的计算
        self.experience_replay = False  # 经验回放标志，强化学习训练的一种策略
        self.start_cycle = 0  # 训练的起始周期数
        self.end_cycle = 0  # 训练的结束周期数
        self.cycle_count = 100  # 周期总数,默认训练中 end 是第100个周期
        self.train_data_path = "./dataSet/tc_data_paintcontrol.csv"  # 数据集
        self.data_filename = "paintcontrol-additional-features"  # 数据集文件名
        self.notes = ""  # 备注
        self.reward_threshold = 0.5  # 收益阈值
        self.tim_ratio = 0.6  # 时间预算
        self.mode = "pointwise"  # 3 种模式
        self.algo = "A2C"  # 强化学习算法
        self.episodes = 100  # 训练的轮数
        self.log_dir = str(
            Path('./experiments') / f'{self.mode}/{self.algo}/{self.data_filename}_{self.win_size}')  # 日志文件输出文件夹
        self.log_filename = str(
            f'{self.mode}_{self.algo}_{self.data_filename}_{self.episodes}_{self.win_size}_log.txt')  # 日志文件名
        # WARNING 忘记作用了
        self.top_k_ratio = 0.5  # 选择前 K 个测试用例
        self.time_threshold = 0.5  # 时间阈值

    '''
    生成日志文件路径
    '''

    def generate_log_file_path(self):
        filename = f"{self.mode}_{self.algo}_{self.data_filename}_{self.episodes}_{self.win_size}_log.txt"
        return str(Path(self.log_dir) / filename)


class ConfigBuilder:
    def __init__(self):
        self.config = Config()

    def with_fill_value(self, fill_value):
        if fill_value is None:
            return self
        self.config.fill_value = fill_value
        return self

    def with_win_size(self, win_size):
        if win_size is None or win_size <= 0:
            return self
        self.config.win_size = win_size
        return self

    def with_dataset_type(self, dataset_type):
        if not dataset_type:
            return self
        self.config.dataset_type = dataset_type
        return self

    def with_max_test_case_cnt(self, max_test_case_cnt):
        if max_test_case_cnt is None:
            return self
        self.config.max_test_case_cnt = max_test_case_cnt
        return self

    def with_train_steps(self, train_steps):
        if train_steps is None:
            return self
        self.config.train_steps = train_steps
        return self

    def with_discount_factor(self, discount_factor):
        if discount_factor is None:
            return self
        self.config.discount_factor = discount_factor
        return self

    def with_experience_replay(self, experience_replay):
        if experience_replay:
            return self
        self.config.experience_replay = experience_replay
        return self

    def with_start_cycle(self, start_cycle):
        if start_cycle is None or start_cycle < 0:
            return self
        self.config.start_cycle = start_cycle
        return self

    def with_cycle_count(self, cycle_count):
        if cycle_count is None:
            return self
        self.config.cycle_count = cycle_count
        return self

    def with_train_data_path(self, train_data_path):
        if not train_data_path:
            return self
        self.config.train_data_path = train_data_path
        return self

    def with_data_filename(self, data_filename):
        if not self.config.data_filename:
            return self
        self.config.data_filename = data_filename
        return self

    def with_log_dir(self, log_dir=None):
        if not log_dir:
            return self
        self.config.log_dir = log_dir
        return self

    def with_log_filename(self, log_filename=None):
        if not log_filename:
            return self
        self.config.log_filename = log_filename
        return self

    def with_notes(self, notes):
        self.config.notes = notes
        return self

    def with_reward_threshold(self, reward_threshold):
        if reward_threshold is None:
            return self
        self.config.reward_threshold = reward_threshold
        return self

    def with_tim_ratio(self, tim_ratio):
        if tim_ratio is None:
            return self
        self.config.tim_ratio = tim_ratio
        return self

    def with_mode(self, mode):
        if not mode:
            return self
        self.config.mode = mode
        return self

    def with_algo(self, algo):
        if not algo:
            return self
        self.config.algo = algo
        return self

    def with_episodes(self, episodes):
        if episodes is None:
            return self
        self.config.episodes = int(episodes)
        return self

    def with_top_k_ratio(self, top_k_ratio):
        if top_k_ratio is None:
            return self
        self.config.top_k_ratio = top_k_ratio
        return self

    def with_time_threshold(self, time_threshold):
        if time_threshold is None:
            return self
        self.config.time_threshold = time_threshold
        return self

    def build(self):
        return self.config
