from config.Config import Config
from dataSet.CiCycle import CycleTestCases
from select.env.PointWiseEnv import CIPointWiseEnv
from select.env.envFactory.AbstractEnvironment import AbstractEnvironment


class PointwiseEnvFactory(AbstractEnvironment):

    def create_environment(self, config: Config, cycle_test_cases: CycleTestCases):
        return CIPointWiseEnv(config, cycle_test_cases)

