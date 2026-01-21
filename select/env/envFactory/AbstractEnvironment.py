from abc import ABC, abstractmethod

from config.Config import Config
from dataSet.CiCycle import CycleTestCases


class AbstractEnvironment(ABC):
    @abstractmethod
    def create_environment(self, config: Config, cycle_test_cases: CycleTestCases):
        pass



