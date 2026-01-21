from select.env.envFactory.ListwiseEnvFactory import ListwiseEnvFactory
from select.env.envFactory.PairwiseEnvFactory import PairwiseEnvFactory
from select.env.envFactory.PointwiseEnvFactory import PointwiseEnvFactory


class FactoryRegistry:
    _factories = {
        'POINTWISE': PointwiseEnvFactory(),
        'PAIRWISE': PairwiseEnvFactory(),
        'LISTWISE': ListwiseEnvFactory()
    }

    @classmethod
    def getFactory(cls, mode):
        factory = cls._factories.get(mode.upper())
        if not factory:
            raise ValueError('Invalid mode: %s' % mode)
        return factory

    @classmethod
    def registerFactory(cls, mode, factory):
        cls._factories[mode.upper()] = factory
