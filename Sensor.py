import abc
from itertools import islice
from numpy import random


class SpecialValues:
    NO_OBJECT_IN_RANGE = -1
    UNABLE_TO_SENSE = -2


class Sensor:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sense(self, model):
        pass


class SimpleSensor(Sensor):
    def sense(self, model):
        try:
            obs = list(islice(model.grid.data[model.config.y], model.config.x, model.grid.width)).index(True)
            return obs
        except ValueError:
            return SpecialValues.NO_OBJECT_IN_RANGE


class DeterministicSensor(Sensor):
    def __init__(self, n=3, sensor_range=None):
        assert(n % 2 == 1)
        self.n = n
        self.sensor_range = sensor_range

    def sense(self, model):
        d = self.n // 2
        obs = [0] * self.n
        upper_bound = model.get_width()
        if self.sensor_range:
            upper_bound = min(model.get_width(), model.config.x+range+1)

        for idx, y in enumerate(range(model.config.y - d, model.config.y + d + 1)):
            if 0 <= y < model.get_height():
                try:
                    obs[idx] = list(islice(model.grid.data[y], model.config.x, upper_bound)).index(True)
                except ValueError:
                    obs[idx] = SpecialValues.NO_OBJECT_IN_RANGE
            else:
                obs[idx] = SpecialValues.UNABLE_TO_SENSE
        return tuple(obs)


class StochasticSensor(DeterministicSensor):
    def __init__(self, sigma=3, n=3, sensor_range=None):
        super(StochasticSensor, self).__init__(n, sensor_range)
        self.sigma = sigma
        self.n = n

    def sense(self, model):
        obs = list(super(StochasticSensor, self).sense(model))
        for i in range(self.n):
            if obs[i] > 0:
                off = random.normal(scale=self.sigma)
                while obs[i] - off < 0:
                    off = random.normal(scale=self.sigma)
                # obs[i] = round(obs[i])
                obs[i] -= off
        return obs
