import abc
import math
from collections import deque
import Sensor as Sensor
import numpy as np
import utils as utils
from scipy.stats import norm
import numpy.random as random
import matplotlib.pyplot as plt
import seaborn as sns


def clamp(x, a, b):
    y = a if x < a else x
    y = b if y > b else y
    return y


class Configuration:

    def __init__(self, X_SPEED, height, ACC_BOUND=10):
        self.position = (0, 0)
        self.height = height
        self.x = 0
        self.y = 0
        self.direction = None
        self.speed = 0
        self.ACC_BOUND = ACC_BOUND
        self.X_SPEED = X_SPEED

    def set(self, x, y):
        self.position = (x, y)
        self.x = x
        self.y = y

    def copy(self):
        copy = Configuration(self.X_SPEED, self.ACC_BOUND)
        copy.x = self.x
        copy.y = self.y
        copy.position = self.position
        copy.speed = self.speed
        return copy

    def update(self, acc, clamping=True):
        assert(abs(acc) < self.ACC_BOUND)
        '''self.speed = self.speed + acc
        y_p = round(self.position[1] + self.speed)
        # clamping works but can do better with get_legal_actions()
        if clamping:
            y_p = clamp(y_p, 0, self.height-1)
        self.position = (0, y_p)
        self.y = self.position[1]'''
        y_p = self.y + acc
        self.position = (0, y_p)
        self.y = y_p


class Grid:
    '''
    2D discretized grid of the world
    '''

    def __init__(self, width, height, initialValue=False):
        self.width = width
        self.height = height
        self.data = [deque([initialValue for x in range(width)]) for y in range(height)]

    def copy(self):
        copy = Grid(self.width, self.height)
        for x in range(self.width):
            for y in range(self.height):
                copy.data[y][x] = self.data[y][x]
        return copy

    def update(self, newObject= False, height=25):
        for y in range(self.height):
            self.data[y].rotate(-1)
            self.data[y][self.width-1] = False

        if newObject:
            for y in range(height):
                self.data[y][self.width-1] = True

    def object_positions(self):
        list = []
        for x in range(self.width):
            for y in range(self.height):
                if self.data[y][x]: list.append((x, y))
        return list

    def __str__(self):
        str = " GRID WORD \n"
        for y in range(self.height):
            str = str + "| " + self.data[y].__str__() + " |\n"

        return str

    def to_nparray(self):
        array = np.zeros((self.width, self.height))
        for x in range(self.width):
            for y in range(self.height):
                array[x, y] = self.data[y][x]
        return array


    def save_heatmap(self, mask=None, lower=-2, upper=2):
        sns.set()
        image = np.zeros((self.width, self.height), dtype=np.float64)
        for w in range(self.width):
            for h in range(self.height):
                image[w, h] = float(-2) if self.data[h][w] == 'x' else float(self.data[h][w])

        #image = (image - lower) / (upper - lower)
        if mask is not None:
            mask = np.transpose(mask)
        ax = sns.heatmap(np.transpose(image), linewidths=0.5, annot=True, mask=mask, square=True)
        ax.invert_yaxis()
        plt.show()


class ModelState:
    __metaclass__ = abc.ABCMeta
    '''
    Class to store my states
    '''

    def __init__(self, width, height, X_SPEED, episode_length=None, print_values=False):
        self.print_values=print_values
        self.grid = Grid(width, height)
        self.isStart = True
        self.isOver = False
        self.time = 0
        self.X_SPEED = X_SPEED
        self.config = Configuration(X_SPEED, height)
        self.observation = None
        self.episode_length = episode_length

    def reset(self):
        self.grid = Grid(self.grid.width, self.grid.height)
        self.config = Configuration(self.X_SPEED, self.grid.height)
        self.isStart = True
        self.isOver = False
        self.time = 0
        self.observation = None

    def get_height(self):
        return self.grid.height

    def get_width(self):
        return self.grid.width

    def has_collied(self):
        return self.grid.data[self.config.y][self.config.x]

    def get_legal_actions(self, lower, upper):
        assert(lower <= upper)
        if self.isOver:
            return [None]
        for a in range(lower, upper+1):
            conf = self.config.copy()
            conf.update(a, clamping=False)
            if 0 <= conf.y < self.grid.height:
                lower = a
                break
        for a in range(upper, lower-1, -1):
            conf = self.config.copy()
            conf.update(a, clamping=False)
            if 0 <= conf.y < self.grid.height:
                upper = a
                break
        return range(lower, upper+1)

    # TODO: keep this method?
    @abc.abstractmethod
    def get_successor(self, action):
        pass

    def update(self, acc=0, new_object=False):
        self.time += 1
        self.grid.update(new_object, 5)
        self.config.update(acc)
        self.isStart = False
        self.observation = None
        if self.grid.data[self.config.y][self.config.x]:
            self.isOver = True
        # No more objects
        if self.episode_length:
            if self.time > self.episode_length:
                self.isOver = True
        else:
            if not self.grid.object_positions():
                self.isOver = True

    @staticmethod
    def n_update(m, acc=0, n=1, new_object=False):
        for i in range(n):
            m.update(acc=acc, new_object=new_object)

    def object_within(self, epsilon=2,  distance_fct="square"):
        if distance_fct == "square":
            for i in range(self.config.x - epsilon, self.config.x + epsilon+1):
                for j in range(self.config.y - epsilon, self.config.y + epsilon+1):
                    x_c = clamp(i, 0, self.grid.width-1)
                    y_c = clamp(j, 0, self.grid.height-1)
                    if self.grid.data[y_c][x_c]:
                        return True
            return False
        else:
            raise Exception("Distance metric unknown")

    def observe(self):
        if self.observation is None:
            self.observation = self.sense()
            return self.observation
        else:
            raise AssertionError("One should observe only once in a State. Use getObservation()")

    @abc.abstractmethod
    def sense(self):
        pass

    @abc.abstractmethod
    def act_upon(self, policy, new_object=False):
        pass

    def __str__(self):
        if self.print_values:
            str = " GRID WORD \n"
            g_copy = self.grid.copy()
            # g_copy.data[self.config.y][0] = '->'
            for y in range(self.grid.height - 1, -1, -1):
                str = str + "| " + list(g_copy.data[y]).__str__() + " |\n"
            return str
        else:
            str = " GRID WORD \n"
            for y in range(self.grid.height-1, -1, -1):
                str = str + "| " + list(map(lambda b: "X" if b else "_", self.grid.data[y])).__str__() + " |\n"
            return str

    def get_observation(self):
        return self.observation


class DeterministicModel(ModelState):
    def __init__(self, width, height, X_SPEED, sensor, episode_length=None, print_values=False, estimator=None):
        super(DeterministicModel, self).__init__(width, height, X_SPEED, episode_length, print_values)
        self.sensor = sensor
        self.estimator = estimator
        self.n = sensor.n

    def sense(self):
        if not self.estimator:
            sensing = map(lambda x: -1 if x >= self.get_width() else round(x), list(self.sensor.sense(self)))
            return tuple(sensing)
        else:
            return self.sensor.sense(self)

    def get_successor(self, action):
        next_state = self.copy()
        next_state.update(action)
        return next_state

    def copy(self):
        copy = DeterministicModel(
            self.grid.width,
            self.grid.height,
            self.X_SPEED,
            self.sensor
        )

        copy.grid = self.grid.copy()
        copy.config = self.config.copy()
        return copy

    def get_state(self):
        return tuple(self.observation), self.config.y

    def _grid_from_policy(self, policy):
        action_grid = self.grid.copy()
        copy = self.copy()
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                copy.config.set(x, y)
                # Dirty but enforces new observation
                copy.observation = None
                copy.observe()
                try:
                    action_grid.data[y][x] = policy.get_action(copy.get_state())
                except KeyError:
                    action_grid.data[y][x] = "x"
        copy.grid = action_grid
        copy.print_values = True
        return copy

    def _grid_from_cost(self, cost):
        cost_grid = self.grid.copy()
        copy = self.copy()
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                copy.config.set(x, y)
                try:
                    cost_grid.data[y][x] = cost(copy)
                except KeyError:
                    cost_grid.data[y][x] = "x"
        copy.grid = cost_grid
        copy.print_values = True
        return copy

    def print_policy(self, policy):
        print(self._grid_from_policy(policy))

    def plot_policy(self, policy):
        self._grid_from_policy(policy).grid.save_heatmap(self.grid.to_nparray())

    def plot_cost(self, cost):
        self._grid_from_cost(cost).grid.save_heatmap()

    def act_upon(self, policy, new_object=False):
        if not self.observation:
            self.observe()
        if self.estimator == "avg":
            self.avg()
        elif self.estimator == "min":
            self.min()

        try:
            act = policy.get_action(self.get_state())
        except KeyError:
            print("key error with key : " + str(self.get_state()))
            act = random.choice(self.get_legal_actions(lower=-2, upper=2))
        self.update(acc=act, new_object=new_object)

    ##############
    # ESTIMATORS #
    ##############
    def avg(self):
        s = 0
        n = 0
        avg = 1
        for i in range(self.n):
            obs = self.get_observation()[i]
            if obs >= 0:
                n += 1
                s += obs
        if n > 0:
            avg = s / n
        for i in range(self.n):
            obs = self.get_observation()[i]
            if obs >= 0:
                self.observation[i] = round(avg)

    def min(self):
        m = float("Inf")
        for i in range(self.n):
            obs = self.get_observation()[i]
            if obs >= 0:
                if obs < m:
                    m = obs
        for i in range(self.n):
            obs = self.get_observation()[i]
            if obs >= 0:
                self.observation[i] = round(m)

class ProbabilisticModel(ModelState):
    def __init__(self, width, height, X_SPEED, sensor,
                 NUM_PARTICLES_PER_ROW=5000, episode_length=None,
                 print_values=False, sigma=2):
        super(ProbabilisticModel, self).__init__(width, height, X_SPEED, episode_length, print_values)
        self.sigma = sigma
        self.sensor = sensor
        self.n = sensor.n
        self.NUM_PARTICLES_PER_ROW = NUM_PARTICLES_PER_ROW
        # initialize
        self.particles = None
        self.estimate = None
        self.init_particles(NUM_PARTICLES_PER_ROW, n=None)
        self.likelihood = {}
        self.track_list = set()

    def init_particles(self, NUM_PARTICLES_PER_ROW, n=None):
        if n is None:
            self.particles = np.zeros((self.get_height(), NUM_PARTICLES_PER_ROW))
        for i in range(NUM_PARTICLES_PER_ROW):
            if n is None:
                for j in range(self.get_height()):
                    self.particles[j, i] = i % self.get_width()
            else:
                self.particles[n, i] = i % self.get_width()

    def update(self, acc=0, new_object=False):
        # make particles move forward
        # X_t+1 = X_t - 1
        for idx in self.track_list:
            for n in range(self.NUM_PARTICLES_PER_ROW):
                self.particles[idx, n] -= self.X_SPEED
                # make sure no negative particles
                if self.particles[idx, n] < 0:
                    self.particles[idx, n] = random.randint(low=0, high=self.get_width())
        super(ProbabilisticModel, self).update(acc, new_object)

    def sense(self):
        return self.sensor.sense(self)

    def observe(self):
        super(ProbabilisticModel, self).observe()
        self.update_particles()

    def update_particles(self):
        if not self.observation:
            print(self.observe())
        # assume Gaussian distribution with parameter self.sigma
        for i in range(- self.n // 2, self.n // 2 + 1):
            idx = self.config.y + i
            if 0 <= idx < self.get_height():
                obs = self.get_observation()[i + self.n // 2]
                if obs < 0:
                    if idx in self.track_list:
                        self.track_list.remove(idx)
                    self.init_particles(self.NUM_PARTICLES_PER_ROW, idx)
                else:
                    self.track_list.add(idx)
                    # p(e_t+1 | X_t+1)
                    for j in range(self.get_width()):
                        self.likelihood[j] = (norm.cdf(math.ceil(obs), loc=j, scale=self.sigma) -
                                              norm.cdf(math.floor(obs), loc=j, scale=self.sigma)) / (
                                             1 - norm.cdf(0, loc=j, scale=self.sigma))

                    else:
                        utils.normalize(self.likelihood)
                        distribution = list(map(lambda x: self.likelihood[x], self.particles[idx]))
                        if sum(distribution) == 0:
                            self.init_particles(self.NUM_PARTICLES_PER_ROW, idx)
                        else:
                            distribution = utils.normalize(distribution)
                            self.particles[idx] = np.array(utils.nSample(distribution, self.particles[idx], self.NUM_PARTICLES_PER_ROW))

    def print_particles(self):
        grid = Grid(self.get_width(), self.get_height(), initialValue=0)
        for y in range(self.get_height()):
            for particle in self.particles[y]:
                grid.data[y][int(particle)] += 1
        print(grid)

    def percentile_estimate(self, percentile=70):
        if not self.observation:
            self.observe()

        new_obs = np.repeat([-1], self.n)
        min_perc = float("Inf")
        for i in range(self.n):
            if self.observation[i] < 0:
                new_obs[i] = self.observation[i]
            else:
                perc = np.percentile(self.particles[i], percentile, interpolation='nearest')
                if perc < min_perc:
                    min_perc = perc

        for i in range(self.n):
            if self.observation[i] >= 0:
                new_obs[i] = min_perc
        return tuple(new_obs)

    def act_upon(self, policy, new_object=False):
        new_estimate = self.percentile_estimate()
        try:
            act = policy.get_action((new_estimate, self.config.y))
        except KeyError:
            print("key error with key : " + str(self.get_state()))
            act = random.choice(self.get_legal_actions(lower=-2, upper=2))
        self.update(acc=act, new_object=new_object)

    def get_state(self):
        pass

    def print_policy(self):
        pass

def manhattan(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


def cost(state, distance_fct=manhattan):
    COLLISION_COST = -1000
    PROXIMITY_COST = -100
    # cost per square out of trajectory
    OUT_OF_LINE_COST = -5
    # check for collision
    if state.grid.data[state.config.y][state.config.x]:
        return COLLISION_COST
    elif state.object_within():
        return PROXIMITY_COST
    else:
        return state.config.y * OUT_OF_LINE_COST

if __name__ == "__main__":
    model = ProbabilisticModel(10, 10, X_SPEED=1, sigma=2, sensor=Sensor.StochasticSensor(n=3, sigma=2), episode_length=20)
    model.update_particles()
    model.print_particles()
    model.update(new_object=True)
    model.update_particles()
    model.print_particles()
    model.update(new_object=False)
    model.update_particles()
    model.update(new_object=False)
    model.update_particles()
    model.update(new_object=True)
    model.update_particles()




    while not model.isOver:
        print("Please select an action among : " + str(model.get_legal_actions(-2, 2)))
        i = input()
        model.update_particles()
        model.update(acc=int(i))
        print(model)
        print(np.average(model.particles, axis=1))

    print("Game is over")

