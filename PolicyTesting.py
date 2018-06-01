import Model as md
from scipy.stats import poisson
from numpy import random
import numpy as np
from Learning import *
import sys

def simulate(
        model,
        policy,
        iterations=1000,
        lbda=.05
):
    n_collisions = 0
    avg_time_in_risk = 0
    avg_cost = 0
    for i in range(iterations):
        sys.stdout.write("\r%d%%" % i)
        sys.stdout.flush()
        time_in_risk = 0
        model.reset()
        model.update(new_object=True)
        avg_cost += md.cost(model)
        while not model.isOver:
            model.act_upon(policy)
            avg_cost += md.cost(model)
            if model.object_within():
                time_in_risk += 1
        if model.has_collied():
            n_collisions += 1

        time_in_risk / model.time
        avg_time_in_risk += time_in_risk
    avg_time_in_risk /= iterations
    avg_cost /= iterations
    return n_collisions, avg_time_in_risk, avg_cost

def opt_policy():
    model = md.DeterministicModel(10, 10, 1, sensor=Sensor.DeterministicSensor(3), episode_length=20)
    model2 = md.DeterministicModel(10, 10, 1, sensor=Sensor.StochasticSensor(3), episode_length=20)
    policy = q_learning(model, -2, 2, 50000, epsilon=0.2, alpha=.9)
    model.update(new_object=True)
    model.n_update(model, n=3)
    model2.update(new_object=True)
    model2.n_update(model2, n=3)
    model.plot_policy(policy)
    model2.plot_policy(policy)
    print(simulate(model2, policy, iterations=100))

def plot_models(sigmas, n):
    cost = np.zeros((4, len(sigmas)))
    fail = np.zeros((4, len(sigmas)))
    time = np.zeros((4, len(sigmas)))

    for idx, s in enumerate(sigmas):
        deterministic_model = md.DeterministicModel(10, 10, 1, sensor=Sensor.DeterministicSensor(n), episode_length=25)
        deter_stoch_model = md.DeterministicModel(10, 10, 1, sensor=Sensor.StochasticSensor(n=n, sigma=s), episode_length=25)

        policy_deter = q_learning(deterministic_model, -2, 2, 10000, epsilon=0.2, alpha=.9)
        policy_stoch = q_learning(deterministic_model, -2, 2, 10000, epsilon=0.2, alpha=.9)

        deter_avg_model = md.DeterministicModel(10, 10, 1, sensor=Sensor.StochasticSensor(n=n,sigma=s), episode_length=25, estimator="avg")
        deter_min_model = md.DeterministicModel(10, 10, 1, sensor=Sensor.StochasticSensor(n=n,sigma=s), episode_length=25, estimator="min")

        prob_model = md.ProbabilisticModel(10, 10, 1, sensor=Sensor.StochasticSensor(n=n, sigma=s), episode_length=25)

        iterations = 1000

        fail[0, idx], time[0, idx], cost[0, idx] = simulate(deter_stoch_model, policy_stoch, iterations)
        fail[1, idx], time[1, idx], cost[1, idx] = simulate(deter_avg_model, policy_deter, iterations)
        fail[2, idx], time[2, idx], cost[2, idx] = simulate(deter_min_model, policy_deter, iterations)
        fail[3, idx], time[3, idx], cost[3, idx] = simulate(prob_model, policy_deter, iterations)

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set()
    plt.plot(sigmas, fail[0])
    plt.plot(sigmas, fail[1])
    plt.plot(sigmas, fail[2])
    plt.plot(sigmas, fail[3])
    plt.legend(["Deterministic model without estimators", "Deterministic model with avg estimate", "Deterministic model with min estimate", "Particle Filter Stochastic model"])
    plt.show()
    plt.figure(2)
    plt.plot(sigmas, time[0])
    plt.plot(sigmas, time[1])
    plt.plot(sigmas, time[2])
    plt.plot(sigmas, time[3])
    plt.legend(["Deterministic model without estimators", "Deterministic model with avg estimate",
                "Deterministic model with min estimate", "Particle Filter Stochastic model"])
    plt.show()
    plt.figure(3)
    plt.plot(sigmas, cost[0])
    plt.plot(sigmas, cost[1])
    plt.plot(sigmas, cost[2])
    plt.plot(sigmas, cost[3])
    plt.legend(["Deterministic model without estimators", "Deterministic model with avg estimate",
                "Deterministic model with min estimate", "Particle Filter Stochastic model"])
    plt.show()


def simulatation():
    model = md.DeterministicModel(10, 10, 1, sensor=Sensor.DeterministicSensor(3), episode_length=20)
    model2 = md.DeterministicModel(10, 10, 1, sensor=Sensor.StochasticSensor(3), episode_length=25,
                                            estimator="min")
    policy = q_learning(model, -2, 2, 25000, epsilon=0.2, alpha=.9)
    print(simulate(model2, policy, iterations=1000))

if __name__ == "__main__":
    model = md.DeterministicModel(10, 10, 1, sensor=Sensor.DeterministicSensor(3), episode_length=20)
    model.update(new_object=True)
    model.n_update(model, n=3)
    model.plot_cost(md.cost)