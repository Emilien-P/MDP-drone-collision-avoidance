import Model as md
import Sensor as Sensor
import random as random
import numpy as np
import utils as utils
import sys


class Policy:
    def __init__(self, q_values):
        self.pi = {}
        # Keep only the best action per state
        for (state, action) in q_values.keys():
            if action is not None:
                q = q_values[(state, action)]
                if state in self.pi.keys():
                    if q > self.pi[state][1]:
                        self.pi[state] = (action, q)
                else:
                    self.pi[state] = (action, q)

    def get_action(self, state):
        if state in self.pi.keys():
            return self.pi[state][0]
        else:
            raise KeyError("This state has no q_value")


def exploration_function(q_value, n, N_e=10, upper_bound=0):
    if n < N_e:
        return upper_bound
    else:
        return q_value


def q_learning(
        model,
        lower,
        upper,
        iterations=1000,
        epsilon=0.0,
        gamma=1,
        alpha=0.9,
        return_convergence=False):
    q_values = {}
    visited = {}

    conv = np.zeros((iterations, 1))
    pol = np.zeros((iterations, 1))
    policy_list = []

    def q_value(act):
        next = model.get_successor(act)
        q_max = float("-Inf")
        found = False
        a_space = next.get_legal_actions(lower, upper)
        next.observe()
        for a in a_space:
            if (next.get_state(), a) in q_values.keys():
                found = True
                v = q_values.get((next.get_state(), a), float("-Inf"))
                if v > q_max:
                    q_max = v
        if found:
            return q_max
        else:
            return 0

    def get_n(state, action):
        if (state, action) in visited.keys():
            return visited[(state, action)]
        else:
            return 0

    def inc_n(state, action):
        if (state, action) in visited.keys():
            visited[(state, action)] += 1
        else:
            visited[(state, action)] = 1

    for i in range(iterations):
        sys.stdout.write("\r q_learning %f%%" % (i/iterations))
        sys.stdout.flush()

        model.reset()
        # TEST WITH UNIQUE OBJECT BEHAVIOUR
        # - - - - - - - - - -
        model.update(0, True)
        # - - - - - - - - - -
        keep_going = True
        while keep_going:
            model.observe()
            state = model.get_state()
            if model.isOver:
                q_values[(state, None)] = md.cost(model)
                keep_going = False
            else:
                # find best action for current state using past q-values
                maxi = float("-Inf")
                best_act = 0
                actions_space = model.get_legal_actions(lower, upper)
                for act in actions_space:
                    q = q_values.get((state, act), float("-Inf"))
                    q = exploration_function(q, get_n(state, act))
                    if q > maxi:
                        maxi = q
                        best_act = act

                # add random noise in exploration policy
                if random.random() < 100 * epsilon / max(100, i - 0.3 * iterations):
                    best_act = random.choice(actions_space)

                inc_n(state, best_act)

                # removed * get_n(state, best_act) from formula for now
                q_values[(state,  best_act)] = q_values.get((state, best_act), 0) \
                                    + alpha * (md.cost(model) + gamma * q_value(best_act)
                                               - q_values.get((state, best_act), 0))

                model.update(best_act)

        ###################################
        # compute statistics for analysis #
        ###################################
        if return_convergence:
            copy = model.copy()
            copy.reset()
            copy.update(new_object=True)
            copy.n_update(copy, n=2)
            s = 0
            for val in q_values.values():
                s += val
            conv[i] = s
            policy_list.append(Policy(q_values))
    if return_convergence:
        last_pol = policy_list[iterations - 1]
        for i in range(iterations):
            s = 0
            for key in last_pol.pi.keys():
                try:
                    if last_pol.get_action(key) != policy_list[i].get_action(key):
                        s += 1
                except KeyError:
                    s += 1
            pol[i] = s


    print("\nq_learning done")
    if return_convergence:
        return Policy(q_values), conv, pol
    else:
        return Policy(q_values)

def convergence_test():
    model = md.DeterministicModel(10, 10, 1, sensor=Sensor.DeterministicSensor(3), episode_length=20)
    iterations = 20000
    policy, c1, p1 = q_learning(model, -2, 2, iterations, epsilon=0.2, alpha=.9, return_convergence=True)
    policy, c2, p2 = q_learning(model, -3, 3, iterations, epsilon=0.2, alpha=.9, return_convergence=True)
    policy, c3, p3 = q_learning(model, -6, 6, iterations, epsilon=0.2, alpha=.9, return_convergence=True)
    model = md.DeterministicModel(25, 10, 1, sensor=Sensor.DeterministicSensor(3), episode_length=50)
    iterations = 20000
    policy, c4, p4 = q_learning(model, -2, 2, iterations, epsilon=0.2, alpha=.9, return_convergence=True)
    policy, c5, p5 = q_learning(model, -3, 3, iterations, epsilon=0.2, alpha=.9, return_convergence=True)
    policy, c6, p6 = q_learning(model, -6, 6, iterations, epsilon=0.2, alpha=.9, return_convergence=True)
    model.reset()
    model.print_policy(policy)
    model.update(0, True)
    model.n_update(model, n=5)
    model.print_policy(policy)
    print(model)
    print(policy.pi)
    utils.plot_convergence([c1, c2, c3, c4, c5, c6], ["action space = [-2, 2], state space 10x10",
                                                      "action space = [-3, 3], state space 10x10",
                                                      "action space = [-6, 6], state space 10x10",
                                                       "action space = [-2, 2], state space 25x10",
                                                       "action space = [-3, 3], state space 25x10",
                                                       "action space = [-6, 6], state space 25x10"])
    utils.plot_convergence([p1, p2, p3, p4, p5, p6], ["action space = [-2, 2], state space 10x10",
                                                      "action space = [-3, 3], state space 10x10",
                                                      "action space = [-6, 6], state space 10x10",
                                                       "action space = [-2, 2], state space 25x10",
                                                       "action space = [-3, 3], state space 25x10",
                                                       "action space = [-6, 6], state space 25x10"])


if __name__ == "__main__":
   convergence_test()

