from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns

def normalize(distribution):
    s = 0
    if isinstance(distribution, dict):
        for elem in distribution.values():
            s += elem
        if s != 0:
            for k in distribution.keys():
                distribution[k] /= s
    else:
        for elem in distribution:
            s += elem
        if s != 0:
            distribution /= s
    return distribution

def nSample(distribution, values, n):
    rand = [random.random() for i in range(n)]
    rand.sort()
    samples = []
    samplePos, distPos, cdf = 0, 0, distribution[0]
    while samplePos < n:
        if rand[samplePos] < cdf:
            samplePos += 1
            samples.append(values[distPos])
        else:
            distPos += 1
            cdf += distribution[distPos]
    return samples

def plot_convergence(array, legend):
    sns.set()
    for c in array:
        plt.plot(c)
    plt.legend(legend, loc='upper right')
    plt.show()