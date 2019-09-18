import numpy as np
import matplotlib.pyplot as plt
from discreteMarkovChain import markovChain


# linear congruential generator
def rng(start_n, stop_n, m=2 ** 32, a=1103515245, c=12345):
    for i in range(1, stop_n):
        if i > start_n:
            yield rng.current / m
        rng.current = (a * rng.current + c) % m


def check_state(state, p):
    psum = 0
    for i in range(len(g[state])):
        psum += g[state, i]
        if p <= psum:
            return i


def simulate(state, start_n, stop_n):
    states = [state]
    for p in rng(start_n, stop_n):
        state = check_state(state - 1, p) + 1
        states.append(state)
    return states


# transition matrix
g = np.array([
    [.3, .2, .4, .1],
    [0, .1, .6, .3],
    [0, .4, 0, .6],
    [0, .5, .5, 0]
])

# random seed
rng.current = 1

# simulation
n = 100
state_1 = simulate(1, 0, n)
state_2 = simulate(2, n, 2 * n)
state_3 = simulate(3, 2 * n, 3 * n)
state_4 = simulate(4, 3 * n, 4 * n)

# plot
n = 100
plt.step(range(n), state_4)
plt.xticks(np.arange(0, n, step=10))
plt.ylim(0, 5)
plt.show()

# steady state distribution
mc = markovChain(g)
mc.powerMetho.d(tol=1e-10, maxiter=1e16)
steady_state = np.round(mc.pi, decimals=4)
print(steady_state)

# experimentally computed stationary distribution
n_max = 500000
n = round(n_max / 4)

states = []
states.extend(simulate(1, 0, n))
states.extend(simulate(2, n, 2 * n))
states.extend(simulate(3, 2 * n, 3 * n))
states.extend(simulate(4, 3 * n, 4 * n))


exp_stat_dist = [0] * 4
for s in states:
    exp_stat_dist[s-1] += 1

for i in range(len(exp_stat_dist)):
    value = np.around(exp_stat_dist[i] / (4 * n), decimals=4)
    exp_stat_dist[i] = value

exp_stat_dist = np.array(exp_stat_dist)
print(exp_stat_dist)

delta = np.round(exp_stat_dist - steady_state, decimals=4)
print(delta)

print(sum(np.round(delta, decimals=4)))

print(sum(delta/steady_state))