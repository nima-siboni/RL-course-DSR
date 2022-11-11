import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def return_pointwise_A(i, j):
    '''
    this function returns all the possible actions for agent at (i,j)
    '''
    A = np.array([
        [0, 0],
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0]
    ])
    return A


def is_the_new_state_allowed(i, j, n):
    '''
    checks if the state i, j is allowed
    '''
    res = True

    if (i < 0 or i == n):
        res = False

    if (j < 0 or j == n):
        res = False

    if (j == n - 2):
        if (i > 1 and i < n - 2):
            res = False

    if (i == n - 2):
        if (j > 1 and j < n - 2):
            res = False

    return res


def step(action_id, i, j, n):
    '''
    transition_s_to_s_prime
    this function returns the new positions of the agents for:
    action_id in action set A for agents in (i, j)
    and the corresponding reward
    '''
    A = return_pointwise_A(i, j)
    action_id = int(action_id)
    action = A[action_id]
    iprime, jprime = np.array([i, j]) + action

    if is_the_new_state_allowed(iprime, jprime, n):
        reward = -1
        if iprime == n - 1 and jprime == n - 1:
            reward = 0
    else:
        reward = -5
        iprime = i
        jprime = j

    return iprime, jprime, reward


def Bellmann_iteration(pi, v, gamma):
    '''
    right-hand side of Eq. (4.5) for pi, v, and gamma

    keywords:
    pi -- policy
    v -- current values 
    gamma -- discount value gamma

    returns:
    the left hand side of Eq. (4.5)
    '''
    n, _ = np.shape(v)
    new_v = np.zeros(shape=(n, n))
    for i in range(0, n):
        for j in range(0, n):
            action_id = pi[i, j]
            # this part is assumed to be only dependent the state and the action
            # and not the probablistic outcome of the action
            iprime, jprime, r = step(action_id, i, j, n)
            new_v[i, j] = r + gamma * v[iprime, jprime]
    return new_v


def Q_estimate(i, j, action_id, gamma, v):
    '''
    Q_estimation_for_state_s where the state s is defined by (i, j)
    returning q defined in Eq. (4.6) of the book

    keywords:

    i, j -- the state is s=(i,j)
    gamma -- discount value
    action_id -- the id of the action a in Q(s, a)

    returns:
    Q(s, a)

    '''
    n, _ = np.shape(v)
    iprime, jprime, r = step(action_id, i, j, n)
    Q = r + gamma * v[iprime, jprime]

    return Q


def simulate(i0, j0, pi, color='orange', nr_actions=5, randomize=0):
    '''
    simulates an episode starting from the given initial state
    untill end of the episode

    keywords:

    i0, j0 -- the initial state (i0, j0)
    pi -- the policy to be followed
    color -- the color of the agent shown in animation
    nr_actions -- the number of actions (!)
    randomize -- the probability that the agent doesnt follow the policy

    returns:

    none
    '''

    # show the obstabcles:
    n, _ = np.shape(pi)
    for i in range(n):
        for j in range(n):
            if not is_the_new_state_allowed(i, j, n):
                plt.scatter(i, j, c='black', marker='x')
                plt.draw()
                plt.show()

    i = i0
    j = j0
    while not (i == n - 1 and j == n - 1):
        plt.scatter(i, j, c=color)
        if (np.random.rand() < randomize):
            action_id = np.random.randint(0, nr_actions)
        else:
            action_id = pi[i, j]
        i, j, _ = step(action_id, i, j, n)
        plt.draw()
        plt.show()
        plt.pause(0.5)
