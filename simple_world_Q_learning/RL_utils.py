import numpy as np

def return_a_random_policy(n, nr_actions):
    '''
    this function returns a random policy

    keywords:
    n -- size of the grid is n x n
    nr_actions --- number of actions from each state

    returns:
    a random policy
    '''
    # lets first create an array of random numbers
    pi = np.random.rand(n, n, nr_actions)
    # now normalize it to make it a probability distribution
    normalization_factor = np.sum(pi, axis=2)
    normalization_factor = np.reshape(normalization_factor, (n, n, 1))
    pi = pi / normalization_factor

    return pi

def choose_an_action_based_on_pi(state, pi):
    '''
    chooses an action from the input state

    keywords:

    state -- the state for which the action should be chosen
    the shape of the state is (1,2)

    pi -- the policy
    '''
    nx, ny = state[0]
    nx = int(nx)
    ny = int(ny)
    policy_for_the_state = pi[nx, ny, :]
    nr_actions = np.size(policy_for_the_state)
    chosen_action = np.random.choice(nr_actions, p=policy_for_the_state)

    return chosen_action


def return_pointwise_A(state):
    '''
    this function returns all the possible actions for agent at state

    keywords:
    state -- position of the agent
    the shape of the state is (1, 2)

    returns:
    a numpy array with all the actions
    '''
    A = np.array([
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0]
    ])

    return A


def is_the_new_state_allowed(state, n):
    '''
    checks if the state is allowed:
    
    the state is not allowed if the agent is steping out the grid
    or on the obstacles.
    
    returns:
    a boolean (res) showing that the state is allowed or not.
    '''

    res = True

    (i, j) = state[0]

    # checking if it is getting out the world.
    if (i < 0 or i == n):
        res = False
    if (j < 0 or j == n):
        res = False

    # setting the obstacles
    if (j == n - 2):
        if (i >= 1 and i <= n - 2):
            res = False
    if (i == n - 2):
        if (j >= 1 and j <= n - 2):
            res = False

    return res

def initialize_the_state(n):
    '''
    randomly initialized the state
    such that the initial state is never the final state!
    '''
    s0 = np.array([[n - 1, n - 1]])
    terminal_state = np.array([[n - 1, n - 1]])
    while np.array_equal(s0, terminal_state):
        s0 = np.random.randint(0, n, size=(1, 2))
    return s0

def step(state, action_id, n):
    '''
    transition_s_to_s_prime
    this function returns the new positions of the agents for:
    action_id in action set A for agents in (i, j)
    and the corresponding reward

    keywords:
    state -- the current state of the agent
    action_id -- the chosen action to be taken from this state
    n -- system size
    
    returns:
    the new_state -- s',
    reward -- i.e. r(s, a),
    terminated -- a flag for termination (the episode is terminted if the flag is True)
    '''
    terminated = False
    (i, j) = state[0]
    A = return_pointwise_A(state)
    action_id = int(action_id)
    action = A[action_id]
    iprime, jprime = np.array([i, j]) + action
    new_state = np.array([[iprime, jprime]])

    if is_the_new_state_allowed(new_state, n):
        reward = -1
        if iprime == n - 1 and jprime == n - 1:
            reward = 0
            terminated = True
    else:
        reward = -5
        new_state = state + 0

    return new_state, reward, terminated


def learn_Q(state, action_id, reward, new_state, Q, gamma, alpha):
    '''
    updates the Q tabel using one s,a,r,s'
    '''
    i, j = state[0].astype(int)
    iprime, jprime = new_state[0].astype(int)

    correction_term = reward + gamma * np.max(Q[iprime, jprime, :]) - Q[i, j, action_id]

    Q[i, j, action_id] = Q[i, j, action_id] + alpha * correction_term

    return Q
    

def return_epsilon_greedy_pi(Q, epsilon):
    '''
    using the Q, this function returns an
    epsilon greedy policy
    '''
    n, _, nr_actions = np.shape(Q)
    #greedy_action_id = np.argmax(Q, axis=2)
    
    pi = np.full((n, n, nr_actions), epsilon / nr_actions)

    # pyhtonic way?!
    for i in range(n):
        for j in range(n):
            Q_max = np.max(Q[i, j])
            all_actions_with_Q_max = []
            for action_id in range(nr_actions):
                if (Q[i, j, action_id] == Q_max):
                    all_actions_with_Q_max.append(action_id)
            greedy_action_id = np.random.choice(np.array(all_actions_with_Q_max))
            pi[i, j, greedy_action_id] += 1. - epsilon

    return pi


def Bellmann_iteration(pi, v, gamma):
    '''
    right-hand side of Eq. (4.5) for
    # policy pi
    # rewards r
    # current values v
    # discount value gamma
    '''
    n, _ = np.shape(v)
    new_v = np.zeros(shape=(n, n))
    for i in range(0, n):
        for j in range(0, n):
            action_id = pi[i, j]
            # this part is assumed to be only dependent the state and the action
            # and not the probablistic outcome of the action
            state = np.array([[i, j]])
            new_state, r, _ = step(state, action_id, n)
            iprime, jprime = new_state[0].astype(int)
            new_v[i, j] = r + gamma * v[iprime, jprime]
    new_v[n - 1, n - 1] = 0.0
    return new_v
