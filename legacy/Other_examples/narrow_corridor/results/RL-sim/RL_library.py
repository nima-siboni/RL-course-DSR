import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#####  return_pointwise_A  #####
# this function returns all the possible actions for
# i,j : the agent one is at point i and the agent two is at point j
# sim : given that sim(ultaneous) moves are allowed or prohibited.
#
# 1. first all the actions which geometry allows are included in A
# 2. then the actions which are not allowed due to the multi agent nature of the problem are excluded.

def return_pointwise_A(i, j, sim):
    
    #### 1. including all the actions which are allowed by the geometry ####
    A = np.array([[0, 0]])

    # 1.1 agent i going to the right and left
    if (0 <= i and i <= 7):
        A = np.append(A, np.array([[1, 0]]), axis = 0)
    if (1 <= i and i <= 8):
        A = np.append(A, np.array([[-1, 0]]), axis = 0)

    # 1.2 agent j going to the right and left
    if (0 <= j and j <= 7):
        A = np.append(A, np.array([[0, 1]]), axis = 0)
    if (1 <= j and j <= 8):
        A = np.append(A, np.array([[0, -1]]), axis = 0)

    # 1.3 simultaneous movement of agents
    if (sim == True):
        if (0<=i and i<=7 and 0<=j and j<=7):
            A = np.append(A, np.array([[1, 1]]), axis = 0) 
        if (1<=i and i<=8 and 1<=j and j<=8):
            A = np.append(A, np.array([[-1, -1]]), axis = 0) 
        if (1<=i and i<=8 and 0<=j and j<=7):
            A = np.append(A, np.array([[-1, 1]]), axis = 0) 
        if (0<=i and i<=7 and 1<=j and j<=8):
            A = np.append(A, np.array([[1, -1]]), axis = 0)
            
    # 1.4 moving to the wider area of the corridor, i.e. from point 2 to 9
    if (i == 2):
        A = np.append(A, np.array([[7, 0]]), axis = 0)
    if (i == 9):
        A = np.append(A, np.array([[-7, 0]]), axis = 0)
    if (j == 2):
        A = np.append(A, np.array([[0, 7]]), axis = 0)
    if (j == 9):
        A = np.append(A, np.array([[0, -7]]), axis = 0)

    #### 2. excluding all the actions which are prohibited due to multi-agent nature of the problem ####
    
    # up to here all the possible moves that the geometry allows are considered in A
    # now removing the actions which lead to ovelapping or swaping cases
    
    nr_actions = np.size(A, 0)
    a = np.array([A[0]]) #the first action is always allowed! which is staying where you are
    for action_id in range(1, nr_actions): #the id starts from 1 (because of the above consideration)
        action = A[action_id]
        iprime, jprime = np.array([i, j]) + action

        # 2.1 checking for the overalps
        if iprime == jprime : # if the action is allowed; this part can be modified 
            overlap = True
        else:
            overlap = False

        # 2.2 checking for the swaps
        swap = False
        if (i == j - 1 and jprime == iprime - 1):
            swap = True
        if (i == j + 1 and jprime == iprime + 1):
            swap = True

        if (i == 2 and j == 9 and iprime == 9 and jprime == 2):
            swap = True
        if (i == 9 and j == 2 and iprime == 2 and jprime == 9):
            swap = True

        if (swap == True or overlap == True):
            action_allowed = False
        else:
            action_allowed = True

        if (action_allowed):
            a = np.append(a, np.array([A[action_id]]), axis = 0)
            
    return a

##### transition_s_to_s_prime #####
# this function returns the new positions of the agents for:
# action_id in
# action set A for agents in 
# (i, j) 

def transition_s_to_s_prime(action_id, A, i, j):
    nr_actions = np.size(A, 0)
    action = A[action_id]
    iprime, jprime = np.array([i, j]) + action
    return iprime, jprime

#####  Bellmann_iteration  #####
# right-hand side of Eq. (4.5) for:
# policy pi
# rewards r
# current values v
# discount value gamma
# and possibility of simu(ultaneous) agent moves

def Bellmann_iteration(pi, r, v, gamma, sim):
    n, n = np.shape(v)
    new_v = np.zeros(shape=(n, n)) 
    for i in range(0, n):
        for j in range(0, n):
            A = return_pointwise_A(i, j, sim)
            nr_actions = np.size(A, 0)
            preferred_action_id = pi[i, j]
            new_v[i, j] = r[i, j] # this part is assumed to be only dependent the state and the action, not the probablistic outcome of the action
            for action_id in range(0, nr_actions): #iterate over all possible actions
                iprime, jprime = transition_s_to_s_prime(action_id, A, i, j)
                if (action_id == preferred_action_id):
                    probability = 1.0
                else:
                    probability = 0.0
                new_v[i, j] += gamma * probability * v[iprime, jprime]
                
    return new_v


#####  Q_estimation_for_state_s  #####
# returning q defined in Eq. (4.6) of the book
# at state (i,j) with
# discount value gamma for 
# action id of candidate_action_id (which is converted to an action in A(i, j) within the function
# sim: refer to the above function

def Q_estimation_for_state_s(i, j, gamma, r, v, candidate_action_id, sim): #state s is defined by (i, j)
    Q = r[i, j]
    A = return_pointwise_A(i, j, sim)
    nr_actions = np.size(A, 0)
    for k in range(0, nr_actions): #iterate over all possible outcome
        iprime, jprime = transition_s_to_s_prime(k, A, i, j)
        if (k == candidate_action_id):
            probability = 1
        else:
            probability = 0 
        Q += gamma * probability * v[iprime, jprime]

    return Q


