import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def return_pointwise_A(i, j, sim):
    A = np.array([[0, 0]])

    # agent i going to the right and left
    if (0 <= i and i <= 7):
        A = np.append(A, np.array([[1, 0]]), axis = 0)
    if (1 <= i and i <= 8):
        A = np.append(A, np.array([[-1, 0]]), axis = 0)

    # agent i going to the right and left
    if (0 <= j and j <= 7):
        A = np.append(A, np.array([[0, 1]]), axis = 0)
    if (1 <= j and j <= 8):
        A = np.append(A, np.array([[0, -1]]), axis = 0)

    # simultaneous movement of agents
    if (sim == True):
        if (0<=i and i<=7 and 0<=j and j<=7):
            A = np.append(A, np.array([[1, 1]]), axis = 0) 
        if (1<=i and i<=8 and 1<=j and j<=8):
            A = np.append(A, np.array([[-1, -1]]), axis = 0) 
        if (1<=i and i<=8 and 0<=j and j<=7):
            A = np.append(A, np.array([[-1, 1]]), axis = 0) 
        if (0<=i and i<=7 and 1<=j and j<=8):
            A = np.append(A, np.array([[1, -1]]), axis = 0) 
    
    if (i == 2):
        A = np.append(A, np.array([[7, 0]]), axis = 0)
    if (i == 9):
        A = np.append(A, np.array([[-7, 0]]), axis = 0)
    if (j == 2):
        A = np.append(A, np.array([[0, 7]]), axis = 0)
    if (j == 9):
        A = np.append(A, np.array([[0, -7]]), axis = 0)
        
    return A



def transition_s_to_s_prime(action_id, A, i, j):
    nr_actions = np.size(A, 0)
    action = A[action_id]
    iprime, jprime = np.array([i, j]) + action

    # checking for the overalps
    if iprime == jprime : # if the action is allowed; this part can be modified 
        overlap = True
    else:
        overlap = False

    # checking for the swaps
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
        iprime = i
        jprime = j
    else:
        action_allowed = True
        
    return action_allowed, iprime, jprime
        

def Bellmann_iteration(n, pi, r, v, gamma, sim):
    new_v = np.zeros(shape=(n, n)) 
    for i in range(0, n):
        for j in range(0, n):
            A = return_pointwise_A(i, j, sim)
            nr_actions = np.size(A, 0)
            preferred_action_id = pi[i, j]
            new_v[i, j] = r[i, j] # this part is assumed to be only dependent the state and the action, not the probablistic outcome of the action
            for action_id in range(0, nr_actions): #iterate over all possible actions
                action_allowed, iprime, jprime = transition_s_to_s_prime(action_id, A, i, j)
                if (action_id == preferred_action_id):
                    probability = 1.0
                else:
                    probability = 0.0
                new_v[i, j] += gamma * probability * v[iprime, jprime]
                
    return new_v


def Q_estimation_for_state_s(i, j, gamma, r, v, candidate_action_id, sim): #state s is defined by (i, j)
    Q = r[i, j]
    A = return_pointwise_A(i, j, sim)
    nr_actions = np.size(A, 0)
    for k in range(0, nr_actions): #iterate over all possible outcome
        action_allowed, iprime, jprime = transition_s_to_s_prime(k, A, i, j)
        if (k == candidate_action_id):
            probability = 1
        else:
            probability = 0 
        Q += gamma * probability * v[iprime, jprime]

    return Q


