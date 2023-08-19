import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt

def return_pointwise_policy(i, j):
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
    if (0<=i and i<=7 and 0<=j and j<=7):
        A = np.append(A, np.array([[1, 1]]), axis = 0) 
    if (1<=i and i<=8 and 1<=j and j<=8):
        A = np.append(A, np.array([[-1, -1]]), axis = 0) 
    if (1<=i and i<=8 and 0<=j and j<=7):
        A = np.append(A, np.array([[-1, 1]]), axis = 0) 
    if (0<=i and i<=7 and 1<=j and j<=8):
        A = np.append(A, np.array([[1, -1]]), axis = 0) 

    
    if (i == 3):
        A = np.append(A, np.array([[6, 0]]), axis = 0)
    if (i == 9):
        A = np.append(A, np.array([[-6, 0]]), axis = 0)
    if (j == 3):
        A = np.append(A, np.array([[0, 6]]), axis = 0)
    if (j == 9):
        A = np.append(A, np.array([[0, -6]]), axis = 0)
        
    return A

def transition_s_to_s_prime(action_id, A, i, j, n):
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

    if (i == 3 and j == 9 and iprime == 9 and jprime == 3):
        swap = True
    if (i == 9 and j == 3 and iprime == 3 and jprime == 9):
        swap = True

    if (swap == True or overlap == True):
        action_allowed = False
        iprime = i
        jprime = j
    else:
        action_allowed = True
        
    return action_allowed, iprime, jprime
        

def Bellmann_iteration(n, pi, r, v):
    new_v = np.zeros(shape=(n, n)) 
    for i in range(0, n):
        for j in range(0, n):
            A = return_pointwise_policy(i, j)
            nr_actions = np.size(A, 0)
            preferred_action_id = pi[i, j]
            new_v[i, j] = r[i, j] # this part is assumed to be only dependent the state and the action, not the probablistic outcome of the action
            for action_id in range(0, nr_actions): #iterate over all possible actions
                action_allowed, iprime, jprime = transition_s_to_s_prime(action_id, A, i, j, n)
                if (action_id == preferred_action_id):
                    probability = 1.0
                else:
                    probability = 0.0
                new_v[i, j] += gamma * probability * v[iprime, jprime]
                
    return new_v


def Q_estimation_for_state_s(i, j, gamma, v, candidate_action_id): #state s is defined by (i, j)
    Q = r[i, j]
    A = return_pointwise_policy(i, j)
    nr_actions = np.size(A, 0)
    for k in range(0, nr_actions): #iterate over all possible outcome
        action_allowed, iprime, jprime = transition_s_to_s_prime(k, A, i, j, n)
        if (k == candidate_action_id):
            probability = 1
        else:
            probability = 0 
        Q += gamma * probability * v[iprime, jprime]

    return Q



n = 10 #grid size along each direction

# Policy $\pi$
pi = np.random.random_integers(low=0, high=4, size=(n, n))
print(pi)
np.savetxt('original_pi.dat', pi)

# values, v
v = np.zeros(shape=(n, n))

# reward
r = np.full((n, n), 0)
r[n-2, 0] = 1

# discount
gamma = 0.99

# policy evaluation
niteration = 0
for iteration in range(0, niteration):
    v = Bellmann_iteration(n, pi, r, v)
print v
print(return_pointwise_policy(4, 5))
#input("Press Enter to continue...")
# policy improvement
niteration = 500
v = np.zeros(shape=(n, n))
step = 0
converged = False
while (step < 10):
    new_pi = np.zeros(shape=(n, n)) 
    # policy evaluation
    v = Bellmann_iteration(n, pi, r, v)
    # policy iteration
    for i in range(0, n):
        for j in range(0, n):
            Q_max = -1000
            A = return_pointwise_policy(i, j)
            nr_actions = np.size(A, 0)
            for candidate_action_id in range(0, nr_actions): #iterate over all candidate to find the largest Q
                Q = Q_estimation_for_state_s(i, j, gamma, v, candidate_action_id)
                if Q >= Q_max :
                    Q_max = Q
                    new_pi[i, j] = candidate_action_id
    converged =  False #np.array_equal(pi, new_pi)
    pi = new_pi + 0.0
    step += 1
    if (step%100 == 1):
        print("#iteration: "+str(step-1))
print(pi)        
print(v)
plt.imshow(v)
plt.show()

np.savetxt('optimal_pi.dat', pi)
