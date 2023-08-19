import numpy as np
np.random.seed(0)
  
def transition_s_to_s_prime(action_id, A, i, j, n):
    nr_actions = np.size(A, 0)
    action = A[action_id]
    iprime, jprime = np.array([i, j]) + action
    if iprime >= 0 and iprime < n and jprime >= 0 and jprime < n : # if the action is allowed; this part can be modified 
        action_allowed = True
    else:
        action_allowed = False
        iprime = i
        jprime = j
        
    return action_allowed, iprime, jprime
        

def Bellmann_iteration(n, pi, r, A, v):
    nr_actions = np.size(A, 0)
    new_v = np.zeros(shape=(n, n)) 
    for i in range(0, n):
        for j in range(0, n):
            preferred_action_id = pi[i, j]
            new_v[i, j] = r[i, j] # this part is assumed to be only dependent the state and the action, not the probablistic outcome of the action
            for action_id in range(0, nr_actions): #iterate over all possible actions
                action_allowed, iprime, jprime = transition_s_to_s_prime(action_id, A, i, j, n)
                if (action_id == preferred_action_id):
                    probability = 0.6
                else:
                    probability = 0.1
                new_v[i, j] += gamma * probability * v[iprime, jprime]
                
    return new_v


def Q_estimation_for_state_s(i, j, A, gamma, v, candidate_action_id): #state s is defined by (i, j)
    Q = r[i, j]
    for k in range(0, nr_actions): #iterate over all possible outcome
        action_allowed, iprime, jprime = transition_s_to_s_prime(k, A, i, j, n)
        if (k == candidate_action_id):
            probability = 0.6
        else:
            probability = 0.1 
        Q += gamma * probability * v[iprime, jprime]

    return Q



n = 10 #grid size along each direction

# all the possible actions; not all of these actions can be assigned to every s
# staying (0) there, (1) right, (2) left, (3) up, (4) down
A = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
np.savetxt('actions.dat', A)
nr_actions = np.size(A, 0)


# Policy $\pi$
pi = np.random.random_integers(low=0, high=4, size=(n, n))

# correcting the policy at boarders
#pi[0,:] = 1
#pi[-1,:] = 2
#pi[:,0] = 3
#pi[:,-1] = 4
#pi[n-1, n-1] = 0
print(pi)
np.savetxt('original_pi.dat', pi)

# values, v
v = np.zeros(shape=(n, n))

# reward
r = np.full((n, n), 0)
r[n-1, n-1] = 1

# discount
gamma = 0.9

# policy evaluation
niteration = 250
for iteration in range(0, niteration):
    v = Bellmann_iteration(n, pi, r, A, v)
print v

# policy improvement
niteration = 50
v = np.zeros(shape=(n, n))
for iteration in range(0, niteration):
    new_pi = np.zeros(shape=(n, n)) 
    # policy evaluation
    v = Bellmann_iteration(n, pi, r, A, v)
    # policy iteration
    for i in range(0, n):
        for j in range(0, n):
            Q_max = -1000
            for candidate_action_id in range(0, nr_actions): #iterate over all candidate to find the largest Q
                Q = Q_estimation_for_state_s(i, j, A, gamma, v, candidate_action_id)
                if Q > Q_max :
                    Q_max = Q
                    new_pi[i, j] = candidate_action_id
    pi = new_pi + 0.0

print(pi)        


np.savetxt('optimal_pi.dat', pi)
