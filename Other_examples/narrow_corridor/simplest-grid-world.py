import numpy as np
np.random.seed(0)

n = 5 #grid size along each direction

# all the possible actions; not all of these actions can be assigned to every s
# staying (0) there, (1) right, (2) left, (3) up, (4) down
A = [np.array([0, 0]), np.array([1, 0]), np.array([-1, 0]), np.array([0, 1]), np.array([0, -1])]
nr_actions = np.size(A, 0)

# Policy $\pi$
pi = np.random.random_integers(low=0, high=4, size=(n, n))

# correcting the policy at boarders
pi[0,:] = 1
pi[-1,:] = 2
pi[:,0] = 3
pi[:,-1] = 4
pi[n-1, n-1] = 0
print(pi)

# values, v
v = np.zeros(shape=(n, n))

# reward
r = np.full((n, n), 0)
r[n-1, n-1] = 1

# discount
gamma = 0.9

# policy evaluation
niteration = 150
for iteration in range(0, niteration):
    new_v = np.zeros(shape=(n, n)) 
    for i in range(0, n):
        for j in range(0, n):
            action_id = pi[i, j]
            action = A[action_id]
            iprime, jprime = np.array([i, j]) + action
            new_v[i, j] = r[i, j] + gamma * v[iprime, jprime]
    v = new_v + 0.0

print v

# policy improvement
niteration = 20
for iteration in range(0, niteration):
    new_v = np.zeros(shape=(n, n)) 
    new_pi = np.zeros(shape=(n, n)) 
    # policy evaluation
    for i in range(0, n):
        for j in range(0, n):
            old_action_id = int(pi[i, j])
            action = A[old_action_id]
            iprime, jprime = np.array([i, j]) + action
            new_v[i, j] = r[i, j] + gamma * v[iprime, jprime]
    v = new_v + 0.0
    print(v)
    # policy iteration
    for i in range(0, n):
        for j in range(0, n):
            Q_max = -1000
            for action_id in range(0, nr_actions): #iterate over all possible actions
                action = A[action_id]
                iprime, jprime = np.array([i, j]) + action
                if iprime >= 0 and iprime < n and jprime >= 0 and jprime < n : # if the action is allowed 
                    Q = r[i, j] + gamma * v[iprime, jprime]
                    if Q >= Q_max :
                        Q_max = Q
                        new_pi[i, j] = action_id
    pi = new_pi + 0.0
    print(pi)        

