import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from RL_library import return_pointwise_A
from RL_library import Bellmann_iteration
from RL_library import Q_estimation_for_state_s

# In this code the optimal policy is found for:
# two agents who want to go to their desired places which are at the two ends of narrow corridor.
# due to the confinement, the agents can not pass each other and they have to find a collaborative solution.

# Dynamic programming is used find the optimal values and policy with the notations in chapter 4 of THE book.
# The book: Reinforcement learning (introduction) by Sutton & Batto, second edition.
# the program is structured as:
# 1. initialization
# 2. policy evaluation
# 3. policy improvement
# 4. printing and saving

##### 1. Initialization #####

# 1.1 inputs
n = 10 #grid size along each direction
sim = True #if True particles can move simultaneously

# 1.2 policy $\pi$
# setting policy to a random policy
pi = np.random.random_integers(low=0, high=4, size=(n, n))
print(pi)
np.savetxt('original_pi.dat', pi)

# 1.3 values, v
# setting all to zero
v = np.zeros(shape=(n, n))

# 1.4 reward
# all points except (n-2, 0) have rewards zero
r = np.full((n, n), 0)
r[n-2, n-4] = 1

# 1.5 discount
# should be set to a value smaller than 1
gamma = 0.99


#### 2. policy evaluation ####
# Eq. (4.5) of the book.

# this part is only executed if you want to find out what are values of the random initial policies
niteration = 0
for iteration in range(0, niteration):
    v = Bellmann_iteration(pi, r, v, gamma, sim)

#### 3. policy improvement ####

# 3.1 number of iterations for the policy improvement
# this can be replaced by a measure of convergence
niteration = 500

# 3.2 initializing values
# values are set to zero.
v = np.zeros(shape=(n, n))

# 3.3 the main iterative loop
# Eq. (4.7) of the book:
# each step is the Bellman operation for policy evaluation
# followed by a policy improvement 
step = 0
while (step < 500):
    new_pi = np.zeros(shape=(n, n)) 
    # policy evaluation
    v = Bellmann_iteration(pi, r, v, gamma, sim)
    # policy iteration
    for i in range(0, n):
        for j in range(0, n):
            Q_max = -1000
            A = return_pointwise_A(i, j, sim)
            nr_actions = np.size(A, 0)
            for candidate_action_id in range(0, nr_actions): #iterate over all candidate to find the largest Q
                Q = Q_estimation_for_state_s(i, j, gamma, r, v, candidate_action_id, sim)
                if Q >= Q_max :
                    Q_max = Q
                    new_pi[i, j] = candidate_action_id
    pi = new_pi + 0.0
    step += 1
    if (step%100 == 1):
        print("#iteration: "+str(step-1))

#### 4. printing and saving ####

print(pi)        
print(v)
plt.imshow(v)
plt.show()
np.savetxt('optimal_pi.dat', pi)
