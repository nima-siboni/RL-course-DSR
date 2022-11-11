import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(0)
def one_step(i, j, pi, A, determinism):
    number_of_repeatition = determinism
    all_actions_with_proper_weight = np.append(np.array([0, 1, 2, 3, 4]), np.full( (1, number_of_repeatition), pi[i, j]))
    rand = np.random.randint(low = 0, high = np.size(all_actions_with_proper_weight))
    chosen_action_id = all_actions_with_proper_weight[rand]
    action = A[chosen_action_id]
    iprime, jprime = np.array([i, j]) + action
    if iprime >= 0 and iprime < n and jprime >= 0 and jprime < n : # if the action is allowed; this part can be modified 
        action_allowed = True
    else:
        action_allowed = False
        iprime = i
        jprime = j
    return iprime, jprime

policyfilename = 'optimal_pi.dat'
determinism = 5 # determinism = 0 means what happens after an action is independent of the action. The training is done for determinism = 5

actionsfilename = 'actions.dat'
#nr_sim_step = 100

# reading the policy
pi = np.loadtxt(policyfilename)
pi = np.array(pi, dtype = int)
n, n = np.shape(pi)

# reading the actions
A = np.loadtxt(actionsfilename)
A = np.array(A, dtype = int)
nr_actions = np.size(A, 0)

# simulation loop
i = j = 0
plt.close('all')
plt.scatter(n-1, n-1, s=100, c='orange', marker='s')
plt.axis([-1, n, -1, n])
plt.axes().set_aspect('equal')
plt.scatter(i, j, c='red')
step = 0
plt.savefig('state_000.png')
while (i!=n-1 or j!=n-1):
    step += 1
    iprime, jprime = one_step(i, j, pi, A, determinism)
    plt.scatter(i, j, s=100, c='#C1C7C9', marker='s')
    plt.scatter(iprime,jprime, s=50, c='red')
    plt.show()
    plt.pause(0.1)
    i = iprime
    j = jprime
    if (step < 10):
        filename = 'state_00'+str(step)+'.png'
    elif (step < 100):
        filename = 'state_0'+str(step)+'.png'
    elif (step < 1000):
        filename = 'state_'+str(step)+'.png'
    plt.savefig(filename)
