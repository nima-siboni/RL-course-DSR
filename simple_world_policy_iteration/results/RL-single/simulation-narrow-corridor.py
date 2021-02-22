import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from RL_library import return_pointwise_A
np.random.seed(0)


def one_step(i, j, pi, sim):
    A = return_pointwise_A(i, j, sim)
    chosen_action_id = pi[i, j]
    action = A[chosen_action_id]
    iprime, jprime = np.array([i, j]) + action

    # checking for the overalps
    if iprime == jprime : # if the action is allowed; this part can be modified 
        overlap = True
    else:
        overlap = False

    # checking for the swaps
    swap = False #swap is anyways always false when particles can move simultaneously, i.e. when sim == True
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
        
    return iprime, jprime

sim = False # True is simultaneous motion of particles are allowed 
policyfilename = 'optimal_pi.dat'

# reading the policy
pi = np.loadtxt(policyfilename)
pi = np.array(pi, dtype = int)
n, n = np.shape(pi)


# simulation loop
i = 0
j = 8
plt.close('all')
plt.ion()
for x in range(0,9):
    plt.scatter(x, 0, s=750, c='white', marker='s',linewidths=1, edgecolor='black' )
plt.scatter(2, 1, s=750, c='white', marker='s',linewidths=1, edgecolor='black' )
plt.scatter(i, 0, s=100, c='red')
plt.scatter(j, 0, s=100, c='blue')
plt.axis([-2, n+1, -2, 3])
plt.axes().set_aspect('equal')
plt.scatter(i, j, c='red')
step = 0
plt.savefig('state_000.png')
while (i!=8 or j!=0):
    step += 1
    iy = jy = iyprime = jyprime = 0
    iprime, jprime = one_step(i, j, pi, sim)
    if (j == 0):
        jprime = 0
    if (i == 8):
        iprime = 8

    ix = i
    jx = j
    if (i == 9):
        ix = 2
        iy = 1
    if (j == 9):
        jx = 2
        jy = 1
    plt.scatter(ix, iy, s=100, c='#C1C7C9')
    plt.scatter(jx, jy, s=100, c='#C1C7C9')

    ixprime = iprime
    jxprime = jprime
    if (iprime == 9):
        ixprime = 2
        iyprime = 1
    if (jprime == 9):
        jxprime = 2
        jyprime = 1
    plt.axis([-2, n+1, -2, 3])
    plt.scatter(ixprime, iyprime, s=100, c='red')
    plt.scatter(jxprime, jyprime, s=100, c='blue')

    
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
