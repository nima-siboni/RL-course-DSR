import numpy as np
import matplotlib
import matplotlib.pyplot as plt
np.random.seed(0)

def return_pointwise_policy(i, j):
    A = np.array([[0, 0]])
    
    if (0 <= i and i <= 7):
        A = np.append(A, np.array([[1, 0]]), axis = 0)
    if (1 <= i and i <= 8):
        A = np.append(A, np.array([[-1, 0]]), axis = 0)

    if (0 <= j and j <= 7):
        A = np.append(A, np.array([[0, 1]]), axis = 0)
    if (1 <= j and j <= 8):
        A = np.append(A, np.array([[0, -1]]), axis = 0)

    if (i == 3):
        A = np.append(A, np.array([[6, 0]]), axis = 0)
    if (i == 9):
        A = np.append(A, np.array([[-6, 0]]), axis = 0)
    if (j == 3):
        A = np.append(A, np.array([[0, 6]]), axis = 0)
    if (j == 9):
        A = np.append(A, np.array([[0, -6]]), axis = 0)
        
    return A

def one_step(i, j, pi):
    A = return_pointwise_policy(i, j)
    chosen_action_id = pi[i, j]
    action = A[chosen_action_id]
    iprime, jprime = np.array([i, j]) + action
    if iprime != jprime : # if the action is allowed; this part can be modified 
        action_allowed = True
    else:
        action_allowed = False
        iprime = i
        jprime = j
    return iprime, jprime

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
plt.scatter(3, 1, s=750, c='white', marker='s',linewidths=1, edgecolor='black' )
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
    iprime, jprime = one_step(i, j, pi)
    ix = i
    jx = j
    if (i == 9):
        ix = 3
        iy = 1
    if (j == 9):
        jx = 3
        jy = 1
    plt.scatter(ix, iy, s=100, c='#C1C7C9')
    plt.scatter(jx, jy, s=100, c='#C1C7C9')

    ixprime = iprime
    jxprime = jprime
    if (iprime == 9):
        ixprime = 3
        iyprime = 1
    if (jprime == 9):
        jxprime = 3
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
