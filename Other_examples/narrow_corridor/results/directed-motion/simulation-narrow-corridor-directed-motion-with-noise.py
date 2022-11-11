import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
from RL_library import return_pointwise_A
np.random.seed(1)

##### plot_two_agents #####
# plots two agents at i, j
# with s = size
# and color 1 and color 2
def plot_two_agents(i, j, zoom, filename1, filename2):
    # ix, and iy are calculated by converting the point i into and x, y pair
    iy = jy = 0 
    ix = i
    jx = j
    if (i == 9):
        ix = 2
        iy = 1
    if (j == 9):
        jx = 2
        jy = 1
    arr_lena = mpimg.imread(filename1)
    imagebox = OffsetImage(arr_lena, zoom=zoom)
    ab = AnnotationBbox(imagebox, (ix, iy), frameon=False)
    ax = plt.axes()
    ax.add_artist(ab)
    arr_lena = mpimg.imread(filename2)
    imagebox = OffsetImage(arr_lena, zoom=zoom)
    ab = AnnotationBbox(imagebox, (jx, jy), frameon=False)
    ax = plt.axes()
    ax.add_artist(ab)

def one_directed_step(i, j, sim):

    if (i==9):
        i_in_the_hole = True
    else:
        i_in_the_hole = False

    if (j==9):
        j_in_the_hole = True
    else:
        j_in_the_hole = False
    
    A = return_pointwise_A(i, j, sim)
    rand = np.random.rand()
    nA = np.size(A, 0)
    progressive_actions = np.array([])
    exist_p_m = exist_p_0 = exist_0_m = False
    for fuck in range(0, nA):
        if (np.array_equal(A[fuck], np.array([1, -1]))):
            exist_p_m = True
        if (np.array_equal(A[fuck], np.array([1, 0]))):
            exist_p_0 = True
        if (np.array_equal(A[fuck], np.array([0, -1]))):
            exist_0_m = True

    # the order of the if statements are important here.    
    if (exist_p_0 == True):
        action = np.array([1, 0])
        if (j_in_the_hole):
            action = np.array([1, -7])
    if (exist_0_m == True):
        action = np.array([0, -1])
        if (i_in_the_hole):
            action = np.array([-7, -1])

    if (exist_p_m == True):
        action = np.array([1, -1])

    if ((exist_p_0 or exist_0_m or exist_p_m)):
        if (rand < 0.2):
            chosen_action_id = np.random.randint(low=0, high=nA)
            action = A[chosen_action_id]
    else:
        chosen_action_id = np.random.randint(low=0, high=nA)
        action = A[chosen_action_id]
        
    print("action")
    print(action)  
    iprime, jprime = np.array([i, j]) + action
    if (iprime == jprime):
        iprime = i
        jprime = j
    print("iprime = "+str(iprime)+"  jprime = "+str(jprime))
    return iprime, jprime

#### The main program #####

#### 1. initialization #####

# 1.1 parameters
n = 10
sim = True # True is simultaneous motion of particles are allowed 

# 1.2 initial position of the agents
i = 0
j = 8


# 1.3 plotting the grid 
plt.close('all')
plt.ion()
ax = plt.axes()
ax.axis('off')
figure = plt.gcf()
figure.set_size_inches(13, 5)
boxsize = 2700
for x in range(1,8):
    plt.scatter(x, 0, s=boxsize, c='white', marker='s',linewidths=1, edgecolor='black' )
plt.scatter(0, 0, s=boxsize, c='white', marker='s',linewidths=2, edgecolor='red' )
plt.scatter(8, 0, s=boxsize, c='white', marker='s',linewidths=2, edgecolor='blue' )
plt.scatter(2, 1, s=boxsize, c='white', marker='s',linewidths=1, edgecolor='black' )
#plt.scatter(i, 0, s=100, c='red')
#plt.scatter(j, 0, s=100, c='blue')
plt.axis([-2, n+1, -2, 3])
plt.axes().set_aspect('equal')
plt.scatter(i, j, c='red')

la_linea_left = './la_linea/la_linea_red_walking_left.png'
la_linea_right = './la_linea/la_linea_blue_walking_right.png'
zoom = 0.11
#### 2. simulation  ####
step = 0

while (i!=8 or j!=0 and step<1):
    # 2.1 one step following the policy
    iprime, jprime = one_directed_step(i, j, sim)
    # 2.2 checking if they have arrived
    # if they arrive they dont leave anymore; not a major intervation
    if (j == 0): 
        jprime = 0
    if (i == 8):
        iprime = 8
    if (jprime == 0): 
        la_linea_left = './la_linea/la_linea_happy_red.png'
    if (iprime == 8):
        la_linea_right = './la_linea/la_linea_happy_blue.png'

    # 2.2 checking if they have arrived
    #if they arrive they dont leave anymore; not a major intervation
    if (j == 0): 
        jprime = 0
        la_linea_left = './la_linea/la_linea_happy_red.png'
    if (i == 8):
        iprime = 8
        la_linea_right = './la_linea/la_linea_happy_blue.png'
    # 2.3 plotting
    # plot_two_agents(i, j, 100, '#C1C7C9', '#C1C7C9') # plotting gray traces of the agents
    plot_two_agents(i, j, zoom*1.05, './la_linea/white.png', './la_linea/white.png') # 
    plot_two_agents(iprime, jprime, zoom, la_linea_right, la_linea_left) # plotting the agents in their new states
    plt.pause(0.1)
    # 2.4 updating the position of the agents
    i = iprime
    j = jprime
    if (step < 10):
        filename = 'state_00'+str(step)+'.png'
    elif (step < 100):
        filename = 'state_0'+str(step)+'.png'
    elif (step < 1000):
        filename = 'state_'+str(step)+'.png'
    plt.savefig(filename)
    step += 1
