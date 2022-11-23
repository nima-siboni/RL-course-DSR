# The code structure:
# 1. Initialization
##  1.1 inputs and the random seed
## 1.2  setting the geometry
## 1.3 create the environment
## 1.4 create a random policy
## 1.5 some important paramters
## 1.6 setting up the plot
## 1.7 create an array for all states


# 2. Evaluate the current policy
## 2.1 a sweep over all the states in the system.
### 2.1.1 for each state, restart the episode
### 2.1.2 run the simulation (following pi) and collect all the rewards

# 3. Policy Improvement
## 3.1 let's first find out the best action for each point
### 3.1.1 calculate the Qs
## 3.2 convert the found best actions to a soft but greedy policy
