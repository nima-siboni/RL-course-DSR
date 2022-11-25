# Monte Carlo Control
In this exercise you work on Monte-Carlo control.

The core of this approach is composed of many repetitions of the following steps:
* Evaluating the current scenario, and
* Improving it

In more details the approach is as follows
```python
Create a initial policy

for _ in nr_training_rounds:
    
    V <-- Evaluate the policy for all the points

    for each state:
        Calculate the Q values of this state using V
        Find the greedy action from the Q values

    Update the epsilon
    Using the greedy actions find the epsilon soft policy
    update the policy to the new epsilon policy
```

## Exercise
In control.py there is a TODO, which is about calculating the Q values for each state

Your solution should look like
```python
Q =  calculate_Qs_for_this_state(env=env, state=state, V=V, gamma=gamma,
                                          nr_actions=nr_actions)
```
Before coding, write down the algorithm behind the Q calculation; what are the main 
ingredients of this calculation?

# Bonus Exercise
The epsilon greedy policy is created from the greedy policy. The chosen approach is 
not the standard way of doing it as discussed in the class. Implement the standard 
approach

# Super Bonus Exercise
Think of a different approach to go from Q values to a soft greedy policy. Start by 
arguing whether this is the best use of Q values or not. You can share your 
arguments with me and the class. If you are not finding an argument, let me know. I 
will give you a set of Q values which could be hint.