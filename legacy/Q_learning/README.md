# Q-learning
Congratulations! You made it to Q-learning. In this exercise you are going to 
implement the core part of the Q-learning algorithm.

The main idea is to start with random Q values (or set them to zero), and update the 
values based on the experiences. In particular, you are going to update the values 
using information you gather from each step:
* s: current state
* a: the action taken from s,
* r(s, a): the reward received from the environment by taking action a from state s,
* terminated: whether the episode is terminated or not.

The alogrithm looks like:
```python
assign random values to Qs

for _ in training_rounds:

    create an epsilon greedy policy from Q values

    start an episode

    while episode is not terminated:
         follow the policy and create (s,a,r,s')
         update the Q(s, a)
    update epsilon
```

## Exercise 1
Write a function that updates the Q values based on the materials discussed during 
the course. There are hints for you in the comments of experience_and_learn.py

## Exercise 2
At the beginning of the training for loop a new policy is created from the newly 
updated Q values. What do you think would happen if you do not update the policy? 
Let's say instead of updating the policy, you update the policy EVERYTIME to a 
different 
random 
policy.