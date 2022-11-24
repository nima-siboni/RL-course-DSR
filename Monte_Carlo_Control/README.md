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
        Calculate the Q values using V
        Find the greedy action from the Q values

    Update the epsilon
    Using the greedy actions find the epsilon soft policy
    update the policy to the new epsilon policy
```