# Investigate an environment together
The goal of this practice is to:
* get familiar with creating an enviornment
* checking some properties of the environment, e.g. observation_space, action_space, 
  etc.
* interact with the environment, e.g. reset it, pass actions to it, get feedback from 
  it, etc.

## 0 - Create an environment
```python
import gym
env = gym.make("CartPole-v1")
```
## 1 - Let's examine the env
### 1.1 - Check the observation space of the agent
The observation space is the space of all possible states that the agent can be in.
### 1.2 - Check the action space of the agent
The action space is the space of all possible actions that the agent can take.
### 1.3 - Render the env
### 1.4 - Reset the environment
### 1.5 - Check the source code of the env.

## 2 - Interact with environment
### 2.1 - Push the cart to the left once
### 2.2 - Repeat the action above until the pole falls, i.e. the episode is terminated

Get started in [template python script](./playground.py) or on [Google Colab](https://colab.research.google.com/drive/1KW9hrQ6CapTs8DPv8QYHlnk22WRWQoMj?authuser=1#scrollTo=bKrYVy3Ab2T9)