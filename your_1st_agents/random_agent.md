# Random Agent
Here you create an env and let an agent interact with it.The goals of this exercise are:
-[ ] repetition of environment creation,
-[ ] Interacting/controlling an environment,
-[ ] Getting familiar with *evaluation* of an agent

## 0 - Create an env.
```
import gym
env = gym.make("CartPole-v1")
```
## 1 - Reset the env.
Every env. should be started after creation and restarted after the episode finished.
These both are done by ```reset()```
## 2 - Let a random agent interact with the env.
Now that we have an up and running environment it is time to interact with it.
### 2.1. Choose a random action (with in the action space of the env.)
Here we just choose a random action. In this case that we have a discrete action
space, choosing an action is just taking a random integer in range [0, nr_actions).
### 2.2. Pass the chose action to the environment
Let's say the chose action is the action number ``a``. Now we want to tell the
environment that this is the action we take, and the environment is going to inform
us about the consequences of this action. Passing the action to environment is done by
method ```step``` implemented in the environment.
### 2.3. How much reward did you get for that action? Keep the score!
When the agent passes an action to the environment, the env. is going to inform the
agent about how much immediate reward it got for that particular action from that
particular state.
### 2.4. Repeat the 2.{1,2,3} until the end of the episode
After that the action is taken the env. is in new a state, from which the agent
should take a new action, and pass that action to the env. This continues until the
environment is terminated.
### 2.5. How much total reward you got?
What does it mean to have large/small reward?

## 3. Repeat the whole section 2.
Do you expect to get the same total reward?
