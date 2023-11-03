# The Learning Agent
Here, step through the process of creating a learning agent, training it, and
evaluting the trained agent. The goal of this exercise is to:
* Get familiar with RLlib as an example of RL frameworks used in industry
* Choosing a learning algorithm
* Configuring the algorithm
* Training the agent
* Visualize and evaluate the trained agent

## 0 - Choose an algorithm
This might sound surprising at decision to make at this stage but you should choose the learning
algorithm for the agent.

Some well-known and well-functioning algorithms are implemented in
[Ray RLlib repo](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html). Here we
choose DQN, but any algorithm which fits our problem could be chosen. The equivalent
choice for supervised learning is (roughly) the choice of the optimizer.

In RLlib you first import the config of the algorithm you want to use, and then and
then build the agent from that config.

The algorithm belabela's config could be imported like this:
```
from ray.rllib.algorithms.belabela import belabelaConfig
```
Note: this is very RLlib specific, in other frameworks the import might be different.

## 1 - Build an agent with belabela algorithm
Now let's get a bit deeper. The algorithm that we imported from RLlib has some
hyperparameter which need tuning :D Let's have a look together into this.
### 1.1 - Get the default config of your algorithm
RLlib offers a default hyperparameter set (aka config) for each of its algorithms! Easy!

 Let's first get that default config and use it as a starting point. The default
 config can be accessed by calling the algorithm's config function, e.g.:
```
config = belabelaConfig()
```
### 1.2/3 - Examine the config and modify it
Now we have a config, let's have a look at it. What do you see?
* What is the type of the config?
* How can you access the inside of the config object? (Hint: to_dict() method of the
  config can help you)
* How many entries does the config have?
* Which are familiar to you?
### 1.4 - Introduce the environment to the config
This is a very important step! Here, we specify what our environment is.
```python
config.envirnoment(env="CartPole-v1")
```
Although we talked a lot about separation of env and agent, we still pass the env to
the config of the learning algorithm! Why do you think this is so?
### 1.5
Finally, create an agent from the config :D
```python
agent = config.build()
```
# 2 - Train the agent
Long live RLlib! This is done simply by
```agent.train()```
What does this function return? Have a look inside the returned object and see if
you find familiar concepts.

# 3 - Run many training steps
Uebung macht den Meister!
# 4 - Visualize the trained agent
This is similar to running the random_agent, except that this time we have a trained
agent to celebrate! Can you tell, what are the differences in practice? Have a look
at the random agent and make a guess which parts should be adopted!
# 5.1 - Create the an environment
So far very similar the random agent! The testing env should be similar to the
training env.
# 5.2. Let the agent choose an action;
This is the part which is different from the random agent!
# 5.3. Pass the action to the environment
Similar to the random agent.
# 5.4. What were you reward?
Similar!
# 5.5. Repeat 6.{2,3, 4}
Similar!
# 5.6. How much total reward you got?
Similar!
print("Good-bye.")
