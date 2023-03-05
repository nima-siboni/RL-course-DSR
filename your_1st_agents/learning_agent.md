# The Learning Agent
Here, step through the process of creating a learning agent, training it, and 
evaluting the trained agent.

## 0 - Choose an algorithm
This might sound surprising at this stage but you should choose the learning 
algorithm for the agent. 

Some well-known and well-functioning algorithms are implemented in 
[Ray RLlib repo](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html). Here we 
choose DQN, but any algorithm which fits our problem could be chosen. The equivalent 
choice for supervised learning is (roughly) the choice of the optimizer.

The algorithm belabela could be imported like 
```
import ray.rllib.algorithms.belabela as belabela
```

## 1 - Import ray and initialize it 
```ray``` is a platform which runs behind the scene and takes care of resources etc 
for your training. One should initialize ray before anything happening. 

## 2 - Configure the belabela algorithm
Now let's get a bit deeper. The algorithm that we imported from RLlib has some 
hyperparameter which need tuning :D Let's have a look together into this.
### 2.1 - Get the default config of your algorithm
RLlib offers a default hyperparameter set (aka config) for each of its algorithms! Easy!

 Let's first get that default config and use it as a starting point. The default 
 config is usually in ```belabela.DEFAULT_CONFIG```
### 2.2 - Examine the config and modify it
* How many entries does the config have?
* Which are familiar to you?
# 3 - Create an agent
Now we go on and create an agent with the aforementioned config; Although we talked 
a lot about separation of env and agent, we still pass the env to 
the agent! Why do you think this is so?

```agent = belabela.BeleBela(config=config, env="CartPole-v1")```

# 4 - Train the agent
Long live RLlib! This is done simply by 

```agent.train()```

What does this function return? Have a look inside the returned object and see if 
you find familiar concepts.

# 5 - Run many training steps
Uebung macht den Meister!
# 6 - Visualize the trained agent
This is similar to running the random_agent, except that this time we have a trained 
agent to celebrate! Can you tell, what are the differences in practice? Have a look 
at the random agent and make a guess which parts should be adopted!
# 6.1 - Create the an environment
So far very similar the random agent! The testing env should be similar to the 
training env.
# 6.2. Let the agent choose an action;
This is the part which is different from the random agent!
# 6.3. Pass the action to the environment
Similar to the random agent.
# 6.4. What were you reward?
Similar!
# 6.5. Repeat 6.{2,3, 4} 
Similar!
# 6.6. How much total reward you got? 
Similar!
print("Good-bye.")
