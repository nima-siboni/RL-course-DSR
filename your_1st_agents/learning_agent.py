# 0 - Choose an alogrithm from ray.rllib.algorithms, e.g. ray.rllib.algorithms.xxx as
# xxx
# 1 - Configure the xxx algorithm
# 1.1 - Convert the config to a dict by config.to_dict()
# 1.2 - Examine the config and modify it if needed, e.g. change the "learnign rate"
# to 0.0001, or the framework, or env
# 2 - Create an agent with .build, train it, and examine the training reports report = agent.train()
# 3 - Train the agent,

# 4 - Run a loop for nr_trainings = 50 times agent.train()

# 5 - Visualize the trained agent; This is similar to running the random_agent,
# except that this time we have a trained agent
# 5.1 - Create an environment similar to the training env.
    # 5.2. Let the agent choose an action;
    # 5.3. and pass it to the environment
    # 5.4. How much reward did you get for that action? Keep the score!
    # 5.5. Repeat the 5.{2,3, 4} until the end of the episode
    # visualize the agent
    # continue with the next step without closing the plot

# 5.6. How much total reward you got? What does it mean to have large/small reward?
