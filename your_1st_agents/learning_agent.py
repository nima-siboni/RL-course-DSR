# 0 - Choose an alogrithm from ray.rllib.algorithms,
# e.g. import ray.rllib.algorithms.xxx as xxx

# 1 - Import ray and initialize it with ray.init()

# 2 - Configure the xxx algorithm
# 2.1 - Get the default config of xxx from xxx.DEFAULT_CONFIG.copy()
# 2.2 - Examine the config and modify it if needed, e.g. change the "num_gpus" to 0,
# and the learning_rate to 0.001

# 3 - Create an agent
# 3.1 - Creating the agent wth config and env; xxx.XXX(config=config, env="CartPole-v1")

# 4 - Train the agent, and examine the training reports: report = agent.train()

# 5 - Run a loop for nr_trainings = 50 times agent.train()

# 6 - Visualize the trained agent; This is similar to running the random_agent,
# except that this time we have a trained agent
# 6.1 - Create the an environment similar to the training env.
# 6.2. Let the agent choose an action;
# 6.3. and pass it to the environment
# 6.4. How much reward did you get for that action? Keep the score!
# 6.5. Repeat the 6.{2,3, 4} until the end of the episode
# 6.6. How much total reward you got? What does it mean to have large/small reward?

print("Good-bye.")
