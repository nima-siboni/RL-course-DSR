# 0 - Choose an alogrithm from ray.rllib.algorithms, e.g. ray.rllib.algorithms.xxx as
# xxx
# 1 - Import ray and initialize it with ray.init()
# 2 - Configure the xxx algorithm
# 2.1 - Get the default config of xxx from xxx.DEFAULT_CONFIG.copy()
# 2.2 - Examine the config and modify it if needed, e.g. change the "num_gpus" to 0,
# and the learning_rate to 0.001
import gymnasium as gym
from your_1st_agents.plot_util import visualize_env

# 0 - Choose an algorithm from ray.rllib.algorithms
from ray.rllib.algorithms.dqn import DQNConfig


# 1- Configure the blah algorithm
# 1.1. get the default config for you algorithm
config = DQNConfig()
# 1.2 - Examine the config and modify it if needed, e.g. change the "learnign rate"
# to 0.0001,
config_as_dict = config.to_dict()

# 2 - Create an agent, and examine the training reports: report = agent.train()
config.environment(env="CartPole-v1")
agent = config.build()
# 3 - Train the agent,
reports = agent.train()
print(reports)

# 4 - Run a loop for nr_trainings = 50 times agent.train()
nr_trainings = 1
for _ in range(nr_trainings):
    reports = agent.train()
    print(_, reports["episode_reward_mean"])

# 5 - Visualize the trained agent; This is similar to running the random_agent,
# except that this time we have a trained agent
# 5.1 - Create an environment similar to the training env.
env = gym.make("CartPole-v1", render_mode="rgb_array")
s, _ = env.reset()
done = False
cumulative_reward = 0

while not done:
    # 5.2. Let the agent choose an action;
    a = agent.compute_single_action(observation=s, explore=False)
    # 5.3. and pass it to the environment
    s, r, terminated, truncated, info = env.step(action=a)

    # 5.4. How much reward did you get for that action? Keep the score!
    cumulative_reward += r
    done = terminated or truncated
    # 5.5. Repeat the 5.{2,3, 4} until the end of the episode
    # visualize the agent
    visualize_env(env=env, pause_sec=0.1)
    # continue with the next step without closing the plot

# 5.6. How much total reward you got? What does it mean to have large/small reward?
print("Total reward:", cumulative_reward)

print("Good-bye.")
