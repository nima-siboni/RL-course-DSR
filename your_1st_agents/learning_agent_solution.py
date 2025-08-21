"""Your first learning agent - DQN on CartPole-v1 using RLlib."""
import gymnasium as gym
from plot_util import visualize_env
from ray.rllib.algorithms.dqn import DQNConfig
import torch

# Step-by-step guide:
# 1. Build an agent
#   1.1 - Get the default config of DQNConfig()
#   1.2 - Examine the config by converting it to a dict via .to_dict() method
#   1.3 - Modify the config if needed (learning rate, environment, etc.)
#   1.4 - Introduce the environment to the agent's config
#   1.5 - Build the agent from the config
# 2. Train the agent for one training round and get the reports
# 3. Run a loop for multiple training iterations
# 4. Visualize the trained agent (similar to random_agent but with learned policy)
#   4.1 - Create an environment similar to the training env
#   4.2 - Let the agent choose an action using the trained policy
#   4.3 - Pass the action to the environment
#   4.4 - Track the reward for that action
#   4.5 - Repeat until the episode ends
#   4.6 - Report the total reward achieved


# 1 - Build an agent
# 1.1 - Get the default config of DQNConfig()
config = DQNConfig()

# 1.2 - Examine the config by converting it to a dict via .to_dict() method
# (Optional: uncomment to inspect all available settings)
# print(config.to_dict())

# 1.3 - Modify the config if needed (learning rate, environment, etc.)
# For more examples of common configs to change check: https://docs.ray.io/en/latest/rllib/algorithm-config.html#generic-config-settings
# For the complete list of configs check: https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig

# 1.4 - Configure the agent with environment and training settings
config.training(lr=0.0005)  # Set learning rate
config.environment(env="CartPole-v1")  # Specify the environment
config.env_runners(
    num_env_runners=4,  # Number of parallel environment runners
    num_envs_per_env_runner=2  # Environments per runner
)
config.evaluation(
    evaluation_config={"explore": False},  # No exploration during evaluation
    evaluation_duration=10,  # Evaluate for 10 episodes
    evaluation_interval=1,  # Evaluate every training iteration
    evaluation_duration_unit="episodes",
)
config.rl_module(
    model_config={
        'fc_hiddens': [256, 256],  # Two hidden layers with 256 units each
        'fcnet_activation': 'tanh'  # Use tanh activation
    }
)

# Optional: inspect the final configuration
config_as_a_dict = config.to_dict()

# 1.5 - Build the agent from the config
agent = config.build_algo()
# 2 - Train the agent for one training round with .train and get the reports
# (Optional: uncomment to see what a single training step returns)
# reports = agent.train()
# print(reports)

# 3 - Run a loop for multiple training iterations
nr_trainings = 20  # pylint: disable=invalid-name
mean_rewards = []
for _ in range(nr_trainings):
    training_logs = agent.train()
    mean_total_reward = training_logs['evaluation']['env_runners']['episode_return_mean']
    print(f'mean total reward: {mean_total_reward}')

print("This is the end of the training 🚀")

# Optional: we can save the algorithm by agent.save_to_path() function, and later load it for inference
# agent.save_to_path("checkpoint_path")

# Get the trained neural network (RL module) for inference
rl_module = agent.get_module()


# 4 - Visualize the trained agent (similar to random_agent but with learned policy)
# 4.1 - Create an environment similar to the training env
env = gym.make("CartPole-v1", render_mode="rgb_array")
s, _ = env.reset()
done = False  # pylint: disable=invalid-name
cumulative_reward = 0  # pylint: disable=invalid-name

while not done:
    # 4.2 - Let the agent choose an action using the trained policy
    obs_batch = torch.from_numpy(s).unsqueeze(0)  # Convert to batch format
    a = rl_module.forward_inference({'obs': obs_batch})['actions'].numpy()[0]  # Get action from trained policy
    
    # 4.3 - Pass the action to the environment
    s, r, terminated, truncated, info = env.step(action=a)

    # 4.4 - Track the reward for that action
    cumulative_reward += r
    done = terminated or truncated
    
    # 4.5 - Visualize the agent in action (repeat until episode ends)
    visualize_env(env=env, pause_sec=0.1)

# 4.6 - Report the total reward achieved
print("Total reward:", cumulative_reward)

print("Good-bye.")
