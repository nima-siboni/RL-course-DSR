import gymnasium as gym
# Create an environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# 0 - Reset the environment
print("Environment is resetted.\n")
state, info = env.reset()
# 1 - Let's examine the env
# 1.1 - Check the observation space of the agent
print(f'Observation space of the environment is: {env.observation_space}\n')
# 1.2 - Check the action space of the agent
print(f'Action space of the environment is {env.action_space}\n')
# 1.3 - Render the env
rendering = env.render()
print(f'The shape of the object returned by render is {rendering.shape}\n')
# 1.4 - Reset the environment
# 1.5 - Check the source code of the env.

# 2 - Interact with environment
# 2.1 - Push the cart to the left once
# 2.2 - Repeat the action above until the pole falls, i.e. the episode is terminated
done = False
while not done:
    new_state, reward, terminated, truncated, info = env.step(action=1)
    print(f'state {new_state}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}')
    done = terminated or truncated
    