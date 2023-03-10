# Here you create an env and let an agent interact with it. You can measure how
# successful is the random agent, i.e. how much reward it accumulates.
import numpy as np
import gymnasium as gym

# 0 - Create an env

env = gym.make("CartPole-v1")

# 1 - Reset the env.
env.reset()
# 2 - Let a random agent interact with the env.

# 2.1. Choose a random action (with in the action space of the env.)
nr_actions = env.action_space.n
done = False
cumulative_reward = 0
while not done:
    a = np.random.randint(nr_actions)
    # 2.2. and pass it to the environment
    s, r, terminated, truncated, info = env.step(action=a)
    # 2.3. How much reward did you get for that action? Keep the score!
    cumulative_reward += r
    # 2.4. Repeat the 2.{1,2,3} until the end of the episode
    done = terminated or truncated
# 2.5. How much total reward you got? What does it mean to have large/small reward?
print("total reward:", cumulative_reward)

# 3. Repeat the whole section 2. Do you get the same total reward?

print("Goodbye")
