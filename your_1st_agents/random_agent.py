# Here you create an env and let an agent interact with it. You can measure how
# successful is the random agent, i.e. how much reward it accumulates.

# 0 - Create an env
import gym
env = gym.make("CartPole-v1")

# 1 - Reset the env.

# 2 - Let a random agent interact with the env.
#
# 2.1. Choose a random action (with in the action space of the env.)
# 2.2. and pass it to the environment
# 2.3. How much reward did you get for that action? Keep the score!

# 2.4. Repeat the 2.{1,2,3} until the end of the episode

# 2.5. How much total reward you got? What does it mean to have large/small reward?

# 3. Repeat the whole section 2. Do you get the same total reward?