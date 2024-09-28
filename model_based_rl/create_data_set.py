"""Creating a data set for model-based reinforcement learning.

Here the idea is to create three different datasets for cartpole environment:
1. A data set created from the interactions of a random data_creator with the environment.
2. A data set created from the interactions of an data_creator which is trained for 10 episodes with
 env.
3. A data set created from the interactions of an data_creator which is trained for 100 episodes
with env.
"""
# 0. Import necessary libraries
from dataset_creator import DataCreator
from matplotlib import pyplot as plt

# 1. Create a class which has an data_creator that can be trained for a number of training episodes.


# 2. Create a data_creator and train it for 0 episodes.
data_creator = DataCreator()  # pylint: disable=invalid-name
nr_training_rounds = 10  # pylint: disable=invalid-name
mean_reward_lst = []
for i in range(5):
    data_set, mean_reward = data_creator.create_data_set(nr_episodes=30)
    # shuffle the data set
    data_set = data_set.sample(frac=1)
    # save the data set
    data_set.to_pickle(f"data_sets/data_set_{i * nr_training_rounds}_rounds.pkl")
    mean_reward_lst.append(mean_reward)
    print(f"Mean reward for {i * nr_training_rounds} rounds: {mean_reward}")
    data_creator.train(_nr_training_rounds=nr_training_rounds)

# 3. Plot and save the mean rewards for further analysis

plt.plot(mean_reward_lst)
plt.xlabel("Training rounds")
plt.ylabel("Mean reward")
plt.title("Mean reward vs. training rounds")
plt.savefig("mean_reward_vs_training_rounds.png")
plt.close()
