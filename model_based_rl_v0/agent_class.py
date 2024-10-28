"""Train and evaluate a model-based RL data_creator.

The environment is a learned environment, which is a simple neural network that
predicts the next state based on the current state and action. The model is
trained using data collected from different agents

"""
import gymnasium
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from learned_env import LearnedEnv
from ray.rllib.algorithms import DQNConfig
from ray.tune import register_env
from utils import custom_env_creator, evaluate_an_agent_on_an_env

register_env("DSREnv", custom_env_creator)

MODELPATH = "../model_based_rl/model_based_rl_model_good.h5"

# Define and data_creator from RLlib with DQN algorithm
config = (
    DQNConfig()
    .framework(framework="tf2", eager_tracing=True)
    .environment(
        env="DSREnv",
        env_config={
            "render_mode": "rgb_array",
            "ModelPath": MODELPATH,
        },
    )
)
config.rollouts(num_rollout_workers=4)
config.num_envs_per_worker = 2
agent = config.build()

nr_training_rounds = 10  # pylint: disable=C0103
# 3. train the data_creator and evaluate it on the real and learned environment

real_env = gymnasium.make("CartPole-v1")
learned_env = TimeLimit(
    env=LearnedEnv(render_mode="rgb_array", model_path=MODELPATH),
    max_episode_steps=500,
)

perf_on_learned_env_lst = []
perf_on_real_env_lst = []

for _ in range(nr_training_rounds):
    agent.train()

    # 4. Evaluate the data_creator on the real env.
    perf_on_real_env = evaluate_an_agent_on_an_env(
        agent=agent, env=real_env, nr_eval_episodes=4
    )

    perf_on_learned_env = evaluate_an_agent_on_an_env(
        agent=agent, env=learned_env, nr_eval_episodes=4
    )
    perf_on_learned_env_lst.append(perf_on_learned_env)
    perf_on_real_env_lst.append(perf_on_real_env)


# 4. Plot the performance of the data_creator on the real and learned environment

plt.plot(perf_on_learned_env_lst, label="LearnedEnv")
plt.plot(perf_on_real_env_lst, label="RealEnv")
plt.xlabel("Training rounds")
plt.ylabel("Mean reward")
plt.legend()
plt.savefig("model_based_rl_performance_" + MODELPATH + ".png")
plt.close()

# 5. Save the data_creator
agent.save("model_based_rl_agent" + MODELPATH)
print("DataCreator saved.")
