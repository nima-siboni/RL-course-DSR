# The Learning Agent 📈
Create, train, and evaluate a learning agent using RLlib (DQN on `CartPole-v1`). Watch it go from random to smart! 🧠

## Objectives 🎯
- [ ] Get familiar with RLlib (industry-grade RL library)
- [ ] Configure and build a DQN agent
- [ ] Train the agent and monitor progress
- [ ] Visualize the trained policy in action

## Prerequisites ✅
- Python 3.10+
- Install RLlib: `pip install "ray[rllib]"`

---

## 1 — Build the agent from a config 🧩
We start by creating a configuration object that will define all the settings for our DQN algorithm.

```python
from ray.rllib.algorithms.dqn import DQNConfig

config = DQNConfig()
```
>[!NOTE]
> This is just RLlib's approach to build a learning algorithm: it start from a configuration

>[!TIP]
> Start with the default config and explore with `config.to_dict()` to see all available settings!

---

## 2 — Customize the configuration ⚙️
We configure the training parameters, environment, parallel processing, evaluation settings, and neural network architecture.

```python
config.training(lr=0.0005)
config.environment(env="CartPole-v1")
config.env_runners(
    num_env_runners=4,
    num_envs_per_env_runner=2
)
config.evaluation(
    evaluation_config={"explore": False},
    evaluation_duration=10,
    evaluation_interval=1,
    evaluation_duration_unit="episodes",
)
config.rl_module(
    model_config={
        'fc_hiddens': [256, 256],
        'fcnet_activation': 'tanh'
    }
)
```

### What each setting does 🔧
- **`config.training(lr=0.0005)`**: Sets the learning rate for the neural network optimizer
- **`config.environment(env="CartPole-v1")`**: Specifies which environment to train on
- **`config.env_runners(...)`**: Configures parallel training with 4 runners, each handling 2 environments (8 total parallel envs)
- **`config.evaluation(...)`**: Sets up evaluation to run every training iteration for 10 episodes without exploration
- **`config.rl_module(...)`**: Defines the neural network architecture (2 hidden layers of 256 units with tanh activation)

>[!NOTE]
> This setup uses 4 parallel environments, evaluation every iteration, and a 2-layer neural network with tanh activation.

### Useful Resources 📚
- [Common RLlib configuration settings](https://docs.ray.io/en/latest/rllib/algorithm-config.html#generic-config-settings)
- [Complete RLlib configuration reference](https://docs.ray.io/en/latest/rllib/package_ref/doc/ray.rllib.algorithms.algorithm_config.AlgorithmConfig.html#ray.rllib.algorithms.algorithm_config.AlgorithmConfig)

---

## 3 — Build and train the agent 🏋️
We construct the agent from our configuration and run multiple training iterations while monitoring the performance.

```python
agent = config.build_algo()

nr_trainings = 20
mean_rewards = []
for _ in range(nr_trainings):
    training_logs = agent.train()
    mean_total_reward = training_logs['evaluation']['env_runners']['episode_return_mean']
    print(f'mean total reward: {mean_total_reward}')
```

>[!TIP]
> Watch the mean reward increase over iterations! 🚀

---

## 4 — Visualize the trained agent 🎬
We test our trained agent by running it through a complete episode and visualizing its behavior in real-time.

```python
import gymnasium as gym
import torch
from plot_util import visualize_env

env = gym.make("CartPole-v1", render_mode="rgb_array")
s, _ = env.reset()
done = False
cumulative_reward = 0

while not done:
    # Let the agent choose an action
    obs_batch = torch.from_numpy(s).unsqueeze(0)
    a = agent.get_module().forward_inference({'obs': obs_batch})['actions'].numpy()[0]
    
    # Step the environment
    s, r, terminated, truncated, info = env.step(action=a)
    cumulative_reward += r
    done = terminated or truncated
    
    # Visualize the agent in action
    visualize_env(env=env, pause_sec=0.1)

print("Total reward:", cumulative_reward)
```

>[!TIP]
> The trained agent should balance the pole much longer than a random agent! 🎯

---

## 5 — Explore and experiment 🔬
We encourage you to experiment with different algorithms, hyperparameters, and configurations to deepen your understanding.

- **Try different algorithms**: Swap `DQNConfig` with `PPOConfig` or `A2CConfig`
- **Adjust hyperparameters**: Change learning rate, network architecture, or exploration
- **Compare performance**: Run multiple training sessions and compare final rewards
- **Save and load**: Use `agent.save_to_path()` to save your trained agent

---

## Troubleshooting 🩹
- Ensure versions are compatible: `ray`, `gymnasium`, `torch`
- If visualization doesn't work, check `plot_util.py` is available
- Keep train and eval environments aligned

## Next steps ➡️
- Compare with the random baseline from [random_agent.md](./random_agent.md) 🎲
- Move on to core algorithms: `../Q_learning/`, `../Monte_Carlo_Control/` 🧠
- Try different environments or algorithms! 🚀
