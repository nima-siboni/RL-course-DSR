# The Learning Agent 📈
Create, train, and evaluate a learning agent using RLlib (DQN on `CartPole-v1`).

## Objectives 🧠
- [ ] Get familiar with RLlib (industry-grade RL library)
- [ ] Choose and configure a learning algorithm
- [ ] Train the agent and monitor progress
- [ ] Evaluate the trained policy vs. a random baseline

## Prerequisites ✅
- Python 3.10+
- Install RLlib: `pip install "ray[rllib]"`

---

## 0 — Choose an algorithm 🎛️
We will use DQN (Deep Q-Network), a classic value-based method. RLlib provides a configuration builder per algorithm.

---

## 1 — Build the agent from a config 🧩
```python
from ray.rllib.algorithms.dqn import DQNConfig

config = (
    DQNConfig()
    .environment(env="CartPole-v1")
    .training(gamma=0.99, lr=1e-3)
    .resources(num_gpus=0)
)

algo = config.build()
```
>[!NOTE]
> RLlib offers sensible defaults. Start simple, then tune! You can inspect all settings with `config.to_dict()`.

---

## 2 — Train the agent 🏋️
```python
for i in range(10):
    result = algo.train()
    print(f"iter={i}  mean_return={result['episode_reward_mean']:.2f}")
```

>[!TIP]
> Increase the number of iterations and watch the mean return improve.

---

## 3 — Evaluate the trained agent 🎯
```python
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0

while not (terminated or truncated):
    action = algo.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Evaluation return: {total_reward}")
```

---

## 4 — Explore and modify the config 🛠️
- Print and skim all settings: `print(config.to_dict())`
- Try a different algorithm (e.g., PPO): swap `DQNConfig` with `PPOConfig`
- Adjust training settings (episodes/iterations, learning rate, exploration)

---

## Troubleshooting 🩹
- Ensure versions are compatible: `ray`, `gymnasium`
- If no window appears, use `render_mode="human"` when creating the env
- Keep train and eval environments aligned (`CartPole-v1` vs. `CartPole-v1`)

## Next steps ➡️
- Compare with the random baseline from [random_agent.md](./random_agent.md)
- Move on to core algorithms: `../Q_learning/`, `../Monte_Carlo_Control/`
