# Random Agent 🎲
Build a simple baseline that samples random actions and interacts with the environment. This exercise cements the RL loop: observe → act → receive reward → repeat.

## Objectives 🧠
- [ ] Create and reset a Gymnasium environment
- [ ] Sample valid actions from `action_space`
- [ ] Step through episodes and accumulate rewards
- [ ] Repeat episodes and compare returns

---

## 0 — Create an environment 🚀
```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="rgb_array")
```
>[!TIP]
> Use `render_mode="human"` for an on-screen window while stepping through the env.

## 1 — Reset the environment ♻️
Every episode starts with `reset()`:
```python
obs, info = env.reset()
```

## 2 — Take random actions 🎯
- Sample a random valid action: `action = env.action_space.sample()`
- Step once and unpack the result tuple:
```python
obs, reward, terminated, truncated, info = env.step(action)
```

## 3 — Run an entire episode ▶️
Loop until the episode ends (either `terminated` or `truncated`):
```python
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0

while not (terminated or truncated):
    action = ... # implement a random action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Episode return: {total_reward}")
```

## 4 — Evaluate over multiple episodes 📊
```python
import numpy as np

returns = []
for _ in range(10):
    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = ... # implement a random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    returns.append(total_reward)

print(f"Mean return over 10 episodes: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
```

>[!TIP]
> Want reproducibility? Set a seed: `env.reset(seed=42)`.

---

## What to notice 🔍
- Returns vary across episodes because actions are random.
- The step result is a 5-tuple: `(obs, reward, terminated, truncated, info)`.
- CartPole actions are discrete: `0` (left), `1` (right).

## Next steps ➡️
Move on to the learning agent: [learning_agent.md](./learning_agent.md) 📈
