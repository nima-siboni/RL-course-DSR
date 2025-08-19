# Investigate an environment together 🔎
Get hands-on with Gymnasium environments:
- [ ] Create an environment 🛠️
- [ ] Inspect its key properties (observation_space, action_space) 🔍
- [ ] Reset and render the environment ▶️
- [ ] Take actions and read the feedback (obs, reward, terminated, truncated, info) 🔄

## 0 — Create an environment 🚀
```python
import gymnasium as gym
env = gym.make("CartPole-v1", render_mode="rgb_array")
```
>[!TIP]
> Use `render_mode="human"` for an on-screen window while stepping through the env.
## 1 — Explore the environment 🧭
### 1.1 — Observation space
What does the agent observe? How is it bounded? Try:
- Print `env.observation_space`
- Sample a fake observation with `env.observation_space.sample()`

### 1.2 — Action space
What actions are available? Discrete or continuous?
- Print `env.action_space`
- Sample a random valid action with `env.action_space.sample()`
- Find another way to sample a random valid action!

### 1.3 — Render the env
Call `env.render()` to obtain an RGB array (or a window with `render_mode="human"`).

### 1.4 — Reset the environment
Call `obs, info = env.reset()` and inspect both `obs` and `info`.

### 1.5 — Peek at the source
Skim the implementation (CartPole) to connect API to dynamics.

## 2 — Interact with the environment 🎮
### 2.1 — Push the cart to the left once
```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(0)  # 0 = left in CartPole
```

### 2.2 — Repeat until the pole falls (episode terminates)
```python
obs, info = env.reset()
terminated = truncated = False
total_reward = 0.0
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(0)  # keep pushing left
    total_reward += reward
```

Get started in the [template script](./playground.py). Happy hacking! 🧑‍💻