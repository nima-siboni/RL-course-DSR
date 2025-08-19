# Your First Agents! 🤖
Build two simple agents to internalize the RL loop end-to-end:
- 🎲 A random agent
- 📈 A learning agent

## What you'll learn 🧠
- [ ] Create and reset Gymnasium environments
- [ ] Inspect and use `observation_space` and `action_space`
- [ ] Step through an environment and accumulate rewards
- [ ] Structure a basic agent and evaluate it
- [ ] Configure, train, and evaluate a learning agent

## Prerequisites ✅
- Python and Gymnasium installed
- Basic familiarity with Python control flow and functions

---

## 1) Random agent (warm-up) 🎲
Goal: interact with an environment by sampling valid actions and tracking returns.

You'll do:
- Create and reset `CartPole-v1`
- Sample actions from `env.action_space`
- Step through an episode until `terminated` or `truncated`
- Keep and report total reward; repeat and compare runs

Start here:
- Local guide: [random_agent.md](./random_agent.md)
- Colab: [Random Agent Notebook](https://colab.research.google.com/drive/1_OZ-qnZO2hYd12HGp_urXZIBQo0cHq8M?authuser=1#scrollTo=MWr_fBo5WT59)

Tips 💡
- Use `env.action_space.sample()` to get a valid random action
- Try both `render_mode="rgb_array"` and `render_mode="human"`

---

## 2) Learning agent (train and evaluate) 📈
Goal: configure a standard RL algorithm, train on `CartPole-v1`, and evaluate.

You'll do:
- Choose an algorithm (e.g., DQN via RLlib)
- Inspect and adjust configuration (env, training settings)
- Train for multiple iterations and inspect returns
- Visualize/evaluate the trained policy vs. the random baseline

Start here:
- Local guide: [learning_agent.md](./learning_agent.md)
- [RLlib algorithms reference](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html)
- Colab: [Learning Agent Notebook](https://colab.research.google.com/drive/1xFcXDlDce1_c4VHtGV3d06lOdvi6RGIt?authuser=1#scrollTo=rktqiU2Ec18P)

Tips 💡
- Keep train/eval environments aligned
- Track episode reward; compare to the random agent

---

## Checklist for success ✅
- [ ] You can explain the tuple returned by `env.step(...)`
- [ ] You can run and repeat episodes, logging total rewards
- [ ] You configured and trained a learning agent
- [ ] You compared performance: random vs. trained

## Where to go next ➡️
- Explore envs and APIs: `interaction_with_env/`
- Implement core algorithms: `Q_learning/`, `Monte_Carlo_Control/`
