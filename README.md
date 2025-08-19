# Introduction to (Deep) Reinforcement Learning
Here are the exercises for the first/second session of our RL course.

## What You'll Learn

This course provides a comprehensive hands-on introduction to Reinforcement Learning (RL) through practical exercises. You'll progress from basic concepts to advanced techniques, building a solid foundation in both theory and implementation.

### Course Structure & Learning Path

**🎯 Getting Started**
- **Environment Interaction** (`interaction_with_env/`): Learn to create, examine, and interact with RL environments using Gymnasium. Understand observation spaces, action spaces, and basic environment dynamics.

- **Your First Agents** (`your_1st_agents/`): Build your first RL agents - from simple random agents to learning agents that improve over time. Master the fundamentals of agent creation, training, and evaluation.

**🧠 Core RL Algorithms**

- **Monte Carlo Control** (`Monte_Carlo_Control/`): Explore policy evaluation and improvement through Monte Carlo methods. Implement the core components of MC control and understand epsilon-greedy policies.

- **Q-Learning** (`Q_learning/`): Implement the foundational Q-learning algorithm from scratch. Learn to update Q-values based on experience and understand the relationship between policies and value functions.

**🔧 Advanced Topics**
- **Model-Based RL** (`model_based_rl_v0/`): Learn to build and train neural network models of environments. Understand how to use learned models for planning and decision-making.

- **Partially Observable Environments** (`partially_observable_env/`): Work with environments where agents have limited information. Implement solutions for CartPole with partial observability using deep RL techniques.

- **Custom Environments** (`uneven_maze/`): Create and work with custom RL environments. Explore multi-objective optimization in a maze with uneven terrain.

**🛠️ Tools & Frameworks**
- **RL Components** (`rlcomponents/`): Build reusable RL components including agents and policies. Learn to structure RL code for maintainability and reusability.


### Key Learning Outcomes

By the end of this course, you will be able to:
- ✅ Create and interact with RL environments
- ✅ Implement fundamental RL algorithms (Q-learning, Monte Carlo)
- ✅ Build and train neural networks for model-based RL
- ✅ Handle partially observable environments
- ✅ Design custom RL environments
- ✅ Structure RL code using best practices
- ✅ Use modern RL frameworks and tools

### Prerequisites
Basic Python programming knowledge and familiarity with machine learning concepts. No prior RL experience required!

## Requirements
Create a virtual environment using uv (recommended) or python, e.g.
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a virtual environment
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```
Then install the required packages:
```bash
uv sync
```

# Development Setup
For development with additional tools like pre-commit:
```bash
uv sync --extra dev
```

## Presentation
The presentation of the course can be found [here](https://docs.google.com/presentation/d/1_REcZjt23UiGeazb8a7_g39gUx-7P_riRnj-WMRWAzU/edit?usp=sharing).

## Some references
The [old testament bible](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf), God him/her/theirselves.

[Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/), Sergey Levine.

[Introduction to Reinforcement Learning](
https://deepmind.com/learning-resources/-introduction-reinforcement-learning-david-silver), David Silver.

[Reinforcement Learning Virtual School](https://rl-vs.github.io/rlvs2021/), 2021.

Finally, the real bible!

## Some places to go

### Environments
* [OpenAI Gym environments](https://gym.openai.com/envs/#classic_control) a collection of RL Hello worlds environments.
* [A short list of interesting environments](https://medium.com/@mauriciofadelargerich/reinforcement-learning-environments-cff767bc241f)
* [A more exhaustive list of environments](https://github.com/clvrai/awesome-rl-envs)
* [InstaDeep's Jumanji](https://instadeepai.github.io/jumanji/)

### RL framework

* [Stable Baselines](https://stable-baselines.readthedocs.io/en/master/index.html) is
a set of improved implementations of Reinforcement Learning (RL) algorithms based on OpenAI Baselines. You can find good papers here!

* [RLlib](https://docs.ray.io/en/master/tune/key-concepts.html) is an open-source
library for reinforcement learning.

* [InstaDeep's Mava](https://github.com/instadeepai/Mava)
