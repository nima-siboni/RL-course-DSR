# Introduction to (Deep) Reinforcement Learning
Here are the exercises for the first/second session of our RL course.

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
