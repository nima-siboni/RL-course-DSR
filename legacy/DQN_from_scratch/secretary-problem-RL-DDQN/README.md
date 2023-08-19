# secretary-problem-RL-DDQN

## Problem

Here we address a modification of [the secretary problem](https://en.wikipedia.org/wiki/Secretary_problem), where the goal is to maximize *the expectation value* for the score of the hired applicant [this modification makes the rewards Markovian].

An evironment, consisted with OpenAI Gym environment, is used for this project. This environment can be found [here](https://github.com/nima-siboni/secretary-problem-env/). In this environment the applicants are ranked by a random scores in range ```[0, 1]```. 

## Solution

Here we have implemented the DDQN approach with a network of 2-layers. Our results show that, for an environment with ```N=20``` candidatres, the agent can increase its performance up to ```0.92-0.93```, where the performance is equal to the average scores of the accepted candidates. 

<img src="./performance-and-animations/results.png" width="60%">

This is a remarkable achievement, as the performance one can obtain using simple Monte-Carlo simulation is almost ```0.8``` (for ```N=20```; this experiment is done [here](https://github.com/nima-siboni/recruiter-problem) ). The reason that our agent can do better than that, could be traced back to the fact that in the secretary problem the secretary does not know anything about the candidates out there. In contrary, our agent had the opportunity to experience the candidate's distribution, together with experimenting to find the optimal stopping point.

## Requirements

```
gym==0.17.2
keras
numpy
random2
tensorflow==2.2.0
tqdm
```

All the requirements are included ```requirements.txt``` file.

## Usage

After cloning the repository, run 

``` python experience-and-learn.py```

This script creates the agent, and the episodes from which it should learn. The performance of the agent during the learning process is tracked in ```performance-and-animations/steps_vs_iterations.dat``` in format of two columns: (i) id of the learning episode, and (ii) average performance after that episode.
