# Monte Carlo Control 🎯
Learn to implement Monte Carlo Control on an uneven terrain maze environment!

## What You'll Learn 🧠
- **Policy Evaluation**: Estimate state values using Monte Carlo sampling
- **Policy Improvement**: Create epsilon-greedy policies from Q-values
- **Exploration vs Exploitation**: Balance with epsilon decay
- **Value Function Approximation**: From V(s) to Q(s,a) to π(s)

## Prerequisites ✅
- Basic understanding of reinforcement learning concepts
- Familiarity with Python and numpy
- Completed previous exercises (Q-learning, basic agents)

---

## Monte Carlo Control Algorithm 🔄

Monte Carlo Control alternates between two key steps until convergence:

### 1. Policy Evaluation 📈
Estimate V(s) for the current policy using Monte Carlo sampling:
```python
# Run multiple episodes following current policy
# Average returns to estimate V(s) for each state
values = agent.evaluate_pi(eval_episodes=100, greedy=False)
```

### 2. Policy Improvement 🎯
Create a new epsilon-greedy policy based on Q-values:
```python
# For each state, calculate Q(s,a) using V(s)
# Create epsilon-greedy policy: mostly greedy, sometimes random
pi = agent.improve_policy(values=values, gamma=0.98, epsilon=epsilon)
```

### 3. Epsilon Decay 📉
Gradually reduce exploration to focus on exploitation:
```python
epsilon = calculate_epsilon(
    initial_epsilon=1.0,
    training_id=iteration,
    epsilon_decay_window=10
)
```

---

## Environment Setup 🗺️

The uneven terrain maze provides a rich environment for learning:
- **5×5 grid** with varying elevation
- **Goal**: Navigate from start to goal efficiently
- **Actions**: Cardinal directions (up, down, left, right)
- **Rewards**: Negative step cost encourages efficiency

---

## Key Implementation Details 🔧

### Q-Value Calculation
The core challenge is computing Q(s,a) from V(s):

```python
# For each state s and action a:
Q(s,a) = Σ P(s'|s,a) * [R(s,a,s') + γ * V(s')]
```

**Main ingredients:**
1. **Transition probabilities** P(s'|s,a)
2. **Reward function** R(s,a,s')
3. **Discount factor** γ
4. **Current value estimates** V(s')

### Epsilon-Greedy Policy
Balance exploration and exploitation:
- **ε probability**: Choose random action
- **(1-ε) probability**: Choose greedy action (argmax Q(s,a))

---

## Running the Code 🚀

```bash
cd /path/to/RL-course-DSR
python Monte_Carlo_Control/control.py
```

**Expected output:**
- Training progress with epsilon values
- Visual episodes showing agent navigation
- Convergence to efficient path-finding policy

---

## Exercises & Challenges 🎯

### Main Exercise
Implement the Q-value calculation function:
```python
def calculate_Qs_for_this_state(env, state, V, gamma, nr_actions):
    """
    Calculate Q(s,a) for all actions in a given state.
    
    Args:
        env: Environment with transition model
        state: Current state
        V: Value function V(s)
        gamma: Discount factor
        nr_actions: Number of possible actions
    
    Returns:
        Q: Array of Q-values for each action
    """
    # TODO: Implement Q-value calculation
    pass
```

**Think about:**
- How do you compute P(s'|s,a) for the environment?
- What's the reward structure?
- How does the discount factor affect future rewards?

### Bonus Exercise 🏆
The current epsilon-greedy implementation uses a non-standard approach. Research and implement the standard epsilon-greedy policy creation method.

### Super Bonus Exercise 🌟
Design an alternative approach to create soft policies from Q-values:
- **Softmax policy**: π(a|s) ∝ exp(Q(s,a)/τ)
- **Boltzmann exploration**: Temperature parameter τ controls exploration
- **UCB-based policies**: Use uncertainty estimates

**Discussion points:**
- When is each approach most effective?
- How do they balance exploration vs exploitation?
- What are the computational trade-offs?

---

## Learning Objectives 📚

By the end of this exercise, you should understand:
- ✅ How Monte Carlo methods estimate value functions
- ✅ The policy iteration framework (evaluate → improve)
- ✅ Epsilon-greedy exploration strategies
- ✅ The relationship between V(s) and Q(s,a)
- ✅ How to implement and debug RL algorithms

---

## Next Steps ➡️
- Compare with other control methods (SARSA, Q-learning)
- Experiment with different exploration strategies
- Try larger environments or different reward structures
- Move on to function approximation methods