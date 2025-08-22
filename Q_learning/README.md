# Q-Learning 🎯
Congratulations! You've reached Q-Learning - one of the most fundamental and powerful reinforcement learning algorithms! 🚀

## What You'll Learn 🧠
- **Q-Learning**: Off-policy temporal difference learning
- **Q-Value Updates**: The core Q-learning update rule
- **Epsilon-Greedy Exploration**: Balancing exploration vs exploitation
- **Experience-Based Learning**: Learning from actual interactions with the environment

## Prerequisites ✅
- Basic understanding of reinforcement learning concepts
- Familiarity with Python and numpy
- Completed previous exercises (basic agents, environment interaction)

---

## Q-Learning Algorithm 🔄

Q-Learning is an **off-policy** algorithm that learns Q(s,a) values directly from experience, without needing a model of the environment.

### Core Concept 💡
Start with random Q-values and update them based on experience using the Q-learning update rule:

```python
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- **α (alpha)**: Learning rate
- **r**: Immediate reward
- **γ (gamma)**: Discount factor
- **max Q(s',a')**: Maximum Q-value in next state

### Algorithm Structure 📋
```python
# Initialize Q-values randomly or to zero
Q = random_values()

for training_round in range(num_rounds):
    # Create epsilon-greedy policy from current Q-values
    policy = create_epsilon_greedy_policy(Q, epsilon)
    
    # Run episode and learn from experience
    for episode in range(episodes_per_round):
        state = env.reset()
        while not terminated:
            # Choose action using current policy
            action = policy.choose_action(state)
            
            # Take action and observe result
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Update Q-value using Q-learning rule
            Q[state, action] = update_q_value(Q, state, action, reward, next_state)
            
            state = next_state
    
    # Decay epsilon for exploration-exploitation balance
    epsilon = decay_epsilon(epsilon, training_round)
```

---

## Environment Setup 🗺️

The uneven terrain maze provides a rich learning environment:
- **5×5 grid** with varying elevation
- **Goal**: Navigate from start to goal efficiently
- **Actions**: Cardinal directions (up, down, left, right)
- **Rewards**: Negative step cost encourages efficiency

---

## Key Implementation Details 🔧

### Q-Value Update Function
The heart of Q-Learning is the update rule:

```python
def update_q_value(Q, state, action, reward, next_state, alpha=0.1, gamma=0.99):
    """
    Update Q-value using Q-learning rule.
    
    Args:
        Q: Q-value table
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        alpha: Learning rate
        gamma: Discount factor
    
    Returns:
        Updated Q-value
    """
    current_q = Q[state, action]
    max_next_q = np.max(Q[next_state, :])  # Maximum Q-value in next state
    target_q = reward + gamma * max_next_q
    new_q = current_q + alpha * (target_q - current_q)
    return new_q
```

### Epsilon-Greedy Policy
Balance exploration and exploitation:
- **ε probability**: Choose random action (exploration)
- **(1-ε) probability**: Choose greedy action argmax Q(s,a) (exploitation)

---

## Running the Code 🚀

```bash
cd /path/to/RL-course-DSR
python Q_learning/experience_and_learn.py
```

**Expected output:**
- Training progress with epsilon values
- Visual episodes showing agent navigation
- Convergence to efficient path-finding policy

---

## Exercises & Challenges 🎯

### Exercise 1: Implement Q-Value Update
Write a function that updates Q-values using the Q-learning rule. Look for hints in the comments of `experience_and_learn.py`.

**Key considerations:**
- How do you compute the target Q-value?
- What's the role of the learning rate α?
- How does the discount factor γ affect learning?

### Exercise 2: Policy Update Analysis
At the beginning of each training round, a new policy is created from the updated Q-values.

**Think about:**
- What happens if you don't update the policy?
- What if you always use a random policy instead?
- How does this affect convergence and learning speed?

### Exercise 3: Hyperparameter Tuning 🔬
Experiment with different hyperparameters:
- **Learning rate (α)**: Try values between 0.01 and 0.5
- **Discount factor (γ)**: Test values between 0.8 and 0.99
- **Epsilon decay**: Adjust exploration vs exploitation balance

### Bonus Exercise: Experience Replay 🏆
Implement experience replay to improve learning:
- Store (s, a, r, s') tuples in a buffer
- Sample random experiences for updates
- How does this affect learning stability?

---

## Learning Objectives 📚

By the end of this exercise, you should understand:
- ✅ How Q-Learning updates Q-values using temporal differences
- ✅ The difference between on-policy and off-policy learning
- ✅ How epsilon-greedy balances exploration and exploitation
- ✅ The role of hyperparameters in Q-Learning
- ✅ How to implement and debug Q-Learning algorithms

---

## Key Differences from Other Algorithms 🔍

| Aspect | Q-Learning | SARSA | Monte Carlo |
|--------|------------|-------|-------------|
| **Policy** | Off-policy | On-policy | On-policy |
| **Update** | TD(0) | TD(0) | Full episode |
| **Bootstrapping** | Yes | Yes | No |
| **Sample Efficiency** | Medium | Medium | Low |

---

## Next Steps ➡️
- Compare with SARSA (on-policy TD learning)
- Experiment with function approximation (Deep Q-Learning)
- Try different exploration strategies
- Move on to policy gradient methods