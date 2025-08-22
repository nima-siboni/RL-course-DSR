"""
Q-Learning Algorithm Implementation 🎯

This script demonstrates Q-Learning on an uneven terrain maze environment.
The algorithm learns Q-values through experience and updates them using the Q-learning update rule.

Key concepts:
- Q-Learning: Off-policy TD learning that learns Q(s,a) directly
- Experience Replay: Learn from episodes and update Q-values
- Epsilon-Greedy: Balance exploration and exploitation
- Temporal Difference: Update Q-values using bootstrapped estimates
"""
import copy
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
from rlcomponents.agent import Agent
from rlcomponents.policy import Policy
from rlcomponents.rl_utils import calculate_epsilon
from tqdm import tqdm
from uneven_maze import UnevenMaze, sample_terrain_function

# ============================================================================
# 1. ENVIRONMENT SETUP 🗺️
# ============================================================================
print("🚀 Setting up the uneven terrain environment for Q-Learning...")

env_config = {
    "width": 5,                    # Grid width
    "height": 5,                   # Grid height  
    "mountain_height": 1.0,        # Maximum terrain elevation
    "goal_position": [4, 0],       # Target position (bottom-right)
    "max_steps": 100,              # Episode length limit
    "cost_height": 0.0,            # No penalty for elevation changes
    "cost_step": 1.0,              # Cost per step (encourages efficiency)
    "terrain_function": sample_terrain_function,  # Terrain generation function
    "diagonal_actions": False,     # Only cardinal directions (up, down, left, right)
    "walled_maze": True,           # Include maze walls/obstacles
}

# Create environment and agent
env = UnevenMaze(config=env_config)
policy = Policy(env)  # Initial random policy
agent = Agent(env, policy=policy, gamma=0.99)  # Discount factor for future rewards

# ============================================================================
# 2. TRAINING PARAMETERS ⚙️
# ============================================================================
INITIAL_EPSILON = 1.0  # Start with 100% exploration
epsilon = copy.deepcopy(INITIAL_EPSILON)
NUM_TRAINING_ITERATIONS = 50
EPISODES_PER_ITERATION = 50
LEARNING_RATE = 0.1

print(f"📊 Training for {NUM_TRAINING_ITERATIONS} iterations...")
print(f"🎲 Starting epsilon: {INITIAL_EPSILON}")
print(f"📚 Learning rate: {LEARNING_RATE}")

# ============================================================================
# 3. Q-LEARNING TRAINING LOOP 🔄
# ============================================================================
for training_iteration in tqdm(range(NUM_TRAINING_ITERATIONS), desc="Training Progress"):
    # Step 1: Update epsilon for exploration-exploitation balance
    epsilon = calculate_epsilon(
        initial_epsilon=INITIAL_EPSILON,
        training_id=training_iteration,
        epsilon_decay_window=50,
    )

    # Step 2: Create epsilon-greedy policy from current Q-values
    q_values = agent.policy.q_values
    pi = agent.find_epsilon_greedy_policy_using_qs(q_values=q_values, epsilon=epsilon)
    agent.set_policy(pi)

    # Step 3: Learn from multiple episodes
    for episode in range(EPISODES_PER_ITERATION):
        agent.run_an_episode_and_learn_from_it(alpha=LEARNING_RATE)

    # Step 4: Visualize Progress 🎬
    # Show current policy performance (every few iterations to avoid too many plots)
    if training_iteration % 10 == 0:  # Show every 10th iteration
        plt.close()  # Close previous plots
        print(f"\n🎬 Visualizing iteration {training_iteration + 1}...")
        agent.run_an_episode_using_q_values(
            state=[0, 0], render=True, epsilon=epsilon, greedy=True, colors="values"
        )
        plt.pause(0.1)  # Pause to show the plot

print("\n✅ Q-Learning training complete!")
print(f"🎯 Final epsilon: {epsilon:.3f}")
print("🚀 The agent should now navigate efficiently to the goal!")

# ============================================================================
# 4. FINAL DEMONSTRATION 🎬
# ============================================================================
print("\n🎬 Final demonstration episode...")
plt.figure(figsize=(8, 6))
agent.run_an_episode_using_q_values(
    state=[0, 0], render=True, epsilon=0.0, greedy=True, colors="values"
)
plt.title("Final Q-Learning Policy")
plt.show()
