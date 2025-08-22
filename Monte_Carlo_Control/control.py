"""
Monte Carlo Control Algorithm Implementation 🎯

This script demonstrates Monte Carlo Control on an uneven terrain maze environment.
The algorithm alternates between policy evaluation and policy improvement until convergence.

Key concepts:
- Policy Evaluation: Estimate V(s) for current policy using Monte Carlo sampling
- Policy Improvement: Create new policy using epsilon-greedy approach on Q-values
- Epsilon Decay: Gradually reduce exploration as training progresses
"""
import copy
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
from rlcomponents.agent import Agent
from rlcomponents.policy import Policy
from rlcomponents.rl_utils import calculate_epsilon
from uneven_maze import UnevenMaze, sample_terrain_function

# ============================================================================
# 1. ENVIRONMENT SETUP 🗺️
# ============================================================================
print("🚀 Setting up the uneven terrain environment...")

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
policy = Policy(env)  # Initializes with a random policy
agent = Agent(env, policy=policy, gamma=0.98)  # Discount factor for future rewards

# ============================================================================
# 2. TRAINING PARAMETERS ⚙️
# ============================================================================
INITIAL_EPSILON = 1.0  # Start with 100% exploration
epsilon = copy.deepcopy(INITIAL_EPSILON)
NUM_TRAINING_ITERATIONS = 3  # Reduced for testing
EVAL_EPISODES = 100  # Episodes per policy evaluation

print(f"📊 Training for {NUM_TRAINING_ITERATIONS} iterations...")
print(f"🎲 Starting epsilon: {INITIAL_EPSILON}")

# ============================================================================
# 3. MONTE CARLO CONTROL LOOP 🔄
# ============================================================================
for training_iteration in range(NUM_TRAINING_ITERATIONS):
    print(f"\n🔄 Iteration {training_iteration + 1}/{NUM_TRAINING_ITERATIONS}")
    
    # Step 1: Policy Evaluation 📈
    # Estimate V(s) for current policy using Monte Carlo sampling
    print("  📈 Evaluating current policy...")
    values = agent.evaluate_pi(eval_episodes=EVAL_EPISODES, greedy=False)
    
    # Step 2: Policy Improvement 🎯
    # Create new epsilon-greedy policy based on Q-values derived from V(s)
    print("  🎯 Improving policy...")
    pi = agent.improve_policy(values=values, gamma=0.98, epsilon=epsilon)
    agent.set_policy(pi)
    
    # Step 3: Epsilon Decay 📉
    # Gradually reduce exploration to focus on exploitation
    epsilon = calculate_epsilon(
        initial_epsilon=INITIAL_EPSILON,
        training_id=training_iteration,
        epsilon_decay_window=10,
    )
    print(f"  📉 Epsilon: {epsilon:.3f}")
    
    # Step 4: Visualize Progress 🎬
    # Show current policy performance (optional)
    plt.close()  # Close previous plots
    print("  🎬 Running visualization episode...")
    agent.run_an_episode(state=[0, 0], render=True, greedy=True, colors="values")
    plt.pause(0.1)  # Pause to show the plot

print("\n✅ Training complete!")
print(f"🎯 Final epsilon: {epsilon:.3f}")
print("🚀 The agent should now navigate efficiently to the goal!")

# Final visualization
print("\n🎬 Final demonstration episode...")
plt.figure(figsize=(8, 6))
agent.run_an_episode(state=[0, 0], render=True, greedy=True, colors="values")
plt.title("Final Trained Policy")
plt.show()
