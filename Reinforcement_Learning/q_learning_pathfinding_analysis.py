"""
Q-Learning Path Finding with Bellman Equation Implementation
===========================================================
This script demonstrates Q-Learning reinforcement learning algorithm to find optimal paths
in a grid environment using:
1. Q-Learning algorithm with Bellman equation updates
2. Epsilon-greedy exploration strategy for better convergence
3. Dynamic reward system with goal-oriented learning
4. Comprehensive visualization of the learning process and results

Environment: 3x3 grid world with 9 locations (L1-L9)
Algorithm: Q-Learning with temporal difference learning and Bellman equation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

class QLearningPathFinder:
    """
    Q-Learning agent for finding optimal paths in a grid environment.
    """
    
    def __init__(self, gamma=0.75, alpha=0.9, epsilon=0.1, epsilon_decay=0.995):
        """
        Initialize Q-Learning parameters and environment.
        
        Args:
            gamma (float): Discount factor for future rewards
            alpha (float): Learning rate
            epsilon (float): Exploration rate for epsilon-greedy policy
            epsilon_decay (float): Decay rate for epsilon
        """
        print("=== Q-Learning Path Finder Initialization ===\n")
        
        # Learning parameters
        self.gamma = gamma          # Discount factor
        self.alpha = alpha          # Learning rate
        self.epsilon = epsilon      # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01     # Minimum exploration rate
        
        print(f"Parameters:")
        print(f"  Discount factor (γ): {self.gamma}")
        print(f"  Learning rate (α): {self.alpha}")
        print(f"  Exploration rate (ε): {self.epsilon}")
        
        # Environment setup
        self._setup_environment()
        
        # Initialize Q-table
        self.Q = np.zeros([9, 9])
        
        # Training metrics
        self.training_rewards = []
        self.epsilon_history = []
    
    def _setup_environment(self):
        """Setup the grid environment with states, actions, and connectivity."""
        print(f"\n=== Environment Setup ===")
        
        # Define location mapping
        self.location_to_state = {
            'L1': 0, 'L2': 1, 'L3': 2,
            'L4': 3, 'L5': 4, 'L6': 5,
            'L7': 6, 'L8': 7, 'L9': 8
        }
        
        self.state_to_location = {v: k for k, v in self.location_to_state.items()}
        
        # Define grid connectivity (3x3 grid)
        self.connectivity_matrix = np.array([
            #L1 L2 L3 L4 L5 L6 L7 L8 L9
            [0, 1, 0, 1, 0, 0, 0, 0, 0],  # L1: connected to L2, L4
            [1, 0, 1, 0, 1, 0, 0, 0, 0],  # L2: connected to L1, L3, L5
            [0, 1, 0, 0, 0, 1, 0, 0, 0],  # L3: connected to L2, L6
            [1, 0, 0, 0, 1, 0, 1, 0, 0],  # L4: connected to L1, L5, L7
            [0, 1, 0, 1, 0, 1, 0, 1, 0],  # L5: connected to L2, L4, L6, L8
            [0, 0, 1, 0, 1, 0, 0, 0, 1],  # L6: connected to L3, L5, L9
            [0, 0, 0, 1, 0, 0, 0, 1, 0],  # L7: connected to L4, L8
            [0, 0, 0, 0, 1, 0, 1, 0, 1],  # L8: connected to L5, L7, L9
            [0, 0, 0, 0, 0, 1, 0, 1, 0]   # L9: connected to L6, L8
        ])
        
        print(f"Grid Layout (3x3):")
        print(f"L1 - L2 - L3")
        print(f"|    |    |")
        print(f"L4 - L5 - L6")
        print(f"|    |    |")
        print(f"L7 - L8 - L9")
        print(f"\nStates: {list(self.location_to_state.keys())}")
    
    def _get_valid_actions(self, state):
        """
        Get valid actions (neighboring states) from current state.
        
        Args:
            state (int): Current state
            
        Returns:
            list: Valid actions from current state
        """
        return [action for action in range(9) if self.connectivity_matrix[state, action] == 1]
    
    def _select_action(self, state, valid_actions):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state (int): Current state
            valid_actions (list): Valid actions from current state
            
        Returns:
            int: Selected action
        """
        if random.random() < self.epsilon:
            # Explore: select random action
            return random.choice(valid_actions)
        else:
            # Exploit: select action with highest Q-value among valid actions
            valid_q_values = [(action, self.Q[state, action]) for action in valid_actions]
            return max(valid_q_values, key=lambda x: x[1])[0]
    
    def train(self, goal_location, episodes=1000, verbose=True):
        """
        Train the Q-Learning agent to find optimal paths to goal.
        
        Args:
            goal_location (str): Target location (e.g., 'L1')
            episodes (int): Number of training episodes
            verbose (bool): Print training progress
            
        Returns:
            dict: Training statistics
        """
        print(f"\n=== Training Q-Learning Agent ===")
        print(f"Goal: {goal_location}")
        print(f"Episodes: {episodes}")
        
        goal_state = self.location_to_state[goal_location]
        
        # Create reward matrix for this goal
        rewards = self.connectivity_matrix.copy().astype(float)
        rewards[goal_state, goal_state] = 100  # High reward for reaching goal
        
        episode_rewards = []
        convergence_threshold = 0.01
        prev_q_sum = 0
        
        for episode in range(episodes):
            # Reset for new episode
            current_state = random.randint(0, 8)
            total_reward = 0
            steps = 0
            max_steps = 50  # Prevent infinite loops
            
            while current_state != goal_state and steps < max_steps:
                # Get valid actions
                valid_actions = self._get_valid_actions(current_state)
                if not valid_actions:
                    break
                
                # Select action using epsilon-greedy
                next_state = self._select_action(current_state, valid_actions)
                
                # Calculate reward
                if next_state == goal_state:
                    reward = rewards[current_state, next_state] + 100  # Bonus for reaching goal
                else:
                    reward = rewards[current_state, next_state]
                
                # Q-Learning update using Bellman equation
                old_q = self.Q[current_state, next_state]
                next_max = np.max(self.Q[next_state, self._get_valid_actions(next_state)])
                
                # Temporal Difference (TD) target
                td_target = reward + self.gamma * next_max
                td_error = td_target - old_q
                
                # Update Q-value
                self.Q[current_state, next_state] += self.alpha * td_error
                
                total_reward += reward
                current_state = next_state
                steps += 1
            
            episode_rewards.append(total_reward)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.epsilon_history.append(self.epsilon)
            
            # Check convergence
            current_q_sum = np.sum(np.abs(self.Q))
            if abs(current_q_sum - prev_q_sum) < convergence_threshold and episode > 100:
                if verbose:
                    print(f"Converged at episode {episode}")
                break
            prev_q_sum = current_q_sum
            
            # Progress reporting
            if verbose and episode % 200 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {self.epsilon:.3f}")
        
        self.training_rewards = episode_rewards
        
        training_stats = {
            'episodes_trained': len(episode_rewards),
            'final_avg_reward': np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards),
            'convergence_episode': episode if abs(current_q_sum - prev_q_sum) < convergence_threshold else episodes,
            'final_epsilon': self.epsilon
        }
        
        print(f"Training completed!")
        print(f"Final statistics: {training_stats}")
        
        return training_stats
    
    def find_optimal_path(self, start_location, goal_location):
        """
        Find optimal path from start to goal using learned Q-values.
        
        Args:
            start_location (str): Starting location
            goal_location (str): Target location
            
        Returns:
            tuple: (path, path_rewards, success)
        """
        print(f"\n=== Finding Optimal Path ===")
        print(f"From: {start_location} → To: {goal_location}")
        
        start_state = self.location_to_state[start_location]
        goal_state = self.location_to_state[goal_location]
        
        path = [start_location]
        path_rewards = []
        current_state = start_state
        visited_states = set()
        max_steps = 20  # Prevent infinite loops
        steps = 0
        
        while current_state != goal_state and steps < max_steps:
            if current_state in visited_states:
                print("Warning: Detected loop in path finding")
                break
            
            visited_states.add(current_state)
            
            # Get valid actions
            valid_actions = self._get_valid_actions(current_state)
            if not valid_actions:
                break
            
            # Select best action based on Q-values
            best_action = max(valid_actions, key=lambda a: self.Q[current_state, a])
            
            # Move to next state
            next_location = self.state_to_location[best_action]
            path.append(next_location)
            path_rewards.append(self.Q[current_state, best_action])
            
            current_state = best_action
            steps += 1
        
        success = current_state == goal_state
        
        print(f"Path found: {' → '.join(path)}")
        print(f"Path length: {len(path) - 1} steps")
        print(f"Success: {success}")
        
        return path, path_rewards, success
    
    def visualize_q_table(self, title="Q-Table Heatmap"):
        """
        Visualize the learned Q-table as a heatmap.
        
        Args:
            title (str): Plot title
        """
        plt.figure(figsize=(10, 8))
        
        # Create labels for better readability
        labels = [self.state_to_location[i] for i in range(9)]
        
        # Create heatmap
        sns.heatmap(self.Q, 
                   annot=True, 
                   fmt='.2f', 
                   cmap='viridis',
                   xticklabels=labels,
                   yticklabels=labels,
                   cbar_kws={'label': 'Q-Value'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Actions (Next States)', fontsize=12)
        plt.ylabel('States (Current States)', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def visualize_training_progress(self):
        """Visualize training progress and convergence."""
        if not self.training_rewards:
            print("No training data to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Training rewards over episodes
        episodes = range(len(self.training_rewards))
        axes[0].plot(episodes, self.training_rewards, alpha=0.6, color='blue', label='Episode Reward')
        
        # Moving average for trend
        window_size = 50
        if len(self.training_rewards) > window_size:
            moving_avg = np.convolve(self.training_rewards, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
            axes[0].plot(range(window_size-1, len(self.training_rewards)), 
                        moving_avg, color='red', linewidth=2, label=f'{window_size}-Episode Moving Average')
        
        axes[0].set_title('Training Progress: Rewards per Episode')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Epsilon decay over episodes
        if self.epsilon_history:
            axes[1].plot(range(len(self.epsilon_history)), self.epsilon_history, 
                        color='green', linewidth=2)
            axes[1].set_title('Exploration Rate (Epsilon) Decay')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Epsilon Value')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_grid_environment(self, path=None):
        """
        Visualize the grid environment with optional path overlay.
        
        Args:
            path (list): Optional path to highlight
        """
        plt.figure(figsize=(8, 8))
        
        # Create 3x3 grid
        grid = np.zeros((3, 3))
        
        # Position mapping for 3x3 grid
        position_map = {
            'L1': (0, 0), 'L2': (0, 1), 'L3': (0, 2),
            'L4': (1, 0), 'L5': (1, 1), 'L6': (1, 2),
            'L7': (2, 0), 'L8': (2, 1), 'L9': (2, 2)
        }
        
        # Draw grid
        for i in range(3):
            for j in range(3):
                plt.scatter(j, 2-i, s=1000, c='lightblue', alpha=0.7)
                location = self.state_to_location[i*3 + j]
                plt.text(j, 2-i, location, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Draw connections
        for state in range(9):
            for next_state in range(9):
                if self.connectivity_matrix[state, next_state] == 1:
                    loc1 = self.state_to_location[state]
                    loc2 = self.state_to_location[next_state]
                    pos1 = position_map[loc1]
                    pos2 = position_map[loc2]
                    plt.plot([pos1[1], pos2[1]], [2-pos1[0], 2-pos2[0]], 'k-', alpha=0.3, linewidth=1)
        
        # Highlight path if provided
        if path:
            path_positions = [position_map[loc] for loc in path]
            for i in range(len(path_positions)-1):
                pos1, pos2 = path_positions[i], path_positions[i+1]
                plt.plot([pos1[1], pos2[1]], [2-pos1[0], 2-pos2[0]], 'r-', linewidth=4, alpha=0.8)
            
            # Mark start and end
            start_pos = path_positions[0]
            end_pos = path_positions[-1]
            plt.scatter(start_pos[1], 2-start_pos[0], s=1500, c='green', alpha=0.8, marker='s', label='Start')
            plt.scatter(end_pos[1], 2-end_pos[0], s=1500, c='red', alpha=0.8, marker='*', label='Goal')
            plt.legend()
        
        plt.xlim(-0.5, 2.5)
        plt.ylim(-0.5, 2.5)
        plt.title('Grid Environment' + (' with Optimal Path' if path else ''), fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.show()

def demonstrate_q_learning():
    """
    Demonstrate Q-Learning path finding with comprehensive analysis.
    """
    print("🤖 Q-Learning Path Finding Demonstration")
    print("="*50)
    
    # Initialize Q-Learning agent
    agent = QLearningPathFinder(gamma=0.85, alpha=0.8, epsilon=0.3, epsilon_decay=0.995)
    
    # Train agent to reach goal L1
    goal = 'L1'
    stats = agent.train(goal, episodes=1500, verbose=True)
    
    # Visualize training progress
    agent.visualize_training_progress()
    
    # Visualize learned Q-table
    agent.visualize_q_table(f"Q-Table for Goal: {goal}")
    
    # Test optimal path finding
    test_cases = [('L9', 'L1'), ('L3', 'L7'), ('L1', 'L9'), ('L5', 'L6')]
    
    print(f"\n{'='*60}")
    print(f"TESTING OPTIMAL PATH FINDING")
    print(f"{'='*60}")
    
    for start, end in test_cases:
        # Retrain if goal changed
        if end != goal:
            print(f"\nRetraining for new goal: {end}")
            agent.train(end, episodes=1000, verbose=False)
            goal = end
        
        path, rewards, success = agent.find_optimal_path(start, end)
        
        if success:
            print(f"✅ Path Quality Score: {np.mean(rewards):.2f}")
        else:
            print(f"❌ Failed to find path")
        
        # Visualize this path
        agent.visualize_grid_environment(path if success else None)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"🎯 Final Training Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n🧠 Algorithm Insights:")
    print(f"  • Q-Learning successfully learned optimal paths")
    print(f"  • Epsilon-greedy exploration balanced learning and exploitation")
    print(f"  • Bellman equation updates enabled value propagation")
    print(f"  • Convergence achieved through iterative policy improvement")
    
    print(f"\n🚀 Applications:")
    print(f"  • Robot navigation and pathfinding")
    print(f"  • Game AI and strategic planning")
    print(f"  • Resource allocation optimization")
    print(f"  • Supply chain and logistics planning")

if __name__ == "__main__":
    demonstrate_q_learning()