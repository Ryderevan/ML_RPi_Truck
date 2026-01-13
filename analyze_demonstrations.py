"""
Analyze demonstrations data before training a model.
Opens demonstrations-evan.npz and visualizes the relationship between
steering commands and heading error.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_demonstration_data(filename='demonstrations-evan.npz'):
    """Load demonstration data from .npz file."""
    try:
        data = np.load(filename)
        states = data['states']
        actions = data['actions']
        
        print(f"✓ Loaded data from {filename}")
        print(f"  Total transitions: {len(states)}")
        print(f"  State shape: {states.shape}")
        print(f"  Action range: [{actions.min():.0f}, {actions.max():.0f}] us")
        
        return states, actions
    
    except FileNotFoundError:
        print(f"✗ File not found: {filename}")
        return None, None

def analyze_data(states, actions):
    """Analyze and print data statistics."""
    print("\n" + "="*50)
    print("DATA ANALYSIS")
    print("="*50)
    
    # State analysis
    heading_error = states[:, 0]
    heading_error_derivative = states[:, 1]
    
    print("\nHeading Error (state[0]):")
    print(f"  Range: [{heading_error.min():.4f}, {heading_error.max():.4f}]")
    print(f"  Mean: {heading_error.mean():.4f}")
    print(f"  Std: {heading_error.std():.4f}")
    
    print("\nHeading Error Rate (state[1]):")
    print(f"  Range: [{heading_error_derivative.min():.4f}, {heading_error_derivative.max():.4f}]")
    print(f"  Mean: {heading_error_derivative.mean():.4f}")
    print(f"  Std: {heading_error_derivative.std():.4f}")
    
    print("\nSteering Command (actions):")
    print(f"  Range: [{actions.min():.0f}, {actions.max():.0f}] us")
    print(f"  Mean: {actions.mean():.0f}")
    print(f"  Std: {actions.std():.0f}")
    print(f"  Unique values: {len(np.unique(actions))}")

def plot_analysis(states, actions):
    """Create plots to visualize relationships in the data."""
    heading_error = states[:, 0]
    heading_error_derivative = states[:, 1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Demonstrations Data Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Steering Command vs Heading Error
    ax = axes[0]
    ax.scatter(heading_error, actions, alpha=0.6, s=20, color='steelblue')
    ax.set_xlabel('Heading Error (radians)', fontsize=11)
    ax.set_ylabel('Steering Command (µs)', fontsize=11)
    ax.set_title('Steering Command vs Heading Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Steering Command vs Heading Error Rate
    ax = axes[1]
    ax.scatter(heading_error_derivative, actions, alpha=0.6, s=20, color='darkorange')
    ax.set_xlabel('Heading Error Rate (rad/s)', fontsize=11)
    ax.set_ylabel('Steering Command (µs)', fontsize=11)
    ax.set_title('Steering Command vs Heading Error Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demonstrations_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved as 'demonstrations_analysis.png'")
    plt.show()

def main():
    """Main analysis function."""
    print("="*50)
    print("DEMONSTRATIONS DATA ANALYZER")
    print("="*50)
    
    # Load data
    states, actions = load_demonstration_data('demonstrations-evan.npz')
    
    if states is None:
        return
    
    # Analyze data
    analyze_data(states, actions)
    
    # Create plots
    print("\n" + "="*50)
    print("GENERATING PLOTS")
    print("="*50)
    plot_analysis(states, actions)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)

if __name__ == '__main__':
    main()
