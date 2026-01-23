#!/usr/bin/env python3
"""
Offline Training Script for Linear Regression Model
Trains linear steering model.

This script:
1. Loads demonstration data from demonstrations.npz
2. Trains a linear regression model
3. Visualizes the results
4. Saves the model for deployment
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import sys


def load_demonstration_data(filename='demonstrations.npz'):
    """Load demonstration data from single file.
    
    Args:
        filename: path to .npz file containing states and actions
        
    Returns:
        states: array of shape [N, 2] (heading_error, error_dot)
        actions: array of shape [N] (steering commands in microseconds)
    """
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
        print("Run Vehicle.py in mode 2 first to collect data.")
        return None, None


class RegressionModel:
    """Wrapper around scikit-learn's LinearRegression for steering prediction.
    
    Model: steering_us = w0 + w1*heading_error + w2*error_dot
    """
    def __init__(self, state_dim=2):
        self.state_dim = state_dim
        self.model = LinearRegression()
        self.predictions = None
        
    def fit(self, states, actions):
        """Fit linear regression using scikit-learn.
        
        Args:
            states: array of states [N, state_dim] (heading_error, error_dot)
            actions: array of steering commands [N] (1000-2000 us)
            
        Returns:
            predictions: model predictions on training data
        """
        # Fit the model
        self.model.fit(states, actions)
        
        # Get predictions
        self.predictions = self.model.predict(states)
        
        # Clip predictions to valid steering range
        self.predictions = np.clip(self.predictions, 1000, 2000)
        
        # Compute metrics
        mse = mean_squared_error(actions, self.predictions)
        mae = mean_absolute_error(actions, self.predictions)
        r2 = r2_score(actions, self.predictions)
        
        print(f"\nLinear Regression Training Complete:")
        print(f"  Intercept (bias): {self.model.intercept_:.2f}")
        print(f"  Coefficients: {self.model.coef_}")
        print(f"  MSE: {mse:.2f} us^2")
        print(f"  MAE: {mae:.2f} us")
        print(f"  R² score: {r2:.4f}")
        print(f"\nModel equation:")
        print(f"  steering = {self.model.intercept_:.2f} + {self.model.coef_[0]:.4f}*heading_error + {self.model.coef_[1]:.4f}*error_dot")
        
        return self.predictions
    
    def predict(self, states):
        """Predict steering commands for given states.
        
        Args:
            states: array of states [N, state_dim] or [state_dim]
            
        Returns:
            predictions: predicted steering commands
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        predictions = self.model.predict(states)
        
        # Clip to valid steering range
        predictions = np.clip(predictions, 1000, 2000)
        
        if len(predictions) == 1:
            return predictions[0]
        return predictions
    
    def save(self, filename='linear_model.pkl'):
        """Save model to file using joblib.
        
        Args:
            filename: output filename for the model
        """
        joblib.dump(self.model, filename)
        print(f"✓ Model saved to {filename}")


def visualize_model_fit(states, actions, predictions):
    """Visualize model fit and performance.
    
    Args:
        states: array of states
        actions: array of actual actions
        predictions: array of predicted actions
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Linear Regression Model Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Steering vs Heading Error
    ax = axes[0, 0]
    ax.scatter(states[:, 0], actions, alpha=0.3, s=10, label='Actual', c='blue')
    ax.scatter(states[:, 0], predictions, alpha=0.3, s=10, label='Predicted', c='red')
    ax.set_xlabel('Heading Error (radians)')
    ax.set_ylabel('Steering (µs)')
    ax.set_title('Steering vs Heading Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Steering vs Error Rate
    ax = axes[0, 1]
    ax.scatter(states[:, 1], actions, alpha=0.3, s=10, label='Actual', c='blue')
    ax.scatter(states[:, 1], predictions, alpha=0.3, s=10, label='Predicted', c='red')
    ax.set_xlabel('Error Rate (rad/s)')
    ax.set_ylabel('Steering (µs)')
    ax.set_title('Steering vs Error Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Predicted vs Actual
    ax = axes[1, 0]
    ax.scatter(actions, predictions, alpha=0.3, s=10)
    ax.plot([1000, 2000], [1000, 2000], 'r--', label='Perfect fit', linewidth=2)
    ax.set_xlabel('Actual Steering (µs)')
    ax.set_ylabel('Predicted Steering (µs)')
    ax.set_title('Predicted vs Actual Steering')
    ax.set_xlim([1000, 2000])
    ax.set_ylim([1000, 2000])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 4: Residuals
    ax = axes[1, 1]
    residuals = predictions - actions
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='purple')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.set_xlabel('Prediction Error (µs)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residuals (Mean: {np.mean(residuals):.1f} µs, Std: {np.std(residuals):.1f} µs)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_fit_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved model fit plot to 'model_fit_results.png'")
    plt.show()


def print_summary(states, actions, predictions):
    """Print training summary and statistics.
    
    Args:
        states: array of states
        actions: array of actual actions
        predictions: array of predicted actions
    """
    residuals = predictions - actions
    mse = mean_squared_error(actions, predictions)
    mae = mean_absolute_error(actions, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actions, predictions)
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(states)}")
    print(f"  State dimension: {states.shape[1]}")
    print(f"\nModel Performance Metrics:")
    print(f"  R² Score:               {r2:.4f}")
    print(f"  Mean Squared Error:     {mse:.2f} µs²")
    print(f"  Root Mean Squared Error: {rmse:.2f} µs")
    print(f"  Mean Absolute Error:    {mae:.2f} µs")
    print(f"\nResidual Statistics:")
    print(f"  Mean: {np.mean(residuals):.2f} µs")
    print(f"  Std:  {np.std(residuals):.2f} µs")
    print(f"  Min:  {np.min(residuals):.2f} µs")
    print(f"  Max:  {np.max(residuals):.2f} µs")
    print("\n" + "="*60)


def main():
    """Main training pipeline."""
    print("="*60)
    print("LINEAR REGRESSION OFFLINE TRAINING")
    print("="*60)
    
    # Step 1: Load data
    print("\nStep 1: Loading demonstration data...")
    states, actions = load_demonstration_data('demonstrations.npz')
    
    if states is None:
        print("\n✗ Failed to load demonstration data!")
        sys.exit(1)
    
    # Step 2: Create and train model
    print("\nStep 2: Creating and training linear regression model...")
    model = RegressionModel(state_dim=2)
    predictions = model.fit(states, actions)
    
    # Step 3: Visualize model fit
    print("\nStep 3: Visualizing model fit...")
    visualize_model_fit(states, actions, predictions)
    
    # Step 4: Print summary
    print_summary(states, actions, predictions)
    
    # Step 5: Save model
    print("\nStep 5: Saving trained model...")
    model.save('linear_model.pkl')
    
    # Print deployment instructions
    print("\n" + "="*60)
    print("DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    print("\n1. In Vehicle.py, set drive_mode to 3 for inference")
    print("\n2. Run BajaRey.py and observe the model's performance!")
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
