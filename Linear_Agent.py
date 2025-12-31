"""
Linear Regression Agent for Steering Control
Simple linear model: steering_us = bias + w1*heading_error + w2*error_dot

State representation:
    - heading_error: difference between current heading and target heading (degrees)
    - error_dot: rate of change of heading (degrees/second)
"""
import numpy as np
import joblib


class LinearAgent:
    """Linear regression model for predicting steering commands from state.
    
    Loads scikit-learn trained models saved as .pkl files.
    """
    
    def __init__(self, state_dim=2):
        """Initialize linear agent.
        
        Args:
            state_dim: dimension of state space (2: heading_error, error_dot)
        """
        self.state_dim = state_dim
        self.model = None  # Will hold the scikit-learn LinearRegression model
        self.bias = 1500.0  # Default to neutral steering
        self.weights = np.zeros(state_dim)
    
    def predict(self, state):
        """Predict steering command for given state.
        
        Args:
            state: state array [heading_error, error_dot] or [N, 2]
            
        Returns:
            steering_us: predicted steering command in microseconds (1000-2000)
        """
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Use scikit-learn model if loaded, otherwise use weights
        if self.model is not None:
            prediction = self.model.predict(state)
        else:
            prediction = self.bias + np.dot(state, self.weights)
        
        # Clip to valid steering range
        prediction = np.clip(prediction, 1000, 2000)
        
        if prediction.shape[0] == 1:
            return float(prediction[0])
        return prediction
    
    def load_model(self, filename='linear_model.pkl'):
        """Load scikit-learn model from file.
        
        Args:
            filename: path to model file (.pkl format from train_offline.ipynb)
        """
        self.model = joblib.load(filename)
        self.bias = float(self.model.intercept_)
        self.weights = self.model.coef_
        print(f"Linear model loaded from {filename}")
        print(f"  Intercept (bias): {self.bias:.2f}")
        print(f"  Coefficients: {self.weights}")
    
    def save_model(self, filename='linear_model.pkl'):
        """Save model to file using joblib.
        
        Args:
            filename: path to save model (.pkl format)
        """
        if self.model is None:
            raise ValueError("No model loaded. Load a model first before saving.")
        joblib.dump(self.model, filename)
        print(f"Linear model saved to {filename}")
