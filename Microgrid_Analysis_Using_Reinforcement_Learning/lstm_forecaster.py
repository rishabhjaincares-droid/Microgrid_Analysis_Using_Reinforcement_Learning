"""
LSTM Solar Irradiance Forecaster
=================================
Predicts solar irradiance 30 minutes (or 1-2 timesteps) ahead using LSTM.

This model is trained separately and then integrated into the RL agent's state.
Forecast acts as a "window into the future" helping the agent make better decisions.

Requires: tensorflow, keras
Install: pip install tensorflow
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os


class LSTMSolarForecaster:
    """
    LSTM model for solar irradiance time series forecasting.
    
    Architecture:
    - Input: 24 hours of historical solar data
    - Hidden: 64-unit LSTM layer
    - Output: Solar irradiance 30 minutes (or multiple hours) ahead
    """
    
    def __init__(self, 
                 lookback_hours: int = 24,
                 forecast_hours: int = 1,
                 lstm_units: int = 64):
        """
        Initialize LSTM forecaster.
        
        Args:
            lookback_hours: Historical window size (hours)
            forecast_hours: Ahead forecast horizon (hours)
            lstm_units: Number of LSTM units
        """
        self.lookback_hours = lookback_hours
        self.forecast_hours = forecast_hours
        self.lstm_units = lstm_units
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
        print(f"LSTM Forecaster initialized:")
        print(f"  Lookback: {lookback_hours}h")
        print(f"  Forecast horizon: {forecast_hours}h")
        print(f"  LSTM units: {lstm_units}")
    
    def build_model(self):
        """Build LSTM architecture."""
        self.model = models.Sequential([
            layers.LSTM(
                self.lstm_units,
                activation='relu',
                input_shape=(self.lookback_hours, 1),
                return_sequences=True
            ),
            layers.Dropout(0.2),
            
            layers.LSTM(
                self.lstm_units // 2,
                activation='relu',
                return_sequences=False
            ),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.Dense(self.forecast_hours, activation='relu')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse',
            metrics=['mae']
        )
        
        print("\nModel architecture:")
        self.model.summary()
    
    def prepare_data(self, 
                    solar_data: np.ndarray,
                    train_split: float = 0.8) -> Tuple[np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training.
        
        Creates sequences of (input, target) pairs where:
        - Input: 24 consecutive hours of solar irradiance
        - Target: Solar irradiance 1 hour ahead
        
        Args:
            solar_data: Time series of solar irradiance values
            train_split: Fraction for training (0.8 = 80% train, 20% val)
            
        Returns:
            (X_train, y_train, X_val, y_val)
        """
        print(f"\nPreparing data for training...")
        print(f"  Total samples: {len(solar_data)}")
        
        # Normalize data
        solar_scaled = self.scaler.fit_transform(solar_data.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(len(solar_scaled) - self.lookback_hours - self.forecast_hours):
            X.append(solar_scaled[i:i + self.lookback_hours])
            y.append(solar_scaled[i + self.lookback_hours:i + self.lookback_hours + self.forecast_hours])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).squeeze()
        
        print(f"  Sequences created: {len(X)}")
        print(f"  Input shape: {X.shape}")
        print(f"  Output shape: {y.shape}")
        
        # Split into train/validation
        split_idx = int(len(X) * train_split)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        
        return X_train, y_train, X_val, y_val
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              epochs: int = 50,
              batch_size: int = 32,
              verbose: int = 1) -> Dict:
        """
        Train LSTM model.
        
        Args:
            X_train: Training input sequences
            y_train: Training targets
            X_val: Validation input sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history dictionary
        """
        print(f"\nTraining LSTM model...")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
        )
        
        return self.history.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test input sequences
            y_test: Test targets
            
        Returns:
            Metrics dictionary
        """
        print("\nEvaluating model...")
        
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions
        y_pred = self.model.predict(X_test, verbose=0)
        
        # Inverse transform to original scale
        y_pred_original = self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_original = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Metrics
        mse = np.mean((y_test_original - y_pred_original) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_original - y_pred_original))
        
        # R² score
        ss_res = np.sum((y_test_original - y_pred_original) ** 2)
        ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        metrics = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Test MAE: {test_mae:.6f}")
        print(f"  RMSE: {rmse:.4f} W/m²")
        print(f"  MAE: {mae:.4f} W/m²")
        print(f"  R²: {r2:.4f}")
        
        return metrics
    
    def predict(self, solar_history: np.ndarray) -> float:
        """
        Predict next solar irradiance value.
        
        Args:
            solar_history: Last 24 hours of solar irradiance (normalized to [0,1])
            
        Returns:
            Predicted solar irradiance (normalized to [0,1])
        """
        if len(solar_history) != self.lookback_hours:
            raise ValueError(f"Expected {self.lookback_hours} values, got {len(solar_history)}")
        
        X = solar_history.reshape(1, self.lookback_hours, 1)
        prediction = self.model.predict(X, verbose=0)[0][0]
        
        return float(prediction)
    
    def save(self, filepath: str = 'lstm_solar_forecaster.h5'):
        """Save model to disk."""
        self.model.save(filepath)
        
        # Save scaler parameters
        scaler_params = {
            'data_min': float(self.scaler.data_min_[0]),
            'data_max': float(self.scaler.data_max_[0]),
            'lookback_hours': self.lookback_hours,
            'forecast_hours': self.forecast_hours,
            'lstm_units': self.lstm_units,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath.replace('.h5', '_params.json'), 'w') as f:
            json.dump(scaler_params, f, indent=2)
        
        print(f"✓ Model saved to {filepath}")
        print(f"✓ Parameters saved to {filepath.replace('.h5', '_params.json')}")
    
    def load(self, filepath: str = 'lstm_solar_forecaster.h5'):
        """Load model from disk."""
        self.model = keras.models.load_model(filepath)
        
        # Load scaler parameters
        params_file = filepath.replace('.h5', '_params.json')
        if os.path.exists(params_file):
            with open(params_file, 'r') as f:
                params = json.load(f)
            
            self.scaler.data_min_ = np.array([params['data_min']])
            self.scaler.data_max_ = np.array([params['data_max']])
            self.lookback_hours = params['lookback_hours']
            self.forecast_hours = params['forecast_hours']
            self.lstm_units = params['lstm_units']
        
        print(f"✓ Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: str = 'lstm_training_history.png'):
        """Plot training and validation loss."""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Training History - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        axes[1].plot(self.history.history['mae'], label='Training MAE')
        axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE (W/m²)')
        axes[1].set_title('Training History - MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"✓ Training history plot saved to {save_path}")
        plt.close()


def train_lstm_forecaster(solar_data: np.ndarray, 
                         model_name: str = 'lstm_solar_forecaster'):
    """
    End-to-end training of LSTM solar forecaster.
    
    Args:
        solar_data: Time series of solar irradiance values
        model_name: Name for saving model
    """
    print("=" * 60)
    print("Training LSTM Solar Irradiance Forecaster")
    print("=" * 60)
    
    # Initialize forecaster
    forecaster = LSTMSolarForecaster(
        lookback_hours=24,
        forecast_hours=1,
        lstm_units=64
    )
    
    # Build model
    forecaster.build_model()
    
    # Prepare data
    X_train, y_train, X_val, y_val = forecaster.prepare_data(solar_data, train_split=0.8)
    
    # Train
    history = forecaster.train(
        X_train, y_train, X_val, y_val,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate
    metrics = forecaster.evaluate(X_val, y_val)
    
    # Save
    forecaster.save(f'{model_name}.h5')
    forecaster.plot_training_history(f'{model_name}_history.png')
    
    return forecaster, metrics


if __name__ == "__main__":
    print("Testing LSTM Solar Forecaster...")
    
    # Create synthetic solar data (same as in demo environment)
    num_days = 365
    num_hours = num_days * 24
    hour_of_day = np.arange(num_hours) % 24
    
    # Solar profile: peaks at noon
    solar_data = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12) * 600)  # 0-600 W/m²
    
    # Add realistic noise
    solar_data += np.random.normal(0, 30, len(solar_data))
    solar_data = np.clip(solar_data, 0, 800)
    
    print(f"\nSynthetic solar data:")
    print(f"  Shape: {solar_data.shape}")
    print(f"  Min: {solar_data.min():.1f} W/m²")
    print(f"  Max: {solar_data.max():.1f} W/m²")
    print(f"  Mean: {solar_data.mean():.1f} W/m²")
    
    # Train forecaster
    forecaster, metrics = train_lstm_forecaster(solar_data, 'test_lstm_forecaster')
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved: test_lstm_forecaster.h5")
    print(f"Metrics: {metrics}")
