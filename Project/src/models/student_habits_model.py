"""
Student Habits Model for Grade Prediction.

This module trains a model on the 80K student habits dataset to learn
the relationship between student behaviors and academic performance.
It provides feature coefficients that can be combined with professor
review sentiment for comprehensive grade prediction.

Author: Enhanced for Resume Portfolio
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging
import joblib
from typing import Dict, Any, Tuple, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentHabitsModel:
    """
    Model to predict academic performance from student habits.
    
    Uses 3 key features for simplicity (web app friendly):
    - study_hours_per_day: Daily study time (0-8+ hours)
    - previous_gpa: Prior academic performance (0.0-4.0)
    - motivation_level: Self-reported motivation (1-10)
    
    Target: exam_score (converted to GPA scale 0-4.0)
    """
    
    # Core features for the model (limited to 3 for web UI simplicity)
    CORE_FEATURES = ['study_hours_per_day', 'previous_gpa', 'motivation_level']
    
    # Feature boundaries for validation and UI
    FEATURE_BOUNDS = {
        'study_hours_per_day': (0, 12),
        'previous_gpa': (0.0, 4.0),
        'motivation_level': (1, 10)
    }
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = self.CORE_FEATURES.copy()
        self.training_metrics = {}
        self._is_trained = False
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load the student habits dataset."""
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} student records from {filepath}")
        return df
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for training or prediction.
        
        Args:
            df: DataFrame with student data
            is_training: Whether this is training data
            
        Returns:
            Tuple of (X, y)
        """
        # Extract features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean() if is_training else 0)
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Get target (convert exam_score 0-100 to GPA 0-4.0 scale)
        y = None
        if 'exam_score' in df.columns:
            # Map 0-100 to 0-4.0 (simple linear mapping)
            y = (df['exam_score'] / 100) * 4.0
        
        return X_scaled, y
    
    def train(self, filepath: str = "datasets/enhanced_student_habits_performance_dataset.csv",
              sample_size: int = 10000) -> Dict[str, Any]:
        """
        Train the student habits model.
        
        Args:
            filepath: Path to the habits dataset
            sample_size: Number of samples to use (for faster training)
            
        Returns:
            Dictionary of training metrics
        """
        # Load data
        df = self.load_dataset(filepath)
        
        # Sample for faster training (80K is a lot)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records for training")
        
        # Prepare features
        X, y = self.prepare_features(df, is_training=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        logger.info("Training student habits model...")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X, y, cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        self.training_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred),
            'cv_rmse': cv_rmse,
            'n_samples': len(df),
            'feature_importance': dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
        }
        
        self._is_trained = True
        
        logger.info(f"\n{'='*50}")
        logger.info("Student Habits Model Results:")
        logger.info(f"  Test RMSE: {self.training_metrics['test_rmse']:.4f}")
        logger.info(f"  Test R²: {self.training_metrics['test_r2']:.4f}")
        logger.info(f"  CV RMSE: {cv_rmse:.4f}")
        logger.info(f"\nFeature Importance:")
        for feat, imp in self.training_metrics['feature_importance'].items():
            logger.info(f"  {feat}: {imp:.4f}")
        logger.info(f"{'='*50}")
        
        return self.training_metrics
    
    def predict_gpa(self, study_hours: float, prior_gpa: float, 
                   motivation: int) -> Dict[str, Any]:
        """
        Predict expected GPA from student habits.
        
        Args:
            study_hours: Daily study hours (0-12)
            prior_gpa: Previous GPA (0.0-4.0)
            motivation: Motivation level (1-10)
            
        Returns:
            Dictionary with predicted GPA and letter grade
        """
        if not self._is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Validate inputs
        study_hours = np.clip(study_hours, *self.FEATURE_BOUNDS['study_hours_per_day'])
        prior_gpa = np.clip(prior_gpa, *self.FEATURE_BOUNDS['previous_gpa'])
        motivation = np.clip(motivation, *self.FEATURE_BOUNDS['motivation_level'])
        
        # Create input
        X = pd.DataFrame([{
            'study_hours_per_day': study_hours,
            'previous_gpa': prior_gpa,
            'motivation_level': motivation
        }])
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        pred_gpa = self.model.predict(X_scaled)[0]
        
        # Clip to valid range
        pred_gpa = np.clip(pred_gpa, 0.0, 4.0)
        
        # Convert to letter grade
        grade_mapping = {
            4.0: 'A', 3.7: 'A-', 3.3: 'B+', 3.0: 'B', 2.7: 'B-',
            2.3: 'C+', 2.0: 'C', 1.7: 'C-', 1.3: 'D+', 1.0: 'D',
            0.7: 'D-', 0.0: 'F'
        }
        letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - pred_gpa))[1]
        
        return {
            'predicted_gpa': round(pred_gpa, 2),
            'predicted_grade': letter,
            'inputs': {
                'study_hours': study_hours,
                'prior_gpa': prior_gpa,
                'motivation': motivation
            }
        }
    
    def get_feature_impact(self, feature: str, value: float) -> str:
        """Get human-readable impact of a feature value."""
        if feature == 'study_hours_per_day':
            if value < 2:
                return "Low study effort - consider increasing"
            elif value < 4:
                return "Moderate study effort"
            else:
                return "Strong study commitment"
        elif feature == 'previous_gpa':
            if value < 2.0:
                return "Academic risk - may struggle"
            elif value < 3.0:
                return "Average baseline"
            else:
                return "Strong academic foundation"
        elif feature == 'motivation_level':
            if value < 4:
                return "Low motivation - major concern"
            elif value < 7:
                return "Moderate motivation"
            else:
                return "Highly motivated"
        return ""
    
    def save_model(self, path: str = "models/student_habits_model.pkl"):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'training_metrics': self.training_metrics
        }
        joblib.dump(model_data, path)
        logger.info(f"Student habits model saved to {path}")
    
    def load_model(self, path: str = "models/student_habits_model.pkl"):
        """Load a trained model."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.training_metrics = model_data.get('training_metrics', {})
        self._is_trained = True
        logger.info(f"Student habits model loaded from {path}")


if __name__ == "__main__":
    # Quick test
    model = StudentHabitsModel()
    metrics = model.train()
    
    # Test prediction
    result = model.predict_gpa(
        study_hours=4.0,
        prior_gpa=3.2,
        motivation=7
    )
    print(f"\nTest Prediction: {result}")
    
    # Save model
    model.save_model()
