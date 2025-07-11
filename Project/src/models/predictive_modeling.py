import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import logging
from typing import Tuple, Dict, Any
import pickle
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.base_features = ['rating', 'difficulty', 'sentiment_score']
        self.feature_columns = self.base_features.copy()
        self.professor_features = ['review_count', 'avg_rating', 'avg_difficulty']
        
    def calculate_sentiment_score(self, text):
        """Calculate sentiment score for a given text."""
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
        
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True):
        """
        Prepare features for model training or prediction.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is for training (True) or prediction (False)
            
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is target Series
        """
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # Calculate sentiment scores
        logger.info("Calculating sentiment scores...")
        df_features['sentiment_score'] = df_features['cleaned_text'].apply(self.calculate_sentiment_score)
        
        # Create professor-level features if professor_name is available
        if 'professor_name' in df_features.columns:
            prof_stats = df_features.groupby('professor_name').agg({
                'rating': ['count', 'mean'],
                'difficulty': 'mean'
            }).reset_index()
            prof_stats.columns = ['professor_name', 'review_count', 'avg_rating', 'avg_difficulty']
            
            # Merge back to main DataFrame
            df_features = df_features.merge(prof_stats, on='professor_name', how='left')
            
            # Add these to feature columns if in training mode
            if is_training:
                self.feature_columns = self.base_features + self.professor_features
        
        # Convert grade to numerical values if needed
        if 'grade' in df_features.columns:
            grade_mapping = {
                'A+': 4.0, 'A': 4.0, 'A-': 3.7,
                'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                'C+': 2.3, 'C': 2.0, 'C-': 1.7,
                'D+': 1.3, 'D': 1.0, 'D-': 0.7,
                'F': 0.0
            }
            df_features['grade_numerical'] = df_features['grade'].map(grade_mapping)
        
        # Add missing professor-level features if needed
        for feature in self.feature_columns:
            if feature not in df_features.columns:
                df_features[feature] = 0.0  # Use default value for missing features
        
        # Prepare feature matrix
        X = df_features[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_columns)
        
        if 'grade_numerical' in df_features.columns:
            y = df_features['grade_numerical'].fillna(df_features['grade_numerical'].mean())
            return X_scaled, y
        else:
            return X_scaled, None
    
    def train(self, train_df: pd.DataFrame):
        """
        Train the grade prediction model.
        
        Args:
            train_df: Training DataFrame
        """
        logger.info("Preparing features for training...")
        X, y = self.prepare_features(train_df, is_training=True)
        
        # Drop rows with missing target values
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        logger.info("Training model...")
        self.model.fit(X, y)
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_rmse = np.sqrt(np.mean((y - train_pred) ** 2))
        logger.info(f"Training RMSE: {train_rmse:.3f}")
    
    def evaluate_model(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary containing evaluation metrics
        """
        X_test, y_test = self.prepare_features(test_df, is_training=False)
        
        # Drop rows with missing target values
        mask = ~y_test.isna()
        X_test = X_test[mask]
        y_test = y_test[mask]
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = self.model.score(X_test, y_test)
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Test RMSE: {rmse:.3f}")
        logger.info(f"Test MAE: {mae:.3f}")
        logger.info(f"Test R2: {r2:.3f}")
        
        return metrics
    
    def predict_grade(self, features_dict: dict) -> tuple:
        """
        Predict grade for new data.
        
        Args:
            features_dict: Dictionary containing feature values
            
        Returns:
            Tuple of (predicted_grade, confidence_interval)
        """
        # Create DataFrame from features
        df = pd.DataFrame([features_dict])
        
        # Prepare features
        X, _ = self.prepare_features(df, is_training=False)
        
        # Make prediction
        pred = self.model.predict(X)[0]
        
        # Calculate confidence interval using prediction std
        predictions = []
        for estimator in self.model.estimators_:
            predictions.append(estimator.predict(X)[0])
        
        ci_lower = np.percentile(predictions, 2.5)
        ci_upper = np.percentile(predictions, 97.5)
        
        # Convert numerical grade to letter grade
        grade_mapping = {
            4.0: 'A',
            3.7: 'A-',
            3.3: 'B+',
            3.0: 'B',
            2.7: 'B-',
            2.3: 'C+',
            2.0: 'C',
            1.7: 'C-',
            1.3: 'D+',
            1.0: 'D',
            0.7: 'D-',
            0.0: 'F'
        }
        
        # Find closest grade
        pred_letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - pred))[1]
        ci_lower_letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - ci_lower))[1]
        ci_upper_letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - ci_upper))[1]
        
        return (pred_letter, (ci_lower_letter, ci_upper_letter)), (pred, (ci_lower, ci_upper))
    
    def save_model(self, model_path: str):
        """Save the trained model to disk."""
        joblib.dump((self.model, self.scaler, self.feature_columns), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        """Load a trained model from disk."""
        self.model, self.scaler, self.feature_columns = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
            
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores from the model.
        
        Returns:
            DataFrame containing feature names and their importance scores
        """
        if not hasattr(self, 'model') or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained or does not support feature importance")
            
        importance_scores = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        })
        
        # Sort by importance in descending order
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance

def main():
    # Example usage
    import pandas as pd
    
    # Load processed data
    train_data = pd.read_csv("data/processed/processed_training_reviews.csv")
    test_data = pd.read_csv("data/processed/processed_test_reviews.csv")
    
    # Initialize and train model
    predictor = GradePredictor()
    predictor.train(train_data)
    
    # Evaluate model
    metrics = predictor.evaluate_model(test_data)
    
    # Save model
    predictor.save_model("models/grade_predictor.pkl")
    
    # Example prediction
    sample_features = {
        'rating': 4.5,
        'difficulty': 3.0,
        'cleaned_text': "The professor was very helpful and explained concepts clearly."
    }
    
    (pred_letter, ci_letter), (pred_num, ci_num) = predictor.predict_grade(sample_features)
    logger.info(f"Predicted grade (letter): {pred_letter}")
    logger.info(f"95% Confidence Interval (letter): ({ci_letter[0]}, {ci_letter[1]})")
    logger.info(f"Predicted grade (numerical): {pred_num:.2f}")
    logger.info(f"95% Confidence Interval (numerical): ({ci_num[0]:.2f}, {ci_num[1]:.2f})")

if __name__ == "__main__":
    main() 