import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

class GradePredictorGB:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.base_features = ['rating', 'difficulty', 'sentiment_score']
        self.feature_columns = self.base_features.copy()
        self.professor_features = ['review_count', 'avg_rating', 'avg_difficulty']
    
    def calculate_sentiment_score(self, text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True):
        df_features = df.copy()
        logger.info("Calculating sentiment scores...")
        df_features['sentiment_score'] = df_features['cleaned_text'].apply(self.calculate_sentiment_score)
        if 'professor_name' in df_features.columns:
            prof_stats = df_features.groupby('professor_name').agg({
                'rating': ['count', 'mean'],
                'difficulty': 'mean'
            }).reset_index()
            prof_stats.columns = ['professor_name', 'review_count', 'avg_rating', 'avg_difficulty']
            df_features = df_features.merge(prof_stats, on='professor_name', how='left')
            if is_training:
                self.feature_columns = self.base_features + self.professor_features
        if 'grade' in df_features.columns:
            grade_mapping = {
                'A+': 4.0, 'A': 4.0, 'A-': 3.7,
                'B+': 3.3, 'B': 3.0, 'B-': 2.7,
                'C+': 2.3, 'C': 2.0, 'C-': 1.7,
                'D+': 1.3, 'D': 1.0, 'D-': 0.7,
                'F': 0.0
            }
            df_features['grade_numerical'] = df_features['grade'].map(grade_mapping)
        for feature in self.feature_columns:
            if feature not in df_features.columns:
                df_features[feature] = 0.0
        X = df_features[self.feature_columns].copy()
        X = X.fillna(X.mean())
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
        logger.info("Preparing features for training...")
        X, y = self.prepare_features(train_df, is_training=True)
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        logger.info("Training Gradient Boosting model...")
        self.model.fit(X, y)
        train_pred = self.model.predict(X)
        train_rmse = np.sqrt(np.mean((y - train_pred) ** 2))
        logger.info(f"Training RMSE: {train_rmse:.3f}")
    
    def evaluate_model(self, test_df: pd.DataFrame) -> dict:
        X_test, y_test = self.prepare_features(test_df, is_training=False)
        mask = ~y_test.isna()
        X_test = X_test[mask]
        y_test = y_test[mask]
        y_pred = self.model.predict(X_test)
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
        df = pd.DataFrame([features_dict])
        X, _ = self.prepare_features(df, is_training=False)
        pred = self.model.predict(X)[0]
        # Confidence interval using staged predictions (approximate)
        # For GB, we can use staged_predict for an ensemble-like CI
        staged_preds = list(self.model.staged_predict(X))
        if len(staged_preds) > 10:
            preds = np.array(staged_preds[-10:])[:, 0]
        else:
            preds = np.array(staged_preds)[:, 0]
        ci_lower = np.percentile(preds, 2.5) if len(preds) > 1 else pred
        ci_upper = np.percentile(preds, 97.5) if len(preds) > 1 else pred
        grade_mapping = {
            4.0: 'A', 3.7: 'A-', 3.3: 'B+', 3.0: 'B', 2.7: 'B-', 2.3: 'C+', 2.0: 'C', 1.7: 'C-', 1.3: 'D+', 1.0: 'D', 0.7: 'D-', 0.0: 'F'
        }
        pred_letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - pred))[1]
        ci_lower_letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - ci_lower))[1]
        ci_upper_letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - ci_upper))[1]
        return (pred_letter, (ci_lower_letter, ci_upper_letter)), (pred, (ci_lower, ci_upper))
    
    def save_model(self, model_path: str):
        joblib.dump((self.model, self.scaler, self.feature_columns), model_path)
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: str):
        self.model, self.scaler, self.feature_columns = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not hasattr(self, 'model') or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained or does not support feature importance")
        importance_scores = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance_scores
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        return feature_importance 