"""
Enhanced Grade Predictor with BERT Embeddings and Hyperparameter Tuning.

This module provides an advanced grade prediction model that combines:
- BERT text embeddings for semantic understanding
- Professor/course disentanglement features
- Hyperparameter tuning via GridSearchCV
- Cross-validation for robust evaluation
- Student input features (for prediction only)

Author: Enhanced for Resume Portfolio
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
from typing import Tuple, Dict, Any, Optional, List
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedGradePredictor:
    """
    Enhanced grade predictor with BERT embeddings and hyperparameter tuning.
    
    This predictor combines review text embeddings with numerical features
    to predict student grades. It supports professor/course disentanglement
    to separate the effect of the professor vs. the course itself.
    
    Features used:
    - BERT embeddings of review text (384 dimensions, reduced via PCA)
    - Rating and difficulty scores
    - Professor-level aggregated features
    - Course-level baseline features
    - Optional student self-assessment features
    """
    
    # Default hyperparameter grid for tuning
    PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    # Reduced grid for faster tuning
    PARAM_GRID_FAST = {
        'n_estimators': [100],
        'max_depth': [3, 5],
        'learning_rate': [0.1],
        'min_samples_split': [2]
    }
    
    def __init__(self, use_bert: bool = True, use_hyperparameter_tuning: bool = True,
                 reduced_embedding_dim: int = 50):
        """
        Initialize the enhanced grade predictor.
        
        Args:
            use_bert: Whether to use BERT embeddings (vs. just numerical features)
            use_hyperparameter_tuning: Whether to tune hyperparameters
            reduced_embedding_dim: Dimension to reduce BERT embeddings to via PCA
        """
        self.use_bert = use_bert
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.reduced_embedding_dim = reduced_embedding_dim
        
        self.model = None
        self.scaler = StandardScaler()
        self.pca = None  # For reducing BERT embedding dimensions
        self.text_encoder = None
        
        # Feature tracking
        self.numerical_features = [
            'rating', 'difficulty', 'sentiment_score'
        ]
        self.professor_features = [
            'professor_avg_rating', 'professor_avg_difficulty', 
            'professor_review_count', 'professor_avg_grade'
        ]
        self.course_features = [
            'course_avg_rating', 'course_avg_difficulty', 'course_avg_grade'
        ]
        self.student_features = [
            'study_hours', 'prior_gpa', 'interest_level'
        ]
        
        self.feature_columns = []
        self.best_params = None
        self.training_metrics = {}
        
        # Initialize BERT encoder if needed
        if use_bert:
            self._init_bert_encoder()
    
    def _init_bert_encoder(self):
        """Initialize BERT text encoder."""
        try:
            from src.models.text_encoder import get_text_encoder
            self.text_encoder = get_text_encoder('bert')
            logger.info(f"Initialized BERT encoder: {self.text_encoder.encoder_type}")
        except ImportError:
            try:
                from text_encoder import get_text_encoder
                self.text_encoder = get_text_encoder('bert')
            except ImportError:
                logger.warning("BERT encoder not available, falling back to numerical features only")
                self.use_bert = False
    
    def _get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get BERT embeddings for texts."""
        if self.text_encoder is None:
            raise ValueError("BERT encoder not initialized")
        
        embeddings = self.text_encoder.encode(texts)
        
        # Reduce dimensions with PCA for efficiency
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
        
        return embeddings
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True,
                        include_student_features: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for training or prediction.
        
        Args:
            df: DataFrame with review data
            is_training: Whether this is training data (affects scaling/PCA fitting)
            include_student_features: Whether to include student self-assessment features
            
        Returns:
            Tuple of (X, y) where y may be None for prediction
        """
        df_features = df.copy()
        
        # Calculate sentiment score if not present
        if 'sentiment_score' not in df_features.columns:
            if 'cleaned_text' in df_features.columns:
                from textblob import TextBlob
                df_features['sentiment_score'] = df_features['cleaned_text'].apply(
                    lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0.0
                )
            else:
                df_features['sentiment_score'] = 0.0
        
        if is_training:
            # Collect all feature columns for training
            all_features = self.numerical_features.copy()
            
            # Add professor features if available
            for feat in self.professor_features:
                if feat in df_features.columns:
                    all_features.append(feat)
            
            # Add course features if available
            for feat in self.course_features:
                if feat in df_features.columns:
                    all_features.append(feat)
            
            # Add student features if requested and available
            if include_student_features:
                for feat in self.student_features:
                    if feat in df_features.columns:
                        all_features.append(feat)
            
            # Store feature columns for later reference (excluding BERT dims)
            self.numerical_feature_columns = list(all_features)
        else:
            # For prediction, use the feature columns from training
            all_features = self.numerical_feature_columns.copy() if hasattr(self, 'numerical_feature_columns') else self.numerical_features.copy()
            
            # Add any missing columns with default values
            for feat in all_features:
                if feat not in df_features.columns:
                    df_features[feat] = 0.0
        
        # Extract numerical features
        X_numerical = df_features[all_features].copy()
        
        # Fill missing values with column means (training) or zeros (prediction)
        for col in X_numerical.columns:
            if X_numerical[col].isna().any():
                if is_training:
                    X_numerical[col] = X_numerical[col].fillna(X_numerical[col].mean())
                else:
                    X_numerical[col] = X_numerical[col].fillna(0.0)
        
        # Store feature columns for later reference
        if is_training:
            self.feature_columns = list(X_numerical.columns)
        
        # Scale numerical features
        if is_training:
            X_scaled = self.scaler.fit_transform(X_numerical)
        else:
            X_scaled = self.scaler.transform(X_numerical)
        
        # Add BERT embeddings if enabled
        if self.use_bert and 'cleaned_text' in df_features.columns:
            texts = df_features['cleaned_text'].fillna('').tolist()
            
            if is_training:
                # Fit PCA on training data
                raw_embeddings = self.text_encoder.encode(texts)
                
                from sklearn.decomposition import PCA
                n_components = min(self.reduced_embedding_dim, raw_embeddings.shape[1], len(texts) - 1)
                self.pca = PCA(n_components=n_components, random_state=42)
                embeddings = self.pca.fit_transform(raw_embeddings)
                
                explained_variance = self.pca.explained_variance_ratio_.sum()
                logger.info(f"PCA reduced BERT embeddings from {raw_embeddings.shape[1]} to {n_components} dims "
                           f"(explained variance: {explained_variance:.2%})")
            else:
                embeddings = self._get_bert_embeddings(texts)
            
            # Combine numerical and embedding features
            X = np.hstack([X_scaled, embeddings])
            
            # Update feature columns to include embedding dimensions
            embedding_cols = [f'bert_dim_{i}' for i in range(embeddings.shape[1])]
            if is_training:
                self.feature_columns.extend(embedding_cols)
        else:
            X = X_scaled
        
        # Get target variable if available
        y = None
        if 'grade_numerical' in df_features.columns:
            y = df_features['grade_numerical'].values
        
        logger.info(f"Prepared features: {X.shape[0]} samples, {X.shape[1]} features")
        
        return X, y
    
    def train(self, train_df: pd.DataFrame, cv_folds: int = 5, 
              use_fast_tuning: bool = True) -> Dict[str, Any]:
        """
        Train the grade prediction model with optional hyperparameter tuning.
        
        Args:
            train_df: Training DataFrame
            cv_folds: Number of cross-validation folds
            use_fast_tuning: Use reduced parameter grid for faster tuning
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Preparing features for training...")
        X, y = self.prepare_features(train_df, is_training=True)
        
        # Remove rows with missing grades
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training on {len(y)} samples with valid grades")
        
        # Train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Hyperparameter tuning
        if self.use_hyperparameter_tuning:
            logger.info("Performing hyperparameter tuning...")
            param_grid = self.PARAM_GRID_FAST if use_fast_tuning else self.PARAM_GRID
            
            base_model = GradientBoostingRegressor(random_state=42)
            grid_search = GridSearchCV(
                base_model, param_grid, 
                cv=cv_folds, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            logger.info(f"Best parameters: {self.best_params}")
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
            )
            self.model.fit(X_train, y_train)
        
        # Cross-validation on full training set
        cv_scores = cross_val_score(
            self.model, X, y, cv=cv_folds, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_rmse_std = np.sqrt(-cv_scores.std())
        
        # Evaluate on train and test
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.training_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'test_r2': r2_score(y_test, test_pred),
            'cv_rmse': cv_rmse,
            'cv_rmse_std': cv_rmse_std,
            'n_features': X.shape[1],
            'n_samples': len(y),
            'best_params': self.best_params
        }
        
        logger.info(f"\n{'='*50}")
        logger.info("Training Results:")
        logger.info(f"  Train RMSE: {self.training_metrics['train_rmse']:.4f}")
        logger.info(f"  Test RMSE: {self.training_metrics['test_rmse']:.4f}")
        logger.info(f"  Test MAE: {self.training_metrics['test_mae']:.4f}")
        logger.info(f"  Test R²: {self.training_metrics['test_r2']:.4f}")
        logger.info(f"  CV RMSE: {cv_rmse:.4f} (+/- {cv_rmse_std:.4f})")
        logger.info(f"{'='*50}")
        
        return self.training_metrics
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_test, y_test = self.prepare_features(test_df, is_training=False)
        
        # Remove rows with missing grades
        valid_mask = ~np.isnan(y_test)
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def predict_grade(self, features_dict: Dict[str, Any], 
                     include_confidence: bool = True) -> Dict[str, Any]:
        """
        Predict grade for a single sample with optional confidence interval.
        
        Args:
            features_dict: Dictionary of input features
            include_confidence: Whether to include confidence interval
            
        Returns:
            Dictionary with prediction results
        """
        # Create DataFrame from input
        df = pd.DataFrame([features_dict])
        
        # Prepare features
        X, _ = self.prepare_features(df, is_training=False, 
                                     include_student_features=True)
        
        # Make prediction
        pred_numerical = self.model.predict(X)[0]
        
        # Clip to valid grade range
        pred_numerical = np.clip(pred_numerical, 0.0, 4.0)
        
        # Convert to letter grade
        grade_mapping = {
            4.0: 'A', 3.7: 'A-', 3.3: 'B+', 3.0: 'B', 2.7: 'B-',
            2.3: 'C+', 2.0: 'C', 1.7: 'C-', 1.3: 'D+', 1.0: 'D', 
            0.7: 'D-', 0.0: 'F'
        }
        pred_letter = min(grade_mapping.items(), key=lambda x: abs(x[0] - pred_numerical))[1]
        
        result = {
            'predicted_grade': pred_letter,
            'predicted_gpa': round(pred_numerical, 2)
        }
        
        # Add confidence interval using staged predictions
        if include_confidence and hasattr(self.model, 'staged_predict'):
            staged_preds = list(self.model.staged_predict(X))
            if len(staged_preds) > 10:
                preds = np.array(staged_preds[-10:])[:, 0]
                ci_lower = np.percentile(preds, 10)
                ci_upper = np.percentile(preds, 90)
                result['confidence_interval'] = {
                    'lower_gpa': round(ci_lower, 2),
                    'upper_gpa': round(ci_upper, 2),
                    'lower_grade': min(grade_mapping.items(), key=lambda x: abs(x[0] - ci_lower))[1],
                    'upper_grade': min(grade_mapping.items(), key=lambda x: abs(x[0] - ci_upper))[1]
                }
        
        return result
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained or doesn't support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def save_model(self, path: str):
        """Save the complete model to disk."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_columns': self.feature_columns,
            'numerical_feature_columns': getattr(self, 'numerical_feature_columns', self.numerical_features),
            'use_bert': self.use_bert,
            'best_params': self.best_params,
            'training_metrics': self.training_metrics
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a saved model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.feature_columns = model_data['feature_columns']
        self.numerical_feature_columns = model_data.get('numerical_feature_columns', self.numerical_features)
        self.use_bert = model_data['use_bert']
        self.best_params = model_data.get('best_params')
        self.training_metrics = model_data.get('training_metrics', {})
        
        if self.use_bert:
            self._init_bert_encoder()
        
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    logger.info("Enhanced Grade Predictor - Example")
    
    # This would be run with actual data
    # predictor = EnhancedGradePredictor(use_bert=True)
    # predictor.train(train_df)
    # metrics = predictor.evaluate(test_df)
