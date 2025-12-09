import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
import logging
from typing import Tuple, Dict, Any, Optional, List
import pickle
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Enhanced sentiment analyzer supporting both BERT and TF-IDF text encoding.
    
    This class provides a flexible sentiment classification pipeline that can use
    either modern BERT embeddings or traditional TF-IDF vectorization.
    
    Attributes:
        encoder_type: Type of text encoder ('bert' or 'tfidf')
        model_type: Type of classifier ('lr' for Logistic Regression, 'nb' for Naive Bayes)
        text_encoder: The text encoder instance (BERT or TF-IDF)
        model: The trained classification model
    """
    
    def __init__(self, encoder_type: str = 'bert', max_features: int = 5000):
        """
        Initialize the sentiment analyzer.
        
        Args:
            encoder_type: Type of encoder to use ('bert' or 'tfidf')
            max_features: Maximum features for TF-IDF (ignored for BERT)
        """
        self.encoder_type = encoder_type.lower()
        self.max_features = max_features
        self.model = None
        self.model_type = None
        self.text_encoder = None
        self._is_encoder_fitted = False
        
        # Initialize encoder
        self._init_encoder()
        
        # For backward compatibility - expose vectorizer attribute
        if self.encoder_type == 'tfidf':
            self.vectorizer = self.text_encoder.vectorizer
        else:
            self.vectorizer = None
    
    def _init_encoder(self):
        """Initialize the text encoder based on encoder_type."""
        try:
            from src.models.text_encoder import get_text_encoder
            self.text_encoder = get_text_encoder(self.encoder_type, max_features=self.max_features)
            logger.info(f"Initialized {self.text_encoder.encoder_type} encoder")
        except ImportError:
            # Fallback to direct import if module path doesn't work
            try:
                from text_encoder import get_text_encoder
                self.text_encoder = get_text_encoder(self.encoder_type, max_features=self.max_features)
            except ImportError:
                # Final fallback to TF-IDF if BERT not available
                logger.warning("Could not import text_encoder module, falling back to TF-IDF")
                self.encoder_type = 'tfidf'
                self.text_encoder = None
                self.vectorizer = TfidfVectorizer(max_features=self.max_features)
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'cleaned_text',
                    rating_column: str = 'rating', threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for sentiment analysis.
        
        Args:
            df: DataFrame containing text and ratings
            text_column: Name of the text column
            rating_column: Name of the rating column
            threshold: Rating threshold for positive sentiment (>= threshold)
            
        Returns:
            Tuple of (X, y) for model training
        """
        # Convert ratings to binary sentiment (positive/negative)
        df = df.copy()
        df['sentiment'] = df[rating_column].apply(lambda x: 1 if x >= threshold else 0)
        
        # Get texts
        texts = df[text_column].fillna('').tolist()
        
        # Encode texts
        if self.text_encoder is not None:
            if self.encoder_type == 'bert':
                # BERT doesn't need fitting
                X = self.text_encoder.encode(texts)
                self._is_encoder_fitted = True
            else:
                # TF-IDF needs fitting
                if not self._is_encoder_fitted:
                    X = self.text_encoder.fit_transform(texts)
                    self._is_encoder_fitted = True
                else:
                    X = self.text_encoder.encode(texts)
        else:
            # Fallback: direct TF-IDF
            if not hasattr(self.vectorizer, 'vocabulary_'):
                X = self.vectorizer.fit_transform(texts)
            else:
                X = self.vectorizer.transform(texts)
            
            # Convert sparse to dense for consistency
            if hasattr(X, 'toarray'):
                X = X.toarray()
        
        y = df['sentiment'].values
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Positive sentiment ratio: {y.mean():.2%}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'lr',
                   cv_folds: int = 5, use_cv: bool = True) -> Dict[str, Any]:
        """
        Train a sentiment analysis model with optional cross-validation.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model ('lr' for Logistic Regression, 'nb' for Naive Bayes, 'svm' for SVM)
            cv_folds: Number of cross-validation folds
            use_cv: Whether to use cross-validation
            
        Returns:
            Dictionary containing model performance metrics
        """
        # BERT embeddings can have negative values, so NB won't work
        if self.encoder_type == 'bert' and model_type == 'nb':
            logger.warning("Naive Bayes requires non-negative features. Switching to Logistic Regression for BERT.")
            model_type = 'lr'
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Select and train model
        if model_type == 'nb':
            self.model = MultinomialNB()
            self.model_type = 'Naive Bayes'
        elif model_type == 'lr':
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model_type = 'Logistic Regression'
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
            self.model_type = 'SVM'
        else:
            raise ValueError(f"Invalid model type: {model_type}. Use 'nb', 'lr', or 'svm'")
        
        # Cross-validation on training set
        cv_scores = {}
        if use_cv:
            cv_accuracy = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            cv_f1 = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='f1')
            cv_scores = {
                'cv_accuracy_mean': cv_accuracy.mean(),
                'cv_accuracy_std': cv_accuracy.std(),
                'cv_f1_mean': cv_f1.mean(),
                'cv_f1_std': cv_f1.std()
            }
            logger.info(f"Cross-validation Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
            logger.info(f"Cross-validation F1: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
        
        # Train on full training set
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_train = self.model.predict(X_train)
        
        # Calculate metrics
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_f1': f1_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train),
            'train_recall': recall_score(y_train, y_pred_train)
        }
        
        test_metrics = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred),
            'test_recall': recall_score(y_test, y_pred)
        }
        
        logger.info(f"\nModel: {self.model_type} with {self.text_encoder.encoder_type if self.text_encoder else 'TF-IDF'} encoder")
        logger.info(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1 Score: {test_metrics['test_f1']:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return {
            **train_metrics,
            **test_metrics,
            **cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'encoder_type': self.encoder_type,
            'model_type': self.model_type
        }
    
    def evaluate_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model on new data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary containing model performance metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred)
        }
        
        logger.info(f"Model Evaluation ({self.model_type}):")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y, y_pred))
        
        return {
            **metrics,
            'classification_report': classification_report(y, y_pred)
        }
    
    def predict_sentiment(self, text: str) -> Tuple[int, float]:
        """
        Predict sentiment for a new text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (predicted class, probability of positive sentiment)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Encode text
        if self.text_encoder is not None:
            X = self.text_encoder.encode([text])
        else:
            X = self.vectorizer.transform([text])
            if hasattr(X, 'toarray'):
                X = X.toarray()
        
        # Predict
        pred_class = self.model.predict(X)[0]
        pred_prob = self.model.predict_proba(X)[0][1]
        
        return pred_class, pred_prob
    
    def get_sentiment_scores(self, texts: List[str]) -> np.ndarray:
        """
        Get sentiment probability scores for multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Array of sentiment probabilities (probability of positive)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Encode texts
        if self.text_encoder is not None:
            X = self.text_encoder.encode(texts)
        else:
            X = self.vectorizer.transform(texts)
            if hasattr(X, 'toarray'):
                X = X.toarray()
        
        # Get probabilities
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, model_path: str, encoder_path: str):
        """
        Save trained model and encoder.
        
        Args:
            model_path: Path to save model
            encoder_path: Path to save encoder/vectorizer
        """
        try:
            # Save model with metadata
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'encoder_type': self.encoder_type
            }
            joblib.dump(model_data, model_path)
            
            # Save encoder (for TF-IDF) or just metadata (for BERT)
            if self.encoder_type == 'tfidf':
                encoder_data = {
                    'type': 'tfidf',
                    'vectorizer': self.text_encoder.vectorizer if self.text_encoder else self.vectorizer
                }
            else:
                encoder_data = {
                    'type': 'bert',
                    'model_name': 'all-MiniLM-L6-v2'  # Store model name for reloading
                }
            joblib.dump(encoder_data, encoder_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Encoder saved to {encoder_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str, encoder_path: str):
        """
        Load trained model and encoder.
        
        Args:
            model_path: Path to load model from
            encoder_path: Path to load encoder from
        """
        try:
            # Load model
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.encoder_type = model_data.get('encoder_type', 'tfidf')
            
            # Load encoder
            encoder_data = joblib.load(encoder_path)
            if encoder_data['type'] == 'tfidf':
                from sklearn.feature_extraction.text import TfidfVectorizer
                self.vectorizer = encoder_data['vectorizer']
                self.text_encoder = None
            else:
                # Reinitialize BERT encoder
                self._init_encoder()
            
            self._is_encoder_fitted = True
            logger.info(f"Model loaded from {model_path}")
            logger.info(f"Encoder type: {self.encoder_type}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def main():
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Load processed data
    df = pd.read_csv("data/processed/processed_reviews.csv")
    
    # Prepare data
    X, y = analyzer.prepare_data(df)
    
    # Train model
    metrics = analyzer.train_model(X, y, model_type='nb')
    
    # Save model
    analyzer.save_model(
        "models/sentiment_model.pkl",
        "models/vectorizer.pkl"
    )
    
    # Example prediction
    sample_text = "This professor is amazing and very helpful!"
    pred_class, pred_prob = analyzer.predict_sentiment(sample_text)
    logger.info(f"Sample text: {sample_text}")
    logger.info(f"Predicted sentiment: {'Positive' if pred_class == 1 else 'Negative'}")
    logger.info(f"Confidence: {pred_prob:.4f}")

if __name__ == "__main__":
    main() 