import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging
from typing import Tuple, Dict, Any
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = None
        self.model_type = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for sentiment analysis.
        
        Args:
            df: DataFrame containing cleaned text and ratings
            
        Returns:
            Tuple of (X, y) for model training
        """
        # Convert ratings to binary sentiment (positive/negative)
        df['sentiment'] = df['rating'].apply(lambda x: 1 if x >= 3.5 else 0)
        
        # Vectorize text
        if not hasattr(self.vectorizer, 'vocabulary_'):
            # Only fit the vectorizer if it hasn't been fit before
            X = self.vectorizer.fit_transform(df['cleaned_text'])
        else:
            # Use transform only if the vectorizer was already fit
            X = self.vectorizer.transform(df['cleaned_text'])
            
        y = df['sentiment'].values
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray, model_type: str = 'nb') -> Dict[str, Any]:
        """
        Train a sentiment analysis model.
        
        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model to train ('nb' for Naive Bayes, 'lr' for Logistic Regression)
            
        Returns:
            Dictionary containing model performance metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        if model_type == 'nb':
            self.model = MultinomialNB()
            self.model_type = 'Naive Bayes'
        elif model_type == 'lr':
            self.model = LogisticRegression(max_iter=1000)
            self.model_type = 'Logistic Regression'
        else:
            raise ValueError("Invalid model type. Use 'nb' or 'lr'")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"Model: {self.model_type}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(y_test, y_pred)
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
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        
        logger.info(f"Model Evaluation ({self.model_type}):")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y, y_pred))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': classification_report(y, y_pred)
        }
    
    def predict_sentiment(self, text: str) -> Tuple[int, float]:
        """
        Predict sentiment for a new text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (predicted class, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Vectorize text
        X = self.vectorizer.transform([text])
        
        # Predict
        pred_class = self.model.predict(X)[0]
        pred_prob = self.model.predict_proba(X)[0][1]
        
        return pred_class, pred_prob
    
    def save_model(self, model_path: str, vectorizer_path: str):
        """
        Save trained model and vectorizer.
        
        Args:
            model_path: Path to save model
            vectorizer_path: Path to save vectorizer
        """
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            logger.info(f"Model and vectorizer saved to {model_path} and {vectorizer_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str, vectorizer_path: str):
        """
        Load trained model and vectorizer.
        
        Args:
            model_path: Path to load model from
            vectorizer_path: Path to load vectorizer from
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("Model and vectorizer loaded successfully")
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