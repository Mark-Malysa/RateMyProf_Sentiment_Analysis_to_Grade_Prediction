"""
Text Encoding Module with BERT and TF-IDF support.

This module provides text encoding strategies for the sentiment analysis 
and grade prediction pipeline. It supports both modern BERT embeddings 
(using sentence-transformers) and traditional TF-IDF vectorization.

Author: Enhanced for Resume Portfolio
"""

import numpy as np
import logging
from typing import List, Union, Optional
from abc import ABC, abstractmethod

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextEncoder(ABC):
    """Abstract base class for text encoders."""
    
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into numerical representations.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the dimensionality of the embeddings."""
        pass
    
    @property
    @abstractmethod
    def encoder_type(self) -> str:
        """Return the type of encoder as a string."""
        pass


class BERTTextEncoder(TextEncoder):
    """
    BERT-based text encoder using sentence-transformers.
    
    Uses the 'all-MiniLM-L6-v2' model by default for a good balance
    of speed and quality. Produces 384-dimensional embeddings.
    
    Advantages over TF-IDF:
    - Captures semantic meaning, not just word overlap
    - Handles synonyms and context
    - Pre-trained on large corpus
    - Works well with short and long texts
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize the BERT encoder.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self._embedding_dim = self.model.get_sentence_embedding_dimension()
            self._model_name = model_name
            logger.info(f"Loaded BERT encoder: {model_name} (dim={self._embedding_dim})")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for BERT encoding. "
                "Install with: pip install sentence-transformers"
            )
    
    def encode(self, texts: List[str], batch_size: int = 32, 
               show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts using BERT.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_texts, 384)
        """
        # Handle empty or None texts
        cleaned_texts = [str(t) if t else "" for t in texts]
        
        embeddings = self.model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def encoder_type(self) -> str:
        return f"BERT ({self._model_name})"


class TFIDFTextEncoder(TextEncoder):
    """
    TF-IDF based text encoder for backward compatibility.
    
    Uses scikit-learn's TfidfVectorizer to convert text to 
    sparse TF-IDF features, then converts to dense numpy array.
    """
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        """
        Initialize TF-IDF encoder.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to include (min_n, max_n)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english'
        )
        self._max_features = max_features
        self._is_fitted = False
        logger.info(f"Initialized TF-IDF encoder (max_features={max_features})")
    
    def fit(self, texts: List[str]) -> 'TFIDFTextEncoder':
        """
        Fit the TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of training texts
            
        Returns:
            self for chaining
        """
        cleaned_texts = [str(t) if t else "" for t in texts]
        self.vectorizer.fit(cleaned_texts)
        self._is_fitted = True
        self._actual_features = len(self.vectorizer.vocabulary_)
        logger.info(f"Fitted TF-IDF encoder with {self._actual_features} features")
        return self
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts using TF-IDF.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (n_texts, n_features)
        """
        if not self._is_fitted:
            # Fit and transform if not fitted
            return self.fit_transform(texts)
        
        cleaned_texts = [str(t) if t else "" for t in texts]
        sparse_matrix = self.vectorizer.transform(cleaned_texts)
        return sparse_matrix.toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform in one step.
        
        Args:
            texts: List of training texts
            
        Returns:
            numpy array of TF-IDF features
        """
        cleaned_texts = [str(t) if t else "" for t in texts]
        sparse_matrix = self.vectorizer.fit_transform(cleaned_texts)
        self._is_fitted = True
        self._actual_features = len(self.vectorizer.vocabulary_)
        return sparse_matrix.toarray()
    
    def get_embedding_dim(self) -> int:
        if self._is_fitted:
            return len(self.vectorizer.vocabulary_)
        return self._max_features
    
    @property
    def encoder_type(self) -> str:
        return "TF-IDF"


def get_text_encoder(encoder_type: str = 'bert', **kwargs) -> TextEncoder:
    """
    Factory function to get the appropriate text encoder.
    
    Args:
        encoder_type: Type of encoder ('bert' or 'tfidf')
        **kwargs: Additional arguments passed to the encoder
        
    Returns:
        TextEncoder instance
    """
    encoder_type = encoder_type.lower()
    
    if encoder_type == 'bert':
        # Filter out TF-IDF specific arguments
        bert_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['model_name', 'device']}
        return BERTTextEncoder(**bert_kwargs)
    elif encoder_type == 'tfidf':
        # Filter out BERT specific arguments
        tfidf_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['max_features', 'ngram_range']}
        return TFIDFTextEncoder(**tfidf_kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Use 'bert' or 'tfidf'.")


# Convenience functions for common use cases
def encode_with_bert(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Quick helper to encode texts with BERT."""
    encoder = BERTTextEncoder(model_name=model_name)
    return encoder.encode(texts)


def encode_with_tfidf(texts: List[str], max_features: int = 5000) -> np.ndarray:
    """Quick helper to encode texts with TF-IDF."""
    encoder = TFIDFTextEncoder(max_features=max_features)
    return encoder.fit_transform(texts)


if __name__ == "__main__":
    # Test the encoders
    test_texts = [
        "This professor is amazing and very helpful!",
        "Terrible class, do not recommend.",
        "Average professor, nothing special."
    ]
    
    print("Testing BERT encoder...")
    try:
        bert_encoder = get_text_encoder('bert')
        bert_embeddings = bert_encoder.encode(test_texts)
        print(f"  BERT embeddings shape: {bert_embeddings.shape}")
        print(f"  Encoder type: {bert_encoder.encoder_type}")
    except ImportError as e:
        print(f"  BERT not available: {e}")
    
    print("\nTesting TF-IDF encoder...")
    tfidf_encoder = get_text_encoder('tfidf', max_features=100)
    tfidf_embeddings = tfidf_encoder.encode(test_texts)
    print(f"  TF-IDF embeddings shape: {tfidf_embeddings.shape}")
    print(f"  Encoder type: {tfidf_encoder.encoder_type}")
