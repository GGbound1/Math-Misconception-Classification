"""
Advanced feature engineering for text classification
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


class AdvancedFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract advanced linguistic and mathematical features from student explanations
    """
    
    def __init__(self):
        self.math_terms = [
            'fraction', 'divide', 'multiply', 'equal', 'equivalent', 
            'whole', 'part', 'half', 'quarter', 'third', 'numerator',
            'denominator', 'decimal', 'point'
        ]
        self.question_words = ['what', 'why', 'how', 'when', 'where', 'which', 'who']
        self.positive_words = ['correct', 'right', 'yes', 'true', 'good', 'accurate', 'proper', 'valid']
        self.negative_words = ['wrong', 'incorrect', 'no', 'false', 'bad', 'mistake', 'error', 'invalid', 'improper']
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, texts):
        features = []
        for text in texts:
            if pd.isna(text):
                text = ""
            text_lower = str(text).lower()
            
            # Basic text statistics
            words = text_lower.split()
            word_count = len(words)
            char_count = len(text_lower)
            unique_words = len(set(words))
            lexical_diversity = unique_words / max(word_count, 1)
            avg_word_length = char_count / max(word_count, 1)
            
            # Sentence structure
            sentence_count = max(len(re.findall(r'[.!?]+', text_lower)), 1)
            avg_sentence_length = word_count / sentence_count
            
            # Mathematical content
            math_terms = len(re.findall(r'\b(' + '|'.join(self.math_terms) + r')\b', text_lower))
            numbers = len(re.findall(r'\d+', text_lower))
            fractions = len(re.findall(r'\d+/\d+', text_lower))
            decimals = len(re.findall(r'\d+\.\d+', text_lower))
            math_operators = len(re.findall(r'[\+\-\*\/=]', text_lower))
            
            # Linguistic patterns
            question_words = len(re.findall(r'\b(' + '|'.join(self.question_words) + r')\b', text_lower))
            positive_words = len(re.findall(r'\b(' + '|'.join(self.positive_words) + r')\b', text_lower))
            negative_words = len(re.findall(r'\b(' + '|'.join(self.negative_words) + r')\b', text_lower))
            
            # Text style features
            capital_ratio = len(re.findall(r'[A-Z]', text)) / max(char_count, 1)
            digit_ratio = len(re.findall(r'\d', text)) / max(char_count, 1)
            punctuation_count = len(re.findall(r'[^\w\s]', text))
            
            feature_vector = [
                word_count, char_count, unique_words, lexical_diversity, avg_word_length,
                sentence_count, avg_sentence_length,
                math_terms, numbers, fractions, decimals, math_operators,
                question_words, positive_words, negative_words,
                capital_ratio, digit_ratio, punctuation_count
            ]
            features.append(feature_vector)
        
        return np.array(features)


def create_feature_pipeline():
    """
    Create comprehensive feature pipeline combining multiple feature types
    """
    return FeatureUnion([
        ('advanced_features', AdvancedFeatureExtractor()),
        ('tfidf_word', TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            max_features=2000,
            stop_words='english',
            min_df=2
        )),
        ('tfidf_char', TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            max_features=1500,
            min_df=2
        ))
    ])