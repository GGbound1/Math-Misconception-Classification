"""
Model definition and ensemble creation
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

from .feature_engineering import create_feature_pipeline
from .data_loader import create_category_mapping


def create_ensemble_model():
    """
    Create ensemble model with multiple classifiers
    
    Returns:
        Pipeline with feature engineering and ensemble classifier
    """
    # LightGBM classifier
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=12,
        num_leaves=80,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=10,
        objective='multi:softprob',
        random_state=42,
        n_jobs=-1
    )
    
    # Random Forest classifier
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Ensemble with weighted voting
    ensemble_model = VotingClassifier(
        estimators=[
            ('lgb', lgb_model),
            ('xgb', xgb_model),
            ('rf', rf_model)
        ],
        voting='soft',
        weights=[3, 2, 1]  # LightGBM has highest weight
    )
    
    # Complete pipeline
    model = Pipeline([
        ('features', create_feature_pipeline()),
        ('classifier', ensemble_model)
    ])
    
    return model


class MathMisconceptionClassifier:
    """
    High-level classifier for math misconception prediction
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.category_mapping = None
    
    def fit(self, X, y, train_df=None):
        """
        Fit the model with training data
        
        Args:
            X: Text features
            y: Target labels
            train_df: Training dataframe for creating category mapping
        """
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        if train_df is not None:
            self.category_mapping = create_category_mapping(train_df)
        
        self.model = create_ensemble_model()
        self.model.fit(X, y_encoded)
        
        return self
    
    def predict_top3(self, X, return_categories_only=False):
        """
        Predict top 3 categories with probabilities
        
        Args:
            X: Text features to predict
            return_categories_only: Whether to return only category names
            
        Returns:
            List of top 3 predictions per sample
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        probabilities = self.model.predict_proba(X)
        categories = self.label_encoder.classes_
        
        all_predictions = []
        for probs_row in probabilities:
            sorted_indices = np.argsort(probs_row)[::-1]
            top_3_indices = sorted_indices[:3]
            selected_categories = [categories[idx] for idx in top_3_indices]
            
            if return_categories_only:
                all_predictions.append(selected_categories)
            else:
                predictions = [f"{cat}:{self.category_mapping.get(cat, 'NA')}" for cat in selected_categories]
                all_predictions.append(" ".join(predictions))
        
        return all_predictions