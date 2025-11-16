"""
Training script for Math Misconception classifier
"""

import os
import joblib
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import log_loss

from data_loader import load_data, preprocess_text
from model import MathMisconceptionClassifier


def train_model(save_path='../models/'):
    """
    Train the model with cross-validation and save results
    
    Args:
        save_path: Path to save trained model
        
    Returns:
        Trained model and label encoder
    """
    # Load and prepare data
    train_df, _, _ = load_data()
    X_train = preprocess_text(train_df['StudentExplanation'])
    y_train = train_df['Category']
    
    # Initialize and train classifier
    classifier = MathMisconceptionClassifier()
    
    print("Starting model training with cross-validation...")
    
    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(
        classifier.model, X_train, classifier.label_encoder.transform(y_train), 
        cv=cv, method='predict_proba', n_jobs=-1
    )
    
    cv_score = log_loss(classifier.label_encoder.transform(y_train), cv_probs)
    print(f"Cross-validation Log Loss: {cv_score:.4f}")
    
    # Train final model on all data
    print("Training final model on full dataset...")
    classifier.fit(X_train, y_train, train_df)
    
    # Save model components
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(classifier.model, os.path.join(save_path, 'math_misconception_model.pkl'))
    joblib.dump(classifier.label_encoder, os.path.join(save_path, 'label_encoder.pkl'))
    joblib.dump(classifier.category_mapping, os.path.join(save_path, 'category_mapping.pkl'))
    
    print(f"Model saved to {save_path}")
    return classifier


if __name__ == "__main__":
    classifier = train_model()