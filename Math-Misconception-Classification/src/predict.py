"""
Prediction script for generating competition submissions
"""

import os
import pandas as pd
import joblib
import numpy as np

from data_loader import load_data, preprocess_text


def generate_submission(model_path='../models/', output_path='../outputs/'):
    """
    Generate competition submission file
    
    Args:
        model_path: Path to saved model files
        output_path: Path to save submission file
    """
    # Load data
    train_df, test_df, sample_df = load_data()
    
    # Load trained model components
    try:
        model = joblib.load(os.path.join(model_path, 'math_misconception_model.pkl'))
        label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        category_mapping = joblib.load(os.path.join(model_path, 'category_mapping.pkl'))
        print("Loaded saved model successfully.")
    except FileNotFoundError:
        print("Saved model not found. Please run training first.")
        return
    
    # Prepare test data
    X_test = preprocess_text(test_df['StudentExplanation'])
    
    # Generate predictions
    print("Generating predictions...")
    probabilities = model.predict_proba(X_test)
    categories = label_encoder.classes_
    
    # Format predictions for submission
    all_predictions = []
    for probs_row in probabilities:
        sorted_indices = np.argsort(probs_row)[::-1]
        top_3_indices = sorted_indices[:3]
        selected_categories = [categories[idx] for idx in top_3_indices]
        predictions = [f"{cat}:{category_mapping.get(cat, 'NA')}" for cat in selected_categories]
        all_predictions.append(" ".join(predictions))
    
    # Create submission file
    submission_df = pd.DataFrame({
        'row_id': test_df['row_id'],
        'Category:Misconception': all_predictions
    })
    
    # Save submission
    os.makedirs(output_path, exist_ok=True)
    submission_file = os.path.join(output_path, 'submission.csv')
    submission_df.to_csv(submission_file, index=False)
    
    print(f"Submission file saved: {submission_file}")
    
    # Display sample predictions
    sample_count = min(5, len(submission_df))
    print(f"\nSample predictions (first {sample_count}):")
    for i in range(sample_count):
        print(f"  {i+1}: {submission_df.iloc[i]['Category:Misconception']}")
    
    return submission_df


if __name__ == "__main__":
    submission = generate_submission()