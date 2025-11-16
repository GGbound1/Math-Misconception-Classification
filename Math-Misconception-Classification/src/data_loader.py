"""
Data loading utilities for MAP Math Misconceptions competition
"""

import os
import pandas as pd
from typing import Tuple

def load_data(data_path: str = '../data/') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load competition data files
    
    Args:
        data_path: Path to directory containing competition data
        
    Returns:
        Tuple of (train_df, test_df, sample_df)
    """
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))
    sample_df = pd.read_csv(os.path.join(data_path, 'sample_submission.csv'))
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_df.shape}")
    
    return train_df, test_df, sample_df

def create_category_mapping(train_df: pd.DataFrame) -> dict:
    """
    Create mapping from category to most common misconception
    
    Args:
        train_df: Training dataframe
        
    Returns:
        Dictionary mapping categories to misconceptions
    """
    mapping = {}
    for category in train_df['Category'].unique():
        category_data = train_df[train_df['Category'] == category]
        misconception_counts = category_data['Misconception'].value_counts()
        if len(misconception_counts) > 0:
            mapping[category] = misconception_counts.index[0]
        else:
            mapping[category] = 'NA'
    return mapping

def preprocess_text(text_series: pd.Series) -> pd.Series:
    """
    Preprocess text data by handling missing values
    
    Args:
        text_series: Series containing text data
        
    Returns:
        Preprocessed text series
    """
    return text_series.fillna("")