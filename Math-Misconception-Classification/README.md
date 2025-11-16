# MAP - Charting Student Math Misunderstandings

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-LightGBM%2CXGBoost%2CRandomForest-orange)](https://scikit-learn.org/)

A machine learning solution for identifying student math misconceptions from textual explanations, achieving a score of 0.62993 in the Kaggle competition.

##  Project Overview

This project tackles the challenge of automatically categorizing student math misconceptions based on their written explanations. The solution employs an ensemble of advanced machine learning models with sophisticated feature engineering to predict both the category and specific misconception.

##  Competition Performance

- **Competition**: MAP - Charting Student Math Misunderstandings
- **Final Score**: 0.62993
- **Rank**: 1685/1858 (Top 91%)
- **Approach**: Ensemble of LightGBM, XGBoost, and RandomForest with TF-IDF and custom features

##  Technical Approach

### Feature Engineering
- **Textual Features**: TF-IDF with n-grams (char & word level)
- **Linguistic Features**: Word count, character count, lexical diversity
- **Mathematical Features**: Math term frequency, operator count, fraction/decimal detection
- **Stylistic Features**: Capitalization ratio, punctuation count, question patterns

### Model Architecture
- **Ensemble Method**: Soft Voting Classifier
- **Base Models**: 
  - LightGBM with class balancing
  - XGBoost with multi-class optimization  
  - RandomForest with feature importance
- **Preprocessing**: Advanced feature union pipeline

### Key Techniques
- Cross-validation with stratified sampling
- Probability calibration for multi-class prediction
- Post-processing based on training data patterns
- Top-3 prediction ranking

## Project Structure
