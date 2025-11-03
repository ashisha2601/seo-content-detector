"""
Content Quality Scorer Module
Implements duplicate detection and quality classification
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib


def detect_duplicates(embeddings, urls, threshold=0.80):
    """
    Detect duplicate content using cosine similarity.
    
    Args:
        embeddings (numpy.ndarray): Embedding matrix
        urls (list): List of URLs
        threshold (float): Similarity threshold for duplicates
        
    Returns:
        pd.DataFrame: DataFrame with duplicate pairs
    """
    print(f"Computing cosine similarity matrix (threshold={threshold})...")
    
    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find pairs above threshold (excluding diagonal)
    duplicate_pairs = []
    n = len(urls)
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i, j]
            if similarity >= threshold:
                duplicate_pairs.append({
                    'url1': urls[i],
                    'url2': urls[j],
                    'similarity': round(similarity, 4)
                })
    
    duplicates_df = pd.DataFrame(duplicate_pairs)
    
    print(f"Found {len(duplicates_df)} duplicate pairs")
    return duplicates_df


def detect_thin_content(df, word_count_threshold=500):
    """
    Flag pages with thin content (low word count).
    
    Args:
        df (pd.DataFrame): DataFrame with 'word_count' column
        word_count_threshold (int): Minimum word count threshold
        
    Returns:
        pd.DataFrame: DataFrame with 'is_thin' column added
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Check if word_count column exists
    if 'word_count' in df.columns:
        # Check for duplicate column names
        word_count_cols = [col for col in df.columns if col == 'word_count']
        
        if len(word_count_cols) > 1:
            # Multiple word_count columns exist, use the first one
            word_count_values = df.iloc[:, df.columns.get_loc('word_count')].values
            # If still a DataFrame, get first column
            if isinstance(word_count_values, pd.DataFrame):
                word_count_values = word_count_values.iloc[:, 0].values
        else:
            # Single word_count column
            word_count_values = df['word_count'].values
        
        # Ensure it's a 1D numpy array
        if isinstance(word_count_values, pd.DataFrame):
            word_count_values = word_count_values.iloc[:, 0].values
        elif isinstance(word_count_values, pd.Series):
            word_count_values = word_count_values.values
        
        # Create is_thin as a numpy array first, then assign
        is_thin_array = (word_count_values < word_count_threshold).astype(int)
        df['is_thin'] = is_thin_array
    else:
        # Fallback: assume word_count doesn't exist, set all to 0
        df['is_thin'] = 0
        print("Warning: 'word_count' column not found, setting all is_thin to 0")
    
    return df


def create_quality_labels(df):
    """
    Create synthetic quality labels based on word count and readability.
    
    Labeling rules:
    - High: word_count > 1500 AND 50 <= readability <= 70
    - Low: word_count < 500 OR readability < 30
    - Medium: all other cases
    
    Args:
        df (pd.DataFrame): DataFrame with 'word_count' and 'flesch_reading_ease' columns
        
    Returns:
        pd.DataFrame: DataFrame with 'quality_label' column added
    """
    def assign_label(row):
        word_count = row['word_count']
        readability = row['flesch_reading_ease']
        
        # High quality
        if word_count > 1500 and 50 <= readability <= 70:
            return 'High'
        # Low quality
        elif word_count < 500 or readability < 30:
            return 'Low'
        # Medium quality
        else:
            return 'Medium'
    
    df['quality_label'] = df.apply(assign_label, axis=1)
    return df


def train_quality_model(df, test_size=0.3, random_state=42):
    """
    Train a quality classification model.
    
    Args:
        df (pd.DataFrame): DataFrame with features and quality labels
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (model, test_accuracy, test_f1, feature_importance)
    """
    print("Preparing data for model training...")
    
    # Features to use
    feature_columns = ['word_count', 'sentence_count', 'flesch_reading_ease']
    X = df[feature_columns].values
    y = df['quality_label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("Training Random Forest classifier...")
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop Features:")
    print(feature_importance)
    
    # Baseline: Rule-based classifier using word count only
    baseline_pred = ['High' if wc > 1500 else ('Low' if wc < 500 else 'Medium') 
                     for wc in X_test[:, 0]]
    baseline_accuracy = accuracy_score(y_test, baseline_pred)
    
    print(f"\nBaseline Accuracy (word count only): {baseline_accuracy:.4f}")
    
    return model, accuracy, f1, feature_importance


def save_model(model, filepath):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from disk."""
    return joblib.load(filepath)


    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from disk."""
    return joblib.load(filepath)


    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from disk."""
    return joblib.load(filepath)

