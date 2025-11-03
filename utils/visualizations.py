"""
Visualization Module
Creates similarity heatmap, feature importance plots, word clouds, distribution charts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_similarity_heatmap(embeddings, urls, top_n=20, figsize=(12, 10)):
    """
    Create similarity heatmap for duplicate detection.
    
    Args:
        embeddings (numpy.ndarray): Embedding matrix
        urls (list): List of URLs
        top_n (int): Number of top URLs to show
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Limit to top N for readability
    if len(urls) > top_n:
        # Use first top_n URLs
        similarity_matrix = similarity_matrix[:top_n, :top_n]
        urls_display = [url[:50] + '...' if len(url) > 50 else url for url in urls[:top_n]]
    else:
        urls_display = [url[:50] + '...' if len(url) > 50 else url for url in urls]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        similarity_matrix,
        xticklabels=urls_display,
        yticklabels=urls_display,
        annot=False,
        cmap='YlOrRd',
        fmt='.2f',
        cbar_kws={'label': 'Cosine Similarity'},
        ax=ax
    )
    ax.set_title('Content Similarity Heatmap', fontsize=16, fontweight='bold')
    ax.set_xlabel('URLs', fontsize=12)
    ax.set_ylabel('URLs', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig


def plot_feature_importance(feature_importance_df, figsize=(10, 6)):
    """
    Plot feature importance from trained model.
    
    Args:
        feature_importance_df (pd.DataFrame): DataFrame with 'feature' and 'importance' columns
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by importance
    df_sorted = feature_importance_df.sort_values('importance', ascending=True)
    
    # Create horizontal bar plot
    ax.barh(df_sorted['feature'], df_sorted['importance'], color='steelblue')
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance for Quality Classification', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(df_sorted['importance']):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    return fig


def plot_quality_distribution(df, figsize=(12, 5)):
    """
    Plot distribution of quality labels and features.
    
    Args:
        df (pd.DataFrame): DataFrame with 'quality_label' and feature columns
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Quality label distribution
    quality_counts = df['quality_label'].value_counts()
    axes[0].bar(quality_counts.index, quality_counts.values, color=['#ff6b6b', '#4ecdc4', '#95e1d3'])
    axes[0].set_title('Quality Label Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Quality Label')
    axes[0].set_ylabel('Count')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(quality_counts.values):
        axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom')
    
    # Word count distribution by quality
    if 'word_count' in df.columns:
        for label in ['Low', 'Medium', 'High']:
            subset = df[df['quality_label'] == label]['word_count']
            if len(subset) > 0:
                axes[1].hist(subset, alpha=0.6, label=label, bins=20)
        
        axes[1].set_title('Word Count Distribution by Quality', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_wordcloud(text, max_words=50, figsize=(10, 6)):
    """
    Create word cloud from text.
    
    Args:
        text (str): Input text
        max_words (int): Maximum number of words
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object or None if text is empty
    """
    if not text or text == '':
        return None
    
    try:
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=max_words,
            colormap='viridis'
        ).generate(text)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Word Cloud', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        print(f"Error creating word cloud: {e}")
        return None


def plot_readability_distribution(df, figsize=(10, 6)):
    """
    Plot readability score distribution.
    
    Args:
        df (pd.DataFrame): DataFrame with 'flesch_reading_ease' column
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if 'flesch_reading_ease' in df.columns:
        ax.hist(df['flesch_reading_ease'].dropna(), bins=30, color='skyblue', edgecolor='black')
        ax.set_title('Readability Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Flesch Reading Ease Score')
        ax.set_ylabel('Frequency')
        ax.axvline(df['flesch_reading_ease'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["flesch_reading_ease"].mean():.1f}')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

