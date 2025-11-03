"""
Advanced NLP Analysis Module
Sentiment analysis, Named Entity Recognition, Topic Modeling
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
import spacy
from collections import Counter
import re

# Try to load spaCy model, fallback if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except OSError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")


def analyze_sentiment(text):
    """
    Analyze sentiment of text.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Sentiment scores (polarity, subjectivity)
    """
    if not text or text == '':
        return {'polarity': 0.0, 'subjectivity': 0.0}
    
    try:
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,  # -1 to 1 (negative to positive)
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1 (objective to subjective)
        }
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0}


def extract_named_entities(text):
    """
    Extract named entities from text using spaCy.
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Counts of different entity types
    """
    if not text or text == '' or not SPACY_AVAILABLE:
        return {
            'person': 0,
            'org': 0,
            'gpe': 0,  # Geopolitical entity
            'product': 0,
            'total_entities': 0
        }
    
    try:
        doc = nlp(text[:100000])  # Limit to avoid memory issues
        
        entity_counts = {
            'person': 0,
            'org': 0,
            'gpe': 0,
            'product': 0,
            'total_entities': 0
        }
        
        for ent in doc.ents:
            entity_counts['total_entities'] += 1
            if ent.label_ == 'PERSON':
                entity_counts['person'] += 1
            elif ent.label_ in ['ORG', 'ORGANIZATION']:
                entity_counts['org'] += 1
            elif ent.label_ == 'GPE':
                entity_counts['gpe'] += 1
            elif ent.label_ == 'PRODUCT':
                entity_counts['product'] += 1
        
        return entity_counts
    
    except Exception as e:
        print(f"Error in NER: {e}")
        return {
            'person': 0,
            'org': 0,
            'gpe': 0,
            'product': 0,
            'total_entities': 0
        }


def extract_topics_simple(text, top_n=3):
    """
    Simple topic extraction using keyword frequency.
    More advanced topic modeling would use LDA/NMF.
    
    Args:
        text (str): Input text
        top_n (int): Number of topics/keywords to return
        
    Returns:
        str: Pipe-separated top topics/keywords
    """
    if not text or text == '':
        return ''
    
    try:
        # Simple approach: extract important noun phrases
        blob = TextBlob(text.lower())
        
        # Get noun phrases
        noun_phrases = blob.noun_phrases
        
        # Count frequency
        phrase_counts = Counter(noun_phrases)
        
        # Get top N
        top_phrases = [phrase for phrase, count in phrase_counts.most_common(top_n)]
        
        return '|'.join(top_phrases[:top_n])
    
    except Exception as e:
        print(f"Error in topic extraction: {e}")
        return ''


def extract_advanced_features(df):
    """
    Extract advanced NLP features for all texts.
    
    Args:
        df (pd.DataFrame): DataFrame with 'body_text' column
        
    Returns:
        pd.DataFrame: DataFrame with advanced features added
    """
    print("Extracting advanced NLP features...")
    print("  - Sentiment analysis...")
    
    # Sentiment analysis
    sentiment_results = df['body_text'].apply(analyze_sentiment)
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    sentiment_df.columns = ['sentiment_polarity', 'sentiment_subjectivity']
    
    print("  - Named Entity Recognition...")
    # Named Entity Recognition
    ner_results = df['body_text'].apply(extract_named_entities)
    ner_df = pd.DataFrame(ner_results.tolist())
    
    print("  - Topic extraction...")
    # Topic extraction
    df['topics'] = df['body_text'].apply(lambda x: extract_topics_simple(x, top_n=3))
    
    # Combine all features
    df = pd.concat([df, sentiment_df, ner_df], axis=1)
    
    print("  ✓ Advanced NLP features extracted!")
    
    return df

