"""
Feature Engineering Module
Extracts readability scores, keywords, and embeddings
"""

import pandas as pd
import numpy as np
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import re


class FeatureExtractor:
    def __init__(self):
        """Initialize feature extractor with models."""
        print("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    
    def clean_text(self, text):
        """Clean text for processing."""
        if pd.isna(text) or text == '':
            return ''
        # Lowercase and remove extra whitespace
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_basic_features(self, text):
        """
        Extract basic text metrics.
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Basic features (word_count, sentence_count, flesch_reading_ease)
        """
        if not text or text == '':
            return {
                'word_count': 0,
                'sentence_count': 0,
                'flesch_reading_ease': 0
            }
        
        word_count = len(text.split())
        sentence_count = textstat.sentence_count(text)
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'flesch_reading_ease': flesch_reading_ease
        }
    
    def extract_keywords(self, texts, top_n=5):
        """
        Extract top keywords using TF-IDF.
        
        Args:
            texts (list): List of text documents
            top_n (int): Number of top keywords to return
            
        Returns:
            list: List of keyword strings (pipe-separated)
        """
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t != '']
        
        if len(valid_texts) == 0:
            return [''] * len(texts)
        
        try:
            # Fit TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top keywords for each document
            keywords_list = []
            valid_idx = 0
            
            for text in texts:
                if text and text != '':
                    # Get top keywords for this document
                    scores = tfidf_matrix[valid_idx].toarray()[0]
                    top_indices = scores.argsort()[-top_n:][::-1]
                    top_keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
                    keywords_list.append('|'.join(top_keywords[:top_n]))
                    valid_idx += 1
                else:
                    keywords_list.append('')
            
            return keywords_list
        
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return [''] * len(texts)
    
    def extract_embeddings(self, texts):
        """
        Extract sentence embeddings.
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Embedding matrix
        """
        # Handle empty texts
        processed_texts = [t if t and t != '' else 'empty' for t in texts]
        
        try:
            embeddings = self.embedding_model.encode(processed_texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            print(f"Error extracting embeddings: {e}")
            return np.zeros((len(texts), 384))  # Default dimension for all-MiniLM-L6-v2
    
    def extract_all_features(self, df):
        """
        Extract all features from extracted content DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with 'body_text' column
            
        Returns:
            pd.DataFrame: DataFrame with all features
        """
        print("Extracting basic features...")
        # Clean text
        df['body_text_clean'] = df['body_text'].apply(self.clean_text)
        
        # Extract basic features
        basic_features = df['body_text_clean'].apply(self.extract_basic_features)
        basic_features_df = pd.DataFrame(basic_features.tolist())
        
        # If word_count already exists from parsing, use it instead of recalculating
        # Update other features but keep original word_count
        if 'word_count' in df.columns:
            # Keep original word_count, update other features
            if 'word_count' in basic_features_df.columns:
                basic_features_df = basic_features_df.drop(columns=['word_count'])
        
        df = pd.concat([df, basic_features_df], axis=1)
        
        print("Extracting keywords...")
        # Extract keywords
        df['top_keywords'] = self.extract_keywords(df['body_text_clean'].tolist())
        
        print("Extracting embeddings...")
        # Extract embeddings
        embeddings = self.extract_embeddings(df['body_text_clean'].tolist())
        
        # Convert embeddings to string representation for CSV storage
        df['embedding'] = [str(emb.tolist()) for emb in embeddings]
        
        # Also store embeddings as numpy array (for duplicate detection)
        df['embedding_array'] = [emb for emb in embeddings]
        
        # Select relevant columns for output
        feature_columns = ['url', 'word_count', 'sentence_count', 
                          'flesch_reading_ease', 'top_keywords', 'embedding']
        
        return df[feature_columns], embeddings

