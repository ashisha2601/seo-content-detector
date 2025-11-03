"""
Real-time URL Analysis Module
Provides analyze_url() function for real-time content analysis
"""

import requests
import time
from bs4 import BeautifulSoup
import re
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .parser import extract_text_from_html
from .features import FeatureExtractor
from .scorer import create_quality_labels, detect_thin_content


class RealTimeAnalyzer:
    def __init__(self, reference_embeddings=None, reference_urls=None, quality_model=None):
        """
        Initialize real-time analyzer.
        
        Args:
            reference_embeddings (numpy.ndarray): Embeddings from reference dataset
            reference_urls (list): URLs from reference dataset
            quality_model: Trained quality classification model
        """
        self.feature_extractor = FeatureExtractor()
        self.reference_embeddings = reference_embeddings
        self.reference_urls = reference_urls
        self.quality_model = quality_model
    
    def scrape_url(self, url, delay=1.0):
        """
        Scrape HTML content from URL.
        
        Args:
            url (str): URL to scrape
            delay (float): Delay between requests (seconds)
            
        Returns:
            str: HTML content
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            time.sleep(delay)  # Rate limiting
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def analyze_url(self, url, similarity_threshold=0.75):
        """
        Analyze a single URL and return quality metrics.
        
        Args:
            url (str): URL to analyze
            similarity_threshold (float): Threshold for duplicate detection
            
        Returns:
            dict: Analysis results
        """
        print(f"Analyzing {url}...")
        
        # Scrape HTML content
        html_content = self.scrape_url(url)
        if not html_content:
            return {
                'url': url,
                'error': 'Failed to scrape URL'
            }
        
        # Extract text
        title, body_text, word_count = extract_text_from_html(html_content)
        
        if word_count == 0:
            return {
                'url': url,
                'error': 'No content extracted'
            }
        
        # Extract features
        cleaned_text = self.feature_extractor.clean_text(body_text)
        basic_features = self.feature_extractor.extract_basic_features(cleaned_text)
        
        # Get embedding
        embedding = self.feature_extractor.extract_embeddings([cleaned_text])[0]
        
        # Detect thin content
        temp_df = pd.DataFrame([{
            'word_count': word_count,
            'flesch_reading_ease': basic_features['flesch_reading_ease']
        }])
        temp_df = detect_thin_content(temp_df)
        is_thin = temp_df['is_thin'].iloc[0] == 1
        
        # Predict quality
        quality_label = 'Medium'  # Default
        if self.quality_model:
            try:
                features = np.array([[
                    basic_features['word_count'],
                    basic_features['sentence_count'],
                    basic_features['flesch_reading_ease']
                ]])
                quality_label = self.quality_model.predict(features)[0]
            except Exception as e:
                print(f"Error predicting quality: {e}")
        else:
            # Fallback to rule-based
            temp_df = create_quality_labels(temp_df)
            quality_label = temp_df['quality_label'].iloc[0]
        
        # Find similar URLs
        similar_to = []
        if self.reference_embeddings is not None and self.reference_urls is not None:
            similarities = cosine_similarity([embedding], self.reference_embeddings)[0]
            similar_indices = np.where(similarities >= similarity_threshold)[0]
            
            for idx in similar_indices:
                similar_to.append({
                    'url': self.reference_urls[idx],
                    'similarity': round(float(similarities[idx]), 4)
                })
        
        # Compile results
        result = {
            'url': url,
            'word_count': word_count,
            'sentence_count': basic_features['sentence_count'],
            'readability': round(basic_features['flesch_reading_ease'], 2),
            'quality_label': quality_label,
            'is_thin': is_thin,
            'similar_to': similar_to
        }
        
        return result

