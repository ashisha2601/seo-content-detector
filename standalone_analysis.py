"""
Standalone Advanced NLP & Visualization Analysis
No external utils dependencies
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

def analyze_sentiment(text):
    """Analyze sentiment of text."""
    if not text or text == '':
        return {'polarity': 0.0, 'subjectivity': 0.0}
    
    try:
        blob = TextBlob(str(text)[:5000])  # Limit text length
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    except Exception as e:
        print(f"Error: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0}

def main():
    print("="*70)
    print("ADVANCED NLP & VISUALIZATION ANALYSIS")
    print("="*70)
    print()
    
    # Load data
    print("📊 Loading data...")
    try:
        df = pd.read_csv('data/extracted_content.csv')
        print(f"   Loaded {len(df)} content items")
    except FileNotFoundError:
        print("   ERROR: data/extracted_content.csv not found!")
        return
    
    try:
        features_df = pd.read_csv('data/features.csv')
        print(f"   Loaded {len(features_df)} feature rows")
    except FileNotFoundError:
        print("   Features file not found, using extracted content only")
        features_df = df.copy()
    
    # Sample for analysis
    sample_size = min(100, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    print(f"   Using sample of {len(df_sample)} items\n")
    
    # ====================================================================
    # 1. SENTIMENT ANALYSIS
    # ====================================================================
    print("💭 1. SENTIMENT ANALYSIS")
    print("-" * 70)
    
    print("   Analyzing sentiment...")
    sentiment_results = []
    for idx, text in enumerate(df_sample['body_text']):
        if idx % 20 == 0:
            print(f"   Progress: {idx}/{len(df_sample)}")
        sentiment_results.append(analyze_sentiment(text))
    
    sentiment_df = pd.DataFrame(sentiment_results)
    df_sample['sentiment_polarity'] = sentiment_df['polarity']
    df_sample['sentiment_subjectivity'] = sentiment_df['subjectivity']
    
    print(f"\n   Average Polarity: {df_sample['sentiment_polarity'].mean():.3f}")
    print(f"   Average Subjectivity: {df_sample['sentiment_subjectivity'].mean():.3f}")
    print(f"   Polarity Range: [{df_sample['sentiment_polarity'].min():.3f}, {df_sample['sentiment_polarity'].max():.3f}]")
    
    # Sentiment visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Polarity distribution
    axes[0].hist(df_sample['sentiment_polarity'], bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df_sample['sentiment_polarity'].mean(), color='red', linestyle='--', linewidth=2)
    axes[0].set_title('Sentiment Polarity Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Polarity (-1: Negative, +1: Positive)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.3)
    
    # Subjectivity distribution
    axes[1].hist(df_sample['sentiment_subjectivity'], bins=25, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(df_sample['sentiment_subjectivity'].mean(), color='red', linestyle='--', linewidth=2)
    axes[1].set_title('Sentiment Subjectivity Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Subjectivity (0: Objective, 1: Subjective)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.3)
    
    # Scatter plot
    scatter = axes[2].scatter(df_sample['sentiment_polarity'], df_sample['sentiment_subjectivity'], 
                             c=df_sample['sentiment_polarity'], cmap='RdYlGn', alpha=0.6, s=80, edgecolors='black')
    axes[2].set_title('Polarity vs Subjectivity', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Polarity')
    axes[2].set_ylabel('Subjectivity')
    axes[2].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=axes[2], label='Polarity')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('notebooks/sentiment_analysis.png', dpi=150, bbox_inches='tight')
    print("\n   ✓ Saved: notebooks/sentiment_analysis.png")
    plt.close()
    
    # ====================================================================
    # 2. WORD CLOUD
    # ====================================================================
    print("\n☁️  2. WORD CLOUD GENERATION")
    print("-" * 70)
    
    all_text = ' '.join(df_sample['body_text'].dropna().astype(str).tolist())
    print(f"   Creating word cloud from {len(all_text):,} characters...")
    
    try:
        wordcloud = WordCloud(
            width=1400,
            height=700,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            min_font_size=10
        ).generate(all_text)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Content Word Cloud - Top 100 Terms', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('notebooks/wordcloud.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: notebooks/wordcloud.png")
        plt.close()
    except Exception as e:
        print(f"   Error creating word cloud: {e}")
    
    # ====================================================================
    # 3. TEXT STATISTICS
    # ====================================================================
    print("\n📊 3. TEXT STATISTICS")
    print("-" * 70)
    
    df_sample['text_length'] = df_sample['body_text'].str.len()
    df_sample['word_count'] = df_sample['body_text'].str.split().str.len()
    
    print(f"   Average Text Length: {df_sample['text_length'].mean():.0f} characters")
    print(f"   Average Word Count: {df_sample['word_count'].mean():.0f} words")
    print(f"   Median Word Count: {df_sample['word_count'].median():.0f} words")
    
    # Text statistics visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Word count distribution
    axes[0].hist(df_sample['word_count'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df_sample['word_count'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f"Mean: {df_sample['word_count'].mean():.0f}")
    axes[0].set_title('Word Count Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Text length distribution
    axes[1].hist(df_sample['text_length'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(df_sample['text_length'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {df_sample['text_length'].mean():.0f}")
    axes[1].set_title('Text Length Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Characters')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('notebooks/text_statistics.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: notebooks/text_statistics.png")
    plt.close()
    
    # ====================================================================
    # 4. QUALITY DISTRIBUTION (if available)
    # ====================================================================
    print("\n📈 4. QUALITY DISTRIBUTION")
    print("-" * 70)
    
    if 'quality_label' in features_df.columns:
        quality_dist = features_df['quality_label'].value_counts()
        print("   Quality Label Distribution:")
        for label, count in quality_dist.items():
            pct = (count / len(features_df)) * 100
            print(f"     {label}: {count} ({pct:.1f}%)")
        
        # Quality distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        colors_qual = ['#ff6b6b', '#ffd93d', '#4ecdc4']
        axes[0].bar(quality_dist.index, quality_dist.values, color=colors_qual, edgecolor='black', alpha=0.8)
        axes[0].set_title('Quality Label Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Quality Label')
        axes[0].set_ylabel('Count')
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(quality_dist.values):
            axes[0].text(i, v + max(quality_dist.values)*0.02, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        axes[1].pie(quality_dist.values, labels=quality_dist.index, colors=colors_qual, 
                    autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        axes[1].set_title('Quality Label Proportion', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('notebooks/quality_distribution.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: notebooks/quality_distribution.png")
        plt.close()
    else:
        print("   Quality labels not available")
    
    # ====================================================================
    # 5. READABILITY (if available)
    # ====================================================================
    print("\n📖 5. READABILITY ANALYSIS")
    print("-" * 70)
    
    if 'flesch_reading_ease' in features_df.columns:
        avg_score = features_df['flesch_reading_ease'].mean()
        print(f"   Average Flesch Reading Ease: {avg_score:.1f}")
        
        if avg_score >= 90:
            interp = "Very Easy (5th grade)"
        elif avg_score >= 80:
            interp = "Easy (6th grade)"
        elif avg_score >= 70:
            interp = "Fairly Easy (7th grade)"
        elif avg_score >= 60:
            interp = "Standard (8th-9th grade)"
        elif avg_score >= 50:
            interp = "Fairly Difficult (10th-12th grade)"
        elif avg_score >= 30:
            interp = "Difficult (College)"
        else:
            interp = "Very Difficult (Graduate)"
        
        print(f"   Interpretation: {interp}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(features_df['flesch_reading_ease'].dropna(), bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.axvline(avg_score, color='red', linestyle='--', linewidth=2, label=f"Mean: {avg_score:.1f}")
        ax.set_title('Readability Score Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Flesch Reading Ease Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('notebooks/readability_distribution.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: notebooks/readability_distribution.png")
        plt.close()
    else:
        print("   Readability scores not available")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total Documents Analyzed: {len(df_sample)}")
    print(f"   Average Sentiment Polarity: {df_sample['sentiment_polarity'].mean():.3f}")
    print(f"   Average Sentiment Subjectivity: {df_sample['sentiment_subjectivity'].mean():.3f}")
    print(f"   Average Word Count: {df_sample['word_count'].mean():.0f}")
    
    print(f"\n📁 Output files saved in: notebooks/")
    print("   - sentiment_analysis.png")
    print("   - wordcloud.png")
    print("   - text_statistics.png")
    if 'quality_label' in features_df.columns:
        print("   - quality_distribution.png")
    if 'flesch_reading_ease' in features_df.columns:
        print("   - readability_distribution.png")
    print("="*70)
    print("\n✨ Open the notebooks/ folder to view the generated visualizations!")

if __name__ == "__main__":
    main()

