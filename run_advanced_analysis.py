"""
Advanced NLP & Visualization Analysis Script
Generates comprehensive analysis with visualizations
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.advanced_nlp import analyze_sentiment, extract_named_entities
from utils.visualizations import (
    plot_similarity_heatmap, 
    plot_feature_importance, 
    plot_quality_distribution,
    create_wordcloud,
    plot_readability_distribution
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

def main():
    print("="*70)
    print("ADVANCED NLP & VISUALIZATION ANALYSIS")
    print("="*70)
    print()
    
    # Load data
    print("📊 Loading data...")
    df = pd.read_csv('data/extracted_content.csv')
    print(f"   Loaded {len(df)} content items")
    
    try:
        features_df = pd.read_csv('data/features.csv')
        print(f"   Loaded {len(features_df)} feature rows")
    except FileNotFoundError:
        print("   Features file not found, will use extracted content only")
        features_df = df.copy()
    
    # Sample for analysis
    sample_size = min(100, len(df))
    df_sample = df.sample(n=sample_size, random_state=42).copy()
    print(f"   Using sample of {len(df_sample)} items for analysis\n")
    
    # ====================================================================
    # 1. SENTIMENT ANALYSIS
    # ====================================================================
    print("💭 1. SENTIMENT ANALYSIS")
    print("-" * 70)
    
    sentiment_results = df_sample['body_text'].apply(analyze_sentiment)
    sentiment_df = pd.DataFrame(sentiment_results.tolist())
    df_sample['sentiment_polarity'] = sentiment_df['polarity']
    df_sample['sentiment_subjectivity'] = sentiment_df['subjectivity']
    
    print(f"   Average Polarity: {df_sample['sentiment_polarity'].mean():.3f}")
    print(f"   Average Subjectivity: {df_sample['sentiment_subjectivity'].mean():.3f}")
    print(f"   Polarity Range: [{df_sample['sentiment_polarity'].min():.3f}, {df_sample['sentiment_polarity'].max():.3f}]")
    print(f"   Subjectivity Range: [{df_sample['sentiment_subjectivity'].min():.3f}, {df_sample['sentiment_subjectivity'].max():.3f}]")
    
    # Sentiment visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Polarity distribution
    axes[0].hist(df_sample['sentiment_polarity'], bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(df_sample['sentiment_polarity'].mean(), color='red', linestyle='--', linewidth=2, 
                    label=f"Mean: {df_sample['sentiment_polarity'].mean():.3f}")
    axes[0].set_title('Sentiment Polarity Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Polarity (-1: Negative, +1: Positive)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Subjectivity distribution
    axes[1].hist(df_sample['sentiment_subjectivity'], bins=25, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(df_sample['sentiment_subjectivity'].mean(), color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {df_sample['sentiment_subjectivity'].mean():.3f}")
    axes[1].set_title('Sentiment Subjectivity Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Subjectivity (0: Objective, 1: Subjective)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
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
    print("   ✓ Saved: notebooks/sentiment_analysis.png")
    plt.show()
    print()
    
    # ====================================================================
    # 2. NAMED ENTITY RECOGNITION
    # ====================================================================
    print("🏷️  2. NAMED ENTITY RECOGNITION (NER)")
    print("-" * 70)
    
    ner_results = df_sample['body_text'].apply(extract_named_entities)
    ner_df = pd.DataFrame(ner_results.tolist())
    
    for col in ner_df.columns:
        df_sample[col] = ner_df[col]
    
    print(f"   Total Entities Found: {df_sample['total_entities'].sum()}")
    print(f"   Person Entities: {df_sample['person'].sum()}")
    print(f"   Organization Entities: {df_sample['org'].sum()}")
    print(f"   Location Entities (GPE): {df_sample['gpe'].sum()}")
    print(f"   Product Entities: {df_sample['product'].sum()}")
    print(f"   Avg Entities per Document: {df_sample['total_entities'].mean():.2f}")
    
    # NER visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    entity_totals = {
        'Person': df_sample['person'].sum(),
        'Organization': df_sample['org'].sum(),
        'Location': df_sample['gpe'].sum(),
        'Product': df_sample['product'].sum()
    }
    
    colors = ['#ff6b6b', '#4ecdc4', '#95e1d3', '#ffd93d']
    axes[0].bar(entity_totals.keys(), entity_totals.values(), color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_title('Named Entity Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Total Count')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (k, v) in enumerate(entity_totals.items()):
        axes[0].text(i, v + max(entity_totals.values())*0.02, str(v), ha='center', fontweight='bold')
    
    # Box plot
    entity_data = [df_sample['person'], df_sample['org'], df_sample['gpe'], df_sample['product']]
    bp = axes[1].boxplot(entity_data, labels=['Person', 'Org', 'Location', 'Product'],
                         patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1].set_title('Entity Distribution per Document', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count per Document')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('notebooks/ner_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: notebooks/ner_analysis.png")
    plt.show()
    print()
    
    # ====================================================================
    # 3. WORD CLOUD
    # ====================================================================
    print("☁️  3. WORD CLOUD GENERATION")
    print("-" * 70)
    
    all_text = ' '.join(df_sample['body_text'].dropna().astype(str).tolist())
    print(f"   Creating word cloud from {len(all_text):,} characters...")
    
    wordcloud_fig = create_wordcloud(all_text, max_words=100, figsize=(14, 8))
    if wordcloud_fig:
        wordcloud_fig.savefig('notebooks/wordcloud.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: notebooks/wordcloud.png")
        plt.show()
    print()
    
    # ====================================================================
    # 4. QUALITY DISTRIBUTION
    # ====================================================================
    print("📈 4. QUALITY DISTRIBUTION")
    print("-" * 70)
    
    if 'quality_label' in features_df.columns:
        quality_dist = features_df['quality_label'].value_counts()
        print("   Quality Label Distribution:")
        for label, count in quality_dist.items():
            pct = (count / len(features_df)) * 100
            print(f"     {label}: {count} ({pct:.1f}%)")
        
        features_sample = features_df.sample(n=min(200, len(features_df)), random_state=42)
        fig = plot_quality_distribution(features_sample, figsize=(14, 6))
        fig.savefig('notebooks/quality_distribution.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: notebooks/quality_distribution.png")
        plt.show()
    else:
        print("   Quality labels not available. Run the full pipeline first.")
    print()
    
    # ====================================================================
    # 5. READABILITY ANALYSIS
    # ====================================================================
    print("📖 5. READABILITY ANALYSIS")
    print("-" * 70)
    
    if 'flesch_reading_ease' in features_df.columns:
        avg_score = features_df['flesch_reading_ease'].mean()
        print(f"   Average Flesch Reading Ease: {avg_score:.1f}")
        
        if avg_score >= 90:
            interpretation = "Very Easy (5th grade level)"
        elif avg_score >= 80:
            interpretation = "Easy (6th grade level)"
        elif avg_score >= 70:
            interpretation = "Fairly Easy (7th grade level)"
        elif avg_score >= 60:
            interpretation = "Standard (8th-9th grade level)"
        elif avg_score >= 50:
            interpretation = "Fairly Difficult (10th-12th grade level)"
        elif avg_score >= 30:
            interpretation = "Difficult (College level)"
        else:
            interpretation = "Very Difficult (Graduate level)"
        
        print(f"   Interpretation: {interpretation}")
        
        features_sample = features_df.sample(n=min(200, len(features_df)), random_state=42)
        fig = plot_readability_distribution(features_sample, figsize=(12, 6))
        fig.savefig('notebooks/readability_distribution.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: notebooks/readability_distribution.png")
        plt.show()
    else:
        print("   Readability scores not available. Run the full pipeline first.")
    print()
    
    # ====================================================================
    # 6. CONTENT SIMILARITY HEATMAP
    # ====================================================================
    print("🔥 6. CONTENT SIMILARITY HEATMAP")
    print("-" * 70)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("   Generating embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        sim_sample = df_sample.head(15)
        embeddings = model.encode(sim_sample['body_text'].tolist(), show_progress_bar=False)
        
        fig = plot_similarity_heatmap(embeddings, sim_sample['url'].tolist(), top_n=15, figsize=(12, 10))
        fig.savefig('notebooks/similarity_heatmap.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: notebooks/similarity_heatmap.png")
        plt.show()
    except ImportError:
        print("   sentence-transformers not installed. Skipping similarity analysis.")
    except Exception as e:
        print(f"   Could not create similarity heatmap: {e}")
    print()
    
    # ====================================================================
    # 7. FEATURE IMPORTANCE
    # ====================================================================
    print("🎯 7. FEATURE IMPORTANCE")
    print("-" * 70)
    
    try:
        import pickle
        
        with open('models/quality_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("   Top 10 Most Important Features:")
            for idx, row in importance_df.head(10).iterrows():
                print(f"     {row['feature']}: {row['importance']:.4f}")
            
            fig = plot_feature_importance(importance_df.head(15), figsize=(12, 8))
            fig.savefig('notebooks/feature_importance.png', dpi=150, bbox_inches='tight')
            print("   ✓ Saved: notebooks/feature_importance.png")
            plt.show()
        else:
            print("   Model does not have feature_importances_ attribute")
    except FileNotFoundError:
        print("   Model file not found. Run the full pipeline to train the model first.")
    except Exception as e:
        print(f"   Error loading model: {e}")
    print()
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\n📊 Summary:")
    print(f"   Total Documents Analyzed: {len(df_sample)}")
    print(f"   Average Sentiment Polarity: {df_sample['sentiment_polarity'].mean():.3f}")
    print(f"   Average Sentiment Subjectivity: {df_sample['sentiment_subjectivity'].mean():.3f}")
    print(f"   Total Named Entities Found: {df_sample['total_entities'].sum()}")
    print(f"\n📁 Output files saved in: notebooks/")
    print("   - sentiment_analysis.png")
    print("   - ner_analysis.png")
    print("   - wordcloud.png")
    if 'quality_label' in features_df.columns:
        print("   - quality_distribution.png")
    if 'flesch_reading_ease' in features_df.columns:
        print("   - readability_distribution.png")
    print("="*70)

if __name__ == "__main__":
    main()

