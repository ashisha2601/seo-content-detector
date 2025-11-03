import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from utils.analyzer import RealTimeAnalyzer
from utils.parser import extract_text_from_html
from utils.features import FeatureExtractor

# Page configuration
st.set_page_config(
    page_title="SEO Content Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .quality-high { color: #28a745; font-weight: bold; }
    .quality-medium { color: #ffc107; font-weight: bold; }
    .quality-low { color: #dc3545; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None

@st.cache_resource
def load_quality_model():
    """Load pre-trained quality model"""
    model_path = "models/quality_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_resource
def load_reference_data():
    """Load reference embeddings and URLs for duplicate detection"""
    features_path = "data/features.csv"
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        if 'url' in df.columns and 'embedding' in df.columns:
            try:
                embeddings = np.array([eval(emb) for emb in df['embedding']])
                urls = df['url'].tolist()
                return embeddings, urls
            except:
                return None, None
    return None, None

def initialize_analyzer():
    """Initialize the real-time analyzer"""
    quality_model = load_quality_model()
    embeddings, urls = load_reference_data()
    
    analyzer = RealTimeAnalyzer(
        reference_embeddings=embeddings,
        reference_urls=urls,
        quality_model=quality_model
    )
    return analyzer

def get_quality_color(quality_label):
    """Get color for quality label"""
    colors = {
        'High': '🟢',
        'Medium': '🟡',
        'Low': '🔴'
    }
    return colors.get(quality_label, '⚪')

def display_analysis_results(result):
    """Display analysis results in a nice format"""
    if 'error' in result:
        st.error(f"❌ Error: {result['error']}")
        return
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Word Count", result['word_count'])
    
    with col2:
        st.metric("Sentences", result['sentence_count'])
    
    with col3:
        st.metric("Readability", f"{result['readability']}")
    
    with col4:
        quality = result['quality_label']
        st.metric("Quality", f"{get_quality_color(quality)} {quality}")
    
    # Thin content warning
    if result['is_thin']:
        st.warning("⚠️ This page has thin content (< 500 words). Consider expanding it for better SEO.")
    
    # Similar content
    if result['similar_to']:
        st.subheader("Similar Content Detected")
        similar_df = pd.DataFrame(result['similar_to'])
        st.dataframe(similar_df, use_container_width=True)
    else:
        st.info("✅ No similar content found in reference dataset")

def main():
    # Header
    st.markdown('<div class="main-header">🔍 SEO Content Quality & Duplicate Detector</div>', unsafe_allow_html=True)
    st.markdown("Analyze web content for SEO quality, readability, and duplicate detection using AI/ML")
    st.divider()
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔗 Analyze URL", "📤 Batch Upload", "📊 Dataset Analysis", "ℹ️ About"])
    
    # TAB 1: URL Analysis
    with tab1:
        st.header("Single URL Analysis")
        st.write("Enter a URL to analyze its SEO quality and detect duplicates")
        
        url = st.text_input("Enter URL:", placeholder="https://example.com")
        
        if st.button("Analyze URL", type="primary", use_container_width=True):
            if not url:
                st.error("Please enter a valid URL")
            else:
                with st.spinner("🔄 Analyzing URL..."):
                    analyzer = initialize_analyzer()
                    result = analyzer.analyze_url(url)
                    display_analysis_results(result)
    
    # TAB 2: Batch Upload
    with tab2:
        st.header("Batch Analysis")
        st.write("Upload CSV with URLs for batch analysis")
        
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} URLs for analysis")
            
            if st.button("Analyze Batch", type="primary", use_container_width=True):
                analyzer = initialize_analyzer()
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    url = row.get('url', row.iloc[0])
                    result = analyzer.analyze_url(url)
                    results.append(result)
                    progress_bar.progress((idx + 1) / len(df))
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results",
                    data=csv,
                    file_name="seo_analysis_results.csv",
                    mime="text/csv"
                )
    
    # TAB 3: Dataset Analysis
    with tab3:
        st.header("Dataset Analysis")
        
        # Load existing data
        extracted_path = "data/extracted_content.csv"
        features_path = "data/features.csv"
        duplicates_path = "data/duplicates.csv"
        
        if os.path.exists(features_path):
            features_df = pd.read_csv(features_path)
            
            # Display statistics
            st.subheader("📈 Dataset Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total URLs", len(features_df))
            
            with col2:
                avg_word_count = features_df['word_count'].mean() if 'word_count' in features_df.columns else 0
                st.metric("Avg Word Count", f"{int(avg_word_count)}")
            
            with col3:
                avg_readability = features_df['flesch_reading_ease'].mean() if 'flesch_reading_ease' in features_df.columns else 0
                st.metric("Avg Readability", f"{avg_readability:.2f}")
            
            with col4:
                if os.path.exists(duplicates_path):
                    dup_df = pd.read_csv(duplicates_path)
                    st.metric("Duplicate Pairs", len(dup_df))
            
            # Visualizations
            st.subheader("📊 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Word count distribution
                if 'word_count' in features_df.columns:
                    fig = px.histogram(features_df, x='word_count', nbins=30, 
                                      title='Word Count Distribution',
                                      labels={'word_count': 'Word Count'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Readability distribution
                if 'flesch_reading_ease' in features_df.columns:
                    fig = px.histogram(features_df, x='flesch_reading_ease', nbins=30,
                                      title='Readability Score Distribution',
                                      labels={'flesch_reading_ease': 'Flesch Reading Ease'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Quality distribution
            if os.path.exists(extracted_path):
                extracted_df = pd.read_csv(extracted_path)
                if 'quality_label' in features_df.columns:
                    quality_counts = features_df['quality_label'].value_counts()
                    fig = px.pie(values=quality_counts.values, names=quality_counts.index,
                                title='Content Quality Distribution',
                                color_discrete_map={'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'})
                    st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("📋 Feature Data")
            st.dataframe(features_df.head(20), use_container_width=True)
        else:
            st.info("📁 No dataset found. Run the pipeline to generate features first.")
    
    # TAB 4: About
    with tab4:
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        **SEO Content Quality & Duplicate Detector** is a machine learning pipeline that analyzes web content for:
        - **SEO Quality Assessment**: Evaluates readability, word count, and content structure
        - **Duplicate Detection**: Identifies similar/duplicate content using semantic embeddings
        - **Thin Content Detection**: Flags pages with insufficient word count
        
        ### 🔧 Technology Stack
        
        - **Python 3.9+**
        - **Machine Learning**: scikit-learn, Random Forest Classifier
        - **NLP**: spaCy, Sentence-Transformers, TextBlob
        - **Readability**: textstat
        - **Web Interface**: Streamlit
        
        ### 📊 Key Features
        
        1. **Real-time URL Analysis** - Scrape and analyze any URL
        2. **Batch Processing** - Analyze multiple URLs at once
        3. **Duplicate Detection** - Find similar content using embeddings (cosine similarity > 0.80)
        4. **Quality Classification** - ML model predicts High/Medium/Low quality
        5. **Dataset Insights** - Visualize feature distributions and statistics
        
        ### 🎓 Model Details
        
        - **Quality Model**: Random Forest with 100 estimators
        - **Features Used**: Word Count, Sentence Count, Readability Score
        - **Embeddings**: all-MiniLM-L6-v2 (384-dimensional vectors)
        - **Similarity Threshold**: 0.80 (cosine similarity)
        
        ### ⚠️ Limitations
        
        - JavaScript-heavy pages may not be fully parsed
        - Quality labels are synthetic and based on heuristics
        - Similarity threshold may need adjustment for different domains
        
        ### 📝 How to Use
        
        1. **URL Analysis Tab**: Paste a URL to get instant analysis
        2. **Batch Upload Tab**: Upload CSV with URLs for bulk processing
        3. **Dataset Analysis Tab**: View statistics and visualizations
        
        ### 🔗 Links
        
        - [GitHub Repository](https://github.com/ashisha2601/seo-content-detector)
        - [Documentation](https://github.com/ashisha2601/seo-content-detector#readme)
        """)

if __name__ == "__main__":
    main()
