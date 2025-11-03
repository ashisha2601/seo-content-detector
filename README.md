# SEO Content Quality & Duplicate Detector

A machine learning pipeline that analyzes web content for SEO quality assessment and duplicate detection. This system processes HTML content, extracts meaningful features, detects near-duplicate content, and scores content quality using ML models.

## Project Overview

This project builds a comprehensive content analysis system that:
- Processes pre-scraped HTML content efficiently
- Extracts structured features (readability, keywords, embeddings)
- Detects near-duplicate content using similarity algorithms
- Scores content quality with ML models
- Provides real-time analysis via Jupyter notebook
- **NEW: Interactive Streamlit web application for real-time URL analysis**

## Setup Instructions

### 1. Clone the repository:
```bash
git clone https://github.com/ashisha2601/seo-content-detector
cd seo-content-detector
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** Requires Python 3.9 or higher.

### 3. Place your dataset (`data.csv`) in the `data/` folder. The CSV should contain:
   - `url`: The webpage URL
   - `html_content`: Raw HTML content (pre-scraped)

## Quick Start

### Running Jupyter Notebook:
```bash
jupyter notebook notebooks/seo_pipeline.ipynb
```

The notebook will generate:
- `extracted_content.csv`: Parsed content (title, body_text, word_count)
- `features.csv`: Extracted features (readability, keywords, embeddings)
- `duplicates.csv`: Duplicate pairs detected
- `quality_model.pkl`: Trained quality classifier

### Running the Streamlit App Locally:
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`

## Features

### 🔗 Analyze URL
- Paste any URL to get instant analysis
- Metrics: Word count, sentence count, readability score, quality label
- Duplicate detection with reference dataset
- Thin content warnings

### 📤 Batch Upload
- Upload CSV file with multiple URLs
- Process bulk URLs with progress tracking
- Download analysis results as CSV

### 📊 Dataset Analysis
- View dataset statistics and visualizations
- Word count distribution histogram
- Readability score distribution
- Content quality pie chart
- Feature data table with sorting

### ℹ️ About
- Project overview and technology stack
- Model details and parameters
- Usage instructions and limitations

## Key Decisions

- **HTML Parsing**: Used BeautifulSoup4 for robust HTML parsing with fallback error handling
- **Feature Engineering**: Combined textstat for readability scores, TF-IDF for keywords, and sentence-transformers for semantic embeddings
- **Similarity Threshold**: Set at 0.80 (cosine similarity) based on content analysis best practices
- **Model Selection**: Random Forest classifier chosen for interpretability and handling of mixed feature types
- **Embeddings**: Sentence-transformers (all-MiniLM-L6-v2) for efficient semantic similarity computation

## Deployment to Streamlit Cloud

### Prerequisites
- GitHub account with the repository pushed
- Streamlit Cloud account (free at https://streamlit.io/cloud)

### Deployment Steps

1. **Push code to GitHub** (already done! ✅)
   ```bash
   git push -u origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit https://share.streamlit.io
   - Sign in with your GitHub account
   - Click "New app"

3. **Configure Deployment**
   - **Repository**: Select `ashisha2601/seo-content-detector`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **App URL**: Will be auto-generated

4. **Deploy**
   - Click "Deploy!"
   - Wait for deployment to complete (2-3 minutes)
   - Your app will be live!

### Deployed App URL
🌐 **Streamlit Cloud**: [Your deployed URL will appear here after deployment]

*After deploying to Streamlit Cloud, your app will be accessible at: `https://share.streamlit.io/ashisha2601/seo-content-detector/main/streamlit_app.py`*

### Environment Variables (if needed)
If you need to set environment variables on Streamlit Cloud:
1. Go to your app settings
2. Click "Secrets"
3. Add your environment variables in TOML format

## Project Structure

```
seo-content-detector/
├── data/
│   ├── data.csv                      # Provided dataset
│   ├── extracted_content.csv         # Parsed content
│   ├── features.csv                  # Extracted features
│   └── duplicates.csv                # Duplicate pairs
├── notebooks/
│   └── seo_pipeline.ipynb            # Main notebook
│   ├── readability_distribution.png  # Visualization
│   ├── sentiment_analysis.png        # Visualization
│   └── ...
├── models/
│   └── quality_model.pkl             # Saved ML model
├── utils/
│   ├── __init__.py
│   ├── parser.py                     # HTML parsing
│   ├── features.py                   # Feature extraction
│   ├── scorer.py                     # Quality scoring & duplicates
│   ├── analyzer.py                   # Real-time analysis
│   ├── advanced_nlp.py               # Advanced NLP features
│   └── visualizations.py             # Visualization utilities
├── .streamlit/
│   └── config.toml                   # Streamlit configuration
├── .gitignore
├── requirements.txt
├── streamlit_app.py                  # Streamlit application
├── run_advanced_analysis.py          # Advanced analysis script
├── standalone_analysis.py            # Standalone script
└── README.md                          # This file
```

## Model Performance

- **Accuracy**: [To be updated with your dataset]
- **F1-Score**: [To be updated with your dataset]
- **Duplicate Pairs Found**: [To be updated]
- **Quality Distribution**: [To be updated]

## Limitations

- HTML parsing may fail for dynamically loaded content (JavaScript-heavy pages)
- Similarity threshold may need adjustment based on domain-specific content
- Quality labels are synthetic and may not reflect actual SEO performance
- Streamlit Cloud has memory and execution time limits for free tier

## Technology Stack

### Core
- **Python 3.9+**
- **Pandas** - Data processing
- **NumPy** - Numerical computing

### Machine Learning & NLP
- **scikit-learn** - ML models and metrics
- **Sentence-Transformers** - Semantic embeddings
- **spaCy** - Advanced NLP
- **TextBlob** - Sentiment analysis
- **textstat** - Readability metrics

### Web & Visualization
- **Streamlit** - Interactive web app
- **Plotly** - Interactive visualizations
- **BeautifulSoup4** - HTML parsing
- **Requests** - HTTP requests

### Data & Utilities
- **Jupyter** - Notebooks
- **Matplotlib & Seaborn** - Static visualizations
- **Joblib** - Model persistence

## Installation Troubleshooting

### Issue: ImportError for specific packages
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt --upgrade
```

### Issue: spaCy model not found
**Solution**: Download the required spaCy model
```bash
python -m spacy download en_core_web_sm
```

### Issue: Streamlit app won't start
**Solution**: Clear Streamlit cache and try again
```bash
streamlit cache clear
streamlit run streamlit_app.py
```

## API Reference

### RealTimeAnalyzer
```python
from utils.analyzer import RealTimeAnalyzer

analyzer = RealTimeAnalyzer(
    reference_embeddings=embeddings,
    reference_urls=urls,
    quality_model=model
)

result = analyzer.analyze_url("https://example.com")
```

### analyze_url() Output
```python
{
    'url': str,                    # Input URL
    'word_count': int,             # Number of words
    'sentence_count': int,         # Number of sentences
    'readability': float,          # Flesch Reading Ease score
    'quality_label': str,          # 'High', 'Medium', or 'Low'
    'is_thin': bool,               # True if word_count < 500
    'similar_to': [                # List of similar content
        {
            'url': str,
            'similarity': float     # Cosine similarity score
        }
    ]
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please reach out via GitHub issues.

---

**Last Updated**: November 2025
**Status**: ✅ Production Ready for Streamlit Cloud Deployment

