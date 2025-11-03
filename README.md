# SEO Content Quality & Duplicate Detector

## Project Overview

A machine learning pipeline that analyzes web content for SEO quality assessment and duplicate detection. This system processes HTML content, extracts features (readability, keywords, embeddings), detects near-duplicates using semantic similarity, and scores content quality with a trained Random Forest classifier.

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/ashisha2601/seo-content-detector
cd seo-content-detector

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook
jupyter notebook notebooks/seo_pipeline.ipynb
```

**Note:** Requires Python 3.9+

## Quick Start

1. Place your dataset (`data.csv`) in the `data/` folder with columns: `url` and `html_content`
2. Open and run `notebooks/seo_pipeline.ipynb` cell by cell
3. The notebook generates:
   - `extracted_content.csv`: Parsed content (title, body text, word count)
   - `features.csv`: Extracted features (readability, keywords, embeddings)
   - `duplicates.csv`: Duplicate pairs detected (similarity ≥ 0.80)
   - `quality_model.pkl`: Trained quality classifier

## Deployed Streamlit URL

🌐 **Live App**: https://share.streamlit.io/ashisha2601/seo-content-detector/main/streamlit_app.py

**To run locally:**
```bash
streamlit run streamlit_app.py
```

## Key Decisions

- **HTML Parsing**: BeautifulSoup4 with fallback error handling for robust extraction
- **Feature Engineering**: Textstat for readability, TF-IDF for keywords, Sentence-Transformers (all-MiniLM-L6-v2) for semantic embeddings
- **Similarity Threshold**: 0.80 cosine similarity based on duplicate detection best practices
- **Model Selection**: Random Forest (100 estimators) for interpretability and handling mixed feature types
- **Embedding Model**: all-MiniLM-L6-v2 for efficient 384-dimensional semantic vectors

## Results Summary

- **Model Accuracy**: 0.87 | **F1-Score**: 0.85
- **Duplicate Pairs Found**: 12 pairs with similarity ≥ 0.80
- **Quality Distribution**: High (34%), Medium (42%), Low (24%)
- **Sample Readability Scores**: Range 25-78 (Flesch Reading Ease)
- **Thin Content Detected**: 8 pages with <500 words

## Limitations

- HTML parsing fails for JavaScript-heavy dynamic content sites
- Similarity threshold may require adjustment for domain-specific content
- Quality labels are synthetic and based on heuristics, not actual SEO performance
- Streamlit Cloud free tier: 1GB RAM, 30s execution limit

---

**Status**: ✅ Production Ready | **Last Updated**: November 2025

