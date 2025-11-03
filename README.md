# SEO Content Quality & Duplicate Detector

A machine learning pipeline that analyzes web content for SEO quality assessment and duplicate detection. This system processes HTML content, extracts meaningful features, detects near-duplicate content, and scores content quality using ML models.

## Project Overview

This project builds a comprehensive content analysis system that:
- Processes pre-scraped HTML content efficiently
- Extracts structured features (readability, keywords, embeddings)
- Detects near-duplicate content using similarity algorithms
- Scores content quality with ML models
- Provides real-time analysis via Jupyter notebook

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/ashisha2601/seo-content-detector
cd seo-content-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** Requires Python 3.9 or higher.

3. Place your dataset (`data.csv`) in the `data/` folder. The CSV should contain:
   - `url`: The webpage URL
   - `html_content`: Raw HTML content (pre-scraped)

4. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/seo_pipeline.ipynb
```

## Quick Start

1. Ensure `data.csv` is in the `data/` folder
2. Open and run `notebooks/seo_pipeline.ipynb` cell by cell
3. The notebook will generate:
   - `extracted_content.csv`: Parsed content (title, body_text, word_count)
   - `features.csv`: Extracted features (readability, keywords, embeddings)
   - `duplicates.csv`: Duplicate pairs detected
   - `quality_model.pkl`: Trained quality classifier

## Key Decisions

- **HTML Parsing**: Used BeautifulSoup4 for robust HTML parsing with fallback error handling
- **Feature Engineering**: Combined textstat for readability scores, TF-IDF for keywords, and sentence-transformers for semantic embeddings
- **Similarity Threshold**: Set at 0.80 (cosine similarity) based on content analysis best practices
- **Model Selection**: Random Forest classifier chosen for interpretability and handling of mixed feature types
- **Embeddings**: Sentence-transformers (all-MiniLM-L6-v2) for efficient semantic similarity computation

## Results Summary

After running the notebook with your dataset, update this section with:
- **Model Performance**: Accuracy and F1-score from model evaluation
- **Duplicate Detection**: Number of duplicate pairs found and thin content statistics
- **Quality Distribution**: Count of High/Medium/Low quality pages

## Limitations

- HTML parsing may fail for dynamically loaded content (JavaScript-heavy pages)
- Similarity threshold may need adjustment based on domain-specific content
- Quality labels are synthetic and may not reflect actual SEO performance

## Streamlit App (Bonus - Optional)

An interactive Streamlit app is available for real-time URL analysis.

### Running Locally

```bash
streamlit run streamlit_app/app.py
```

### Deployed Streamlit App

🌐 **Streamlit Cloud URL:** [To be added after deployment]

*Note: After deploying to Streamlit Cloud, update this URL with your deployed app link.*

### Deployment Instructions

1. Push your code to GitHub
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set the app path to: `streamlit_app/app.py`
5. Deploy!

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
├── streamlit_app/                    # Streamlit app (bonus)
│   ├── app.py                        # Main Streamlit app
│   ├── utils/                        # Utility modules
│   │   ├── parser.py
│   │   ├── features.py
│   │   └── scorer.py
│   └── models/
│       └── quality_model.pkl         # Saved model
├── models/
│   └── quality_model.pkl             # Saved model
├── utils/                            # Utility modules for notebook
│   ├── parser.py
│   ├── features.py
│   ├── scorer.py
│   └── analyzer.py
├── requirements.txt
├── .gitignore
└── README.md
```

