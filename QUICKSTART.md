# Quick Start Guide - SEO Content Detector

## 🚀 Get Started in 5 Minutes

### Option 1: Use the Live Web App (Easiest ✅)

No installation needed! Use the Streamlit Cloud hosted version:

```
🌐 https://share.streamlit.io/ashisha2601/seo-content-detector/main/streamlit_app.py
```

Just open the link and start analyzing!

---

### Option 2: Run Locally (Your Machine)

#### Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/ashisha2601/seo-content-detector.git
cd seo-content-detector

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model (one-time)
python -m spacy download en_core_web_sm
```

#### Run the App (30 seconds)

```bash
streamlit run streamlit_app.py
```

The app opens automatically at: `http://localhost:8501`

---

## 📊 How to Use the App

### 1️⃣ Analyze a Single URL

**Tab**: "🔗 Analyze URL"

```
1. Paste a URL (e.g., https://example.com)
2. Click "Analyze URL"
3. View metrics:
   - Word Count
   - Sentence Count
   - Readability Score
   - Quality Label (High/Medium/Low)
   - Similar Content (if found)
```

### 2️⃣ Analyze Multiple URLs (Batch)

**Tab**: "📤 Batch Upload"

```
1. Create a CSV file with URLs:
   url
   https://example1.com
   https://example2.com
   https://example3.com

2. Upload the CSV
3. Click "Analyze Batch"
4. Download results as CSV
```

### 3️⃣ Explore Dataset Statistics

**Tab**: "📊 Dataset Analysis"

```
- View dataset statistics
- See visualizations:
  - Word count distribution
  - Readability distribution
  - Quality breakdown
- Browse feature data table
```

### 4️⃣ Learn About the Project

**Tab**: "ℹ️ About"

```
- Project overview
- Technology stack
- Model details
- Troubleshooting tips
```

---

## 📈 Output Metrics Explained

| Metric | Meaning | Interpretation |
|--------|---------|-----------------|
| **Word Count** | Total words in content | Higher = More comprehensive |
| **Sentences** | Number of sentences | Shows content structure |
| **Readability** | Flesch Reading Ease score | 60-70 = Easy to read, <30 = Hard |
| **Quality Label** | Predicted quality | High/Medium/Low classification |
| **Thin Content** | Warning | ⚠️ Less than 500 words = Thin |
| **Similarity** | Duplicate score | 0.80+ = Likely duplicate |

---

## 💡 Tips & Tricks

### Tip 1: URL Format
Include the full URL with protocol:
- ✅ `https://example.com`
- ❌ `example.com`

### Tip 2: Batch Processing Speed
- Processing time depends on content size
- Typical: 2-5 seconds per URL
- Batch of 10 URLs: ~30-60 seconds

### Tip 3: Best Results
- Use URLs that are publicly accessible
- Static HTML pages work better than JavaScript-heavy sites
- Simple, text-based content = most accurate

### Tip 4: Customize Similarity Threshold
Want to find similar (but not identical) content?
- Lower threshold = more matches
- Higher threshold = only very similar

---

## 🔧 Troubleshooting

### "Failed to scrape URL"
**Cause**: Website blocked bots or not accessible
**Solution**: 
- Check if URL is correct
- Try in a browser first
- Some sites block automated access

### "No content extracted"
**Cause**: Content is likely dynamic (JavaScript-based)
**Solution**:
- Try a different URL with static HTML
- Most news sites, blogs = work well

### "Module not found" error
**Cause**: Dependencies not installed
**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

### App runs slowly
**Cause**: Large URL list or model loading
**Solution**:
- First run loads models (takes longer)
- Subsequent runs are faster (cached)
- Batch upload: process 5-10 URLs at a time

---

## 📝 Example Workflows

### Workflow 1: Check Single Blog Post
```
1. Copy blog post URL
2. Use "Analyze URL" tab
3. Check readability & quality score
4. Improve if quality = Low
```

### Workflow 2: Find Duplicate Content
```
1. Upload CSV with your site URLs
2. Use "Batch Upload" tab
3. Review "Similar Content" column
4. Merge or rewrite duplicates
```

### Workflow 3: Dataset Analysis
```
1. Run the Jupyter notebook first
   jupyter notebook notebooks/seo_pipeline.ipynb
2. View generated features.csv
3. Use "Dataset Analysis" tab
4. Understand quality distribution
5. Create improvement plan
```

---

## 🎯 Common Use Cases

### Use Case 1: Content Audit
- Upload all your URLs
- Analyze for quality issues
- Identify thin content pages
- Find duplicate content

### Use Case 2: Competitor Analysis
- Paste competitor URLs
- Compare readability scores
- Check content length
- Benchmark quality

### Use Case 3: Content Writing
- Analyze before publishing
- Check readability
- Verify against duplicates
- Get quality score

### Use Case 4: SEO Optimization
- Identify pages needing improvement
- Focus on thin content
- Remove duplicates
- Optimize readability

---

## 📚 Learn More

| Topic | Resource |
|-------|----------|
| **Full Documentation** | [README.md](README.md) |
| **Deployment Guide** | [DEPLOYMENT.md](DEPLOYMENT.md) |
| **GitHub Repository** | [github.com/ashisha2601/seo-content-detector](https://github.com/ashisha2601/seo-content-detector) |
| **Streamlit Docs** | [docs.streamlit.io](https://docs.streamlit.io) |

---

## 🆘 Need Help?

- **Questions?** Open an [issue on GitHub](https://github.com/ashisha2601/seo-content-detector/issues)
- **Found a bug?** Report it with error messages and steps to reproduce
- **Feature request?** Describe what you'd like to see

---

## ✅ Quick Checklist

Before analyzing content:
- [ ] Internet connection available
- [ ] URL is publicly accessible
- [ ] URL includes `https://` or `http://`
- [ ] CSV has `url` column (for batch)

---

**Ready to analyze?** 🚀

👉 **Start here**: [SEO Content Detector App](https://share.streamlit.io/ashisha2601/seo-content-detector/main/streamlit_app.py)

---

**Last Updated**: November 2025
