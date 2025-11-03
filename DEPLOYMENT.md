# Deployment Guide - SEO Content Detector on Streamlit Cloud

## Overview

This guide will walk you through deploying the SEO Content Detector application to Streamlit Cloud, making it publicly accessible.

## Prerequisites

Before deploying, ensure you have:

- ✅ A GitHub account
- ✅ The repository pushed to GitHub (`https://github.com/ashisha2601/seo-content-detector`)
- ✅ A Streamlit Cloud account (free tier available)
- ✅ Python 3.9+ installed locally
- ✅ All dependencies in `requirements.txt`

## Step-by-Step Deployment

### Step 1: Verify Your GitHub Repository

1. Visit https://github.com/ashisha2601/seo-content-detector
2. Confirm all files are present, especially:
   - `streamlit_app.py` (main app file)
   - `requirements.txt` (dependencies)
   - `models/quality_model.pkl` (trained model)
   - `data/` folder with CSV files
   - `utils/` folder with all utility modules
   - `.streamlit/config.toml` (Streamlit configuration)

3. The repository should show as public or private (depending on your preference)

### Step 2: Create Streamlit Cloud Account

1. Go to https://share.streamlit.io
2. Click "Sign up"
3. Choose "Sign up with GitHub"
4. Authorize Streamlit to access your GitHub account
5. Verify your email

### Step 3: Deploy the App

1. After signing in to Streamlit Cloud, click **"New app"** button

2. **Configure Deployment Settings:**
   - **Repository**: Select `ashisha2601/seo-content-detector`
   - **Branch**: Select `main`
   - **Main file path**: Type `streamlit_app.py`

3. Click **"Deploy!"**

4. Streamlit will:
   - Clone your repository
   - Install dependencies from `requirements.txt`
   - Start your Streamlit app
   - Provide you with a public URL

### Step 4: Monitor Deployment

1. You'll see a progress screen showing:
   - "Installing packages..."
   - "Launching app..."

2. The deployment typically takes 2-5 minutes

3. Once complete, your app URL will be displayed in the format:
   ```
   https://share.streamlit.io/ashisha2601/seo-content-detector/main/streamlit_app.py
   ```

## Using Your Deployed App

### Features Available

1. **URL Analysis Tab**
   - Paste a URL to analyze instantly
   - Get quality scores and readability metrics
   - Detect similar/duplicate content

2. **Batch Upload Tab**
   - Upload CSV file with multiple URLs
   - Analyze in bulk
   - Download results

3. **Dataset Analysis Tab**
   - View statistics about your dataset
   - Interactive visualizations
   - Explore feature distributions

4. **About Tab**
   - Project documentation
   - Technology stack
   - Usage instructions

## Troubleshooting

### Issue: "Module not found" error

**Cause**: Missing dependency in `requirements.txt`

**Solution**:
1. Update `requirements.txt` with missing package
2. Push changes to GitHub:
   ```bash
   git add requirements.txt
   git commit -m "Add missing dependency"
   git push
   ```
3. Reboot the app from Streamlit Cloud dashboard
   - Click the three dots (⋮) next to your app
   - Select "Reboot app"

### Issue: "File not found" error (data or models)

**Cause**: File paths are relative in code

**Solution**:
1. Ensure all file paths are relative (not absolute)
2. Check that files exist in the repository:
   ```bash
   git status  # Should show no untracked files needed
   ```
3. Verify in the `streamlit_app.py`:
   ```python
   model_path = "models/quality_model.pkl"  # ✅ Correct
   # NOT: /Users/ashishasharma/Desktop/... (absolute path)
   ```

### Issue: App is slow or timing out

**Cause**: Large models or computations taking too long

**Solution**:
1. Use caching effectively (Streamlit `@st.cache_resource`)
2. Optimize feature extraction
3. Consider upgrading to Streamlit Cloud Pro for more resources

### Issue: "GitHub token" or "Permission denied"

**Cause**: Streamlit doesn't have access to your repository

**Solution**:
1. Make repository public, or
2. Reconnect your GitHub account:
   - Go to account settings
   - "Manage GitHub access"
   - Reauthorize Streamlit

## Advanced Configuration

### Custom Domain

To use a custom domain (e.g., seo-detector.yourdomain.com):

1. Go to your app dashboard
2. Click "Settings" (⚙️)
3. Navigate to "Custom domain"
4. Follow DNS configuration instructions

### Secrets Management

If you need to store API keys or credentials:

1. Go to your app dashboard
2. Click "Settings" (⚙️)
3. Navigate to "Secrets"
4. Add your secrets in TOML format:

   ```toml
   [database]
   url = "postgres://user:password@host/db"
   
   [api]
   key = "your-api-key-here"
   ```

5. Access in your app:
   ```python
   import streamlit as st
   
   db_url = st.secrets["database"]["url"]
   api_key = st.secrets["api"]["key"]
   ```

### Environment Variables

For environment-specific settings, create `.streamlit/config.toml`:

```toml
[client]
showErrorDetails = true

[logger]
level = "info"

[server]
maxUploadSize = 200
```

## Monitoring & Maintenance

### View App Logs

1. Go to your app dashboard
2. Click the app name
3. Logs appear at the bottom
4. Use logs to debug issues

### Update Your App

To deploy new changes:

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Your changes description"
   git push origin main
   ```

3. Streamlit Cloud will automatically redeploy within seconds

### View Traffic & Usage

1. Dashboard shows "Deployment" information
2. Monitor app performance and usage metrics
3. (Pro feature) More detailed analytics available with paid plan

## Performance Tips

1. **Cache Everything Possible**
   ```python
   @st.cache_resource
   def load_model():
       return joblib.load("model.pkl")
   ```

2. **Optimize Data Loading**
   - Load only necessary columns
   - Use filters to reduce data size

3. **Lazy Load Features**
   - Don't compute features until needed
   - Use expanders to hide non-essential info

4. **Monitor Resource Usage**
   - Streamlit Cloud free tier: 1GB RAM, 30s execution limit
   - Optimize model size and computation time

## Common Streamlit Cloud Limits

| Feature | Free Tier | Pro Tier |
|---------|-----------|----------|
| RAM | 1 GB | 8 GB |
| Execution Time | 30 seconds | Unlimited |
| File Upload Size | 200 MB | 2 GB |
| Concurrent Users | 5 | Unlimited |

## Getting Help

- **Streamlit Docs**: https://docs.streamlit.io
- **Community**: https://discuss.streamlit.io
- **Issues**: https://github.com/streamlit/streamlit/issues
- **Your Repo**: https://github.com/ashisha2601/seo-content-detector/issues

## Post-Deployment Checklist

- [ ] App URL is accessible
- [ ] URL analysis works
- [ ] Batch upload functionality works
- [ ] Dataset analysis displays correctly
- [ ] Visualizations load properly
- [ ] No console errors in app logs
- [ ] Share app URL with users/stakeholders

## Sharing Your App

Once deployed, you can share your app:

1. **Direct URL**: Share the Streamlit URL
2. **Shorten URL**: Use bit.ly or similar to create short links
3. **Embed**: Embed in websites using iframe
4. **Social Media**: Share on LinkedIn, Twitter, etc.

**Example Markdown for sharing:**
```markdown
## 🔍 SEO Content Detector

Analyze web content for SEO quality and duplicate detection.

🚀 **Try it now**: https://share.streamlit.io/ashisha2601/seo-content-detector/main/streamlit_app.py

### Features:
- Real-time URL analysis
- Batch processing
- Dataset visualizations
```

## Next Steps

1. ✅ Deploy to Streamlit Cloud
2. Test all features thoroughly
3. Gather user feedback
4. Iterate and improve
5. Consider upgrading to Pro for production use

---

**Last Updated**: November 2025
**Status**: Ready for Deployment ✅
