"""
HTML Parser Module
Extracts title and body text from HTML content
"""

import re
from bs4 import BeautifulSoup
import pandas as pd


def extract_text_from_html(html_content):
    """
    Extract clean text from HTML content.
    
    Args:
        html_content (str): Raw HTML content
        
    Returns:
        tuple: (title, body_text, word_count)
    """
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else ""
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract main content - prioritize article, main, or body
        body_text = ""
        article = soup.find('article')
        main = soup.find('main')
        body = soup.find('body')
        
        if article:
            body_text = article.get_text(separator=' ', strip=True)
        elif main:
            body_text = main.get_text(separator=' ', strip=True)
        elif body:
            body_text = body.get_text(separator=' ', strip=True)
        else:
            # Fallback: get all text
            body_text = soup.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        body_text = re.sub(r'\s+', ' ', body_text).strip()
        
        # Calculate word count
        word_count = len(body_text.split()) if body_text else 0
        
        return title, body_text, word_count
    
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return "", "", 0


def parse_dataset(input_csv_path, output_csv_path):
    """
    Parse HTML content from dataset and extract text features.
    
    Args:
        input_csv_path (str): Path to input CSV with url and html_content
        output_csv_path (str): Path to save extracted content CSV
    """
    print("Reading dataset...")
    df = pd.read_csv(input_csv_path)
    
    print(f"Processing {len(df)} URLs...")
    
    results = []
    for idx, row in df.iterrows():
        url = row['url']
        html_content = row.get('html_content', '')
        
        if pd.isna(html_content) or html_content == '':
            print(f"Warning: No HTML content for {url}")
            results.append({
                'url': url,
                'title': '',
                'body_text': '',
                'word_count': 0
            })
            continue
        
        title, body_text, word_count = extract_text_from_html(html_content)
        
        results.append({
            'url': url,
            'title': title,
            'body_text': body_text,
            'word_count': word_count
        })
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} URLs...")
    
    # Create DataFrame and save
    extracted_df = pd.DataFrame(results)
    extracted_df.to_csv(output_csv_path, index=False)
    print(f"\nExtracted content saved to {output_csv_path}")
    print(f"Successfully processed: {len(extracted_df[extracted_df['word_count'] > 0])} URLs")
    
    return extracted_df

