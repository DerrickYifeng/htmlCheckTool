# Databricks notebook source
# COMMAND ----------

# Install required packages if not already installed
!pip install shap pandas beautifulsoup4 matplotlib

# COMMAND ----------

import pandas as pd
import pickle
from bs4 import BeautifulSoup
import shap
import matplotlib.pyplot as plt
import re
import base64
from io import BytesIO

# COMMAND ----------

# Define helper functions
def normalize_text(s, sep_token=" \n "):
    s = s.lower()
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r". ,","",s)
    s = s.replace("..", ".")
    s = s.replace(". .", ".")
    s = s.replace("\n", "")
    s = s.strip()
    return s

def get_text_to_html_ratio(content, soup):
    text_length = get_text_len(soup)
    html_length = len(content)
    return text_length / html_length if html_length != 0 else 0

def get_text_len(soup):
    return len(soup.get_text())

def get_num_links(soup):
    return get_tag_nums(soup, 'a')

def get_num_imgs(soup):
    return get_tag_nums(soup, 'img')

def get_tag_nums(soup, tag_type):
    return len(soup.find_all(tag_type))

def get_all_tag_nums(soup):
    tags = soup.find_all()
    tag_count = len(tags)
    tag_types = set(tag.name for tag in tags)
    return tag_count, len(tag_types)

def get_max_depth(soup):
    if hasattr(soup, "contents") and soup.contents:
        return max([get_max_depth(child) for child in soup.contents]) + 1
    else:
        return 0

def get_external_resource_num(soup):
    return len(soup.find_all(['img', 'link', 'script']))

def get_css_num(soup):
    return get_tag_nums(soup, 'style')

def get_css_len(soup):
    style_tags = soup.find_all('style')
    inline_css = ''.join(tag.string for tag in style_tags if tag.string)
    return len(inline_css.encode('utf-8'))

def extract_html_fea(content):
    soup = BeautifulSoup(content, 'html.parser')
    dic = {}
    dic['content_len'] = len(content)
    text_content = soup.get_text()
    dic['text_len'] = len(text_content)
    dic['text_to_html_ratio'] = get_text_to_html_ratio(content, soup)
    dic['num_links'] = get_num_links(soup)
    dic['num_imgs'] = get_num_imgs(soup)

    tag_count, tag_type_cnt = get_all_tag_nums(soup)
    dic['tag_count'] = tag_count
    dic['tag_type_cnt'] = tag_type_cnt

    dic['html_max_depth'] = get_max_depth(soup)
    dic['external_resources_count'] = get_external_resource_num(soup)
    dic['num_css_tag'] = get_css_num(soup)

    css_len = get_css_len(soup)
    dic['css_to_html_ratio'] = css_len / len(content)

    return dic

def extract_text_fea(text, label):
    return {f'{label}_len': len(text)}

# COMMAND ----------

# Load the model
model_path = '/dbfs/path/to/your/promotion_xgb.pkl'  # Update this path to your model location
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Get feature names
feature_names = None
if hasattr(model, 'feature_names_in_'):
    feature_names = model.feature_names_in_
elif hasattr(model, 'get_booster'):
    booster = model.get_booster()
    feature_names = booster.feature_names

# COMMAND ----------

# Function to process HTML and get prediction
def predict_promotion(html_content, subject):
    # Extract features
    html_fea = extract_html_fea(html_content)
    subject_fea = extract_text_fea(subject, "email_subject")
    
    # Combine features
    combined_fea = {**html_fea, **subject_fea}
    
    # Convert to DataFrame
    df = pd.DataFrame([combined_fea])
    
    # Fix feature names to match model
    df['css_to_html_ration'] = df['css_to_html_ratio']
    df['text_content_len'] = df['text_len']
    
    # Select only the features needed by the model
    df_model = df[feature_names]
    
    # Get prediction
    prediction = model.predict_proba(df_model)[:, 0][0]
    
    # Get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df_model)
    
    # Create SHAP plot
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0])
    
    # Convert plot to base64 string for display
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    return {
        'promotion_probability': prediction,
        'shap_plot': plot_base64,
        'features': df_model.to_dict('records')[0]
    }

# COMMAND ----------

# Example usage
# Replace these with your actual HTML content and subject
sample_html = """
<html>
<body>
<h1>Sample HTML</h1>
<p>This is a test email.</p>
</body>
</html>
"""
sample_subject = "Test Subject"

result = predict_promotion(sample_html, sample_subject)

# Display results
print(f"Promotion Probability: {result['promotion_probability']:.4f}")
print("\nFeature Values:")
for feature, value in result['features'].items():
    print(f"{feature}: {value}")

# Display SHAP plot
displayHTML(f'<img src="data:image/png;base64,{result["shap_plot"]}"/>') 