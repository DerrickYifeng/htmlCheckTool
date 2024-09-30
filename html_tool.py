import streamlit as st
import pandas as pd
import pickle
from bs4 import BeautifulSoup
# import lime
# from lime.lime_tabular import LimeTabularExplainer
# import matplotlib.pyplot as plt

# Define the functions you already have
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

# Mock extract_text_fea function - implement your actual logic here
def extract_text_fea(text, label):
    return {f'{label}_len': len(text)}

# Function to load the model and get feature names
def load_model_and_features(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    # Try to access feature names from the model
    feature_names = None
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    elif hasattr(model, 'get_booster'):
        # If model is XGBoost, get feature names from the booster
        booster = model.get_booster()
        feature_names = booster.feature_names
    
    return model, feature_names

# Dashboard code
def main():
    st.title("HTML Promotion Predictor Dashboard")

    st.write("Upload an HTML file to extract features and predict promotion probability.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an HTML file", type="html")

    # Subject input field
    subject = st.text_input("Enter the email subject", "Demo Subject")

    # Load the model and get its feature names
    model_path = 'promotion_xgb.pkl'
    model, model_features = load_model_and_features(model_path)

    if uploaded_file is not None and subject:
        # Read the uploaded file
        content = uploaded_file.read().decode("utf-8")

        # Extract HTML features
        html_fea = extract_html_fea(content)

        # Extract subject features
        subject_fea = extract_text_fea(subject, "email_subject")

        # Combine features
        combined_fea = {**html_fea, **subject_fea}

        # Convert to DataFrame
        df = pd.DataFrame([combined_fea])

        # FIXME: model-feature naming correction
        df['css_to_html_ration'] = df['css_to_html_ratio']
        df['text_content_len'] = df['text_len']

        # Display extracted features
        st.write("### Extracted HTML Features:")
        st.dataframe(df)

        # Check for missing features
        if model_features is not None:
            missing_features = set(model_features) - set(df.columns)
            
            if missing_features:
                st.write("### Model Expected Features:")
                st.write(model_features)
                st.write("### Missing Features from Data:")
                st.write(missing_features)
                
        # Generate predictions if no missing features
        if not missing_features:
            df = df[model_features]
            try:
                # Assume that the model expects features in a specific order
                prediction = model.predict_proba(df)[:, 0] # Assuming binary classification
                st.write("### Promotion Probability:")
                st.write(f'### {prediction[0]:.2%}') # Display the probability of promotion

            except Exception as e:
                st.write("Error generating prediction:", e)
        else:
            st.write("Model feature names could not be determined. Please verify the model structure.")

if __name__ == '__main__':
    main()
