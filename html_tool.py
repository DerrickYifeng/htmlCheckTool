import streamlit as st
import pandas as pd
import pickle
from bs4 import BeautifulSoup
import shap
import matplotlib.pyplot as plt

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

    st.write("Upload one or multiple HTML files to extract features and predict promotion probability.")

    # File uploader
    uploaded_files = st.file_uploader("Choose HTML files", type="html", accept_multiple_files=True)

    # Subject input field
    subject = st.text_input("Enter the email subject", "Demo Subject")

    # Load the model and get its feature names
    model_path = 'promotion_xgb.pkl'
    model, model_features = load_model_and_features(model_path)

    if uploaded_files and subject:
        # List to store features of all files
        all_features = []

        for uploaded_file in uploaded_files:
            # Read the uploaded file
            content = uploaded_file.read().decode("utf-8")

            # Extract HTML features
            html_fea = extract_html_fea(content)

            # Extract subject features
            subject_fea = extract_text_fea(subject, "email_subject")

            # Combine features and add file name as a column
            combined_fea = {**html_fea, **subject_fea}
            combined_fea['file_name'] = uploaded_file.name

            # Append to the list of all features
            all_features.append(combined_fea)

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Move 'file_name' to the first column
        df = df[['file_name'] + [col for col in df.columns if col != 'file_name']]

        # FIXME: model-feature naming correction
        df['css_to_html_ration'] = df['css_to_html_ratio']
        df['text_content_len'] = df['text_len']
        df.set_index('file_name', inplace=True)
        # Display combined features for all files
        st.write("### Extracted HTML Features for All Files:")
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
            df_model = df[model_features]
            try:
                # Assume that the model expects features in a specific order
                predictions = model.predict_proba(df_model)[:, 0] # Assuming binary classification
                
                # Add predictions to the DataFrame
                df['promotion_probability'] = predictions

                # Display predictions for each file
                st.write("### Promotion Probabilities:")
                st.dataframe(df[['promotion_probability']])

                # SHAP explanation
                st.write("### SHAP Waterfall Plot for Each Prediction")

                # Generate SHAP explanation for each row
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(df_model)

                # Loop through each prediction and display its waterfall plot
                for i, (index, row) in enumerate(df_model.iterrows()):
                    st.write(f"#### Waterfall Plot for {index}")
                    plt.figure()
                    # Use shap_values[i] directly as it is already an Explanation object
                    shap.plots.waterfall(shap_values[i])
                    st.pyplot(plt)

            except Exception as e:
                st.write("Error generating predictions:", e)
        else:
            st.write("Model feature names could not be determined. Please verify the model structure.")

if __name__ == '__main__':
    main()