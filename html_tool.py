import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

# Define the functions you already have
def normalize_text(s, sep_token=" \n "):
    s = s.lower()
    s = re.sub(r'\s+',  ' ', s).strip()
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
    return {f'{label}_length': len(text)}

# Dashboard code
def main():
    st.title("HTML Feature Extractor Dashboard")

    st.write("Upload an HTML file to extract features.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an HTML file", type="html")

    if uploaded_file is not None:
        # Read the uploaded file
        content = uploaded_file.read().decode("utf-8")

        # Extract HTML features
        html_fea = extract_html_fea(content)

        # For demo, using subject as part of content (You can modify this)
        subject = "Demo Subject"  # Modify this to extract from content if needed
        subject_fea = extract_text_fea(subject, "email_subject")

        # Combine features
        combined_fea = {**html_fea, **subject_fea}

        # Convert to DataFrame and display
        df = pd.DataFrame([combined_fea])
        
        st.write("### Extracted HTML Features:")
        st.dataframe(df)

if __name__ == '__main__':
    main()
