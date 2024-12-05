import streamlit as st
from openai import AzureOpenAI

# Streamlit App
st.title("Email Subject Line Generator")

# User Inputs
st.sidebar.header("Azure Configuration")
api_key = st.sidebar.text_input("Azure API Key", type="password")
azure_endpoint = st.sidebar.text_input("Azure Endpoint", value="https://iegg-gpt.openai.azure.com/")
azure_deployment = st.sidebar.text_input("Azure Deployment Name", value="iegg-gpt-4")
api_version = st.sidebar.text_input("API Version", value="2023-12-01-preview")

st.header("Email Content")
email_content = st.text_area("Enter the email content:")
n_subject = st.number_input(
    "Number of Subject Lines",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

# Generate Button
if st.button("Generate Subject Lines"):
    if not api_key or not email_content:
        st.error("Please provide the API key and email content.")
    else:
        # Azure Configuration
        azure_config = {
            'api_key': api_key,
            'azure_endpoint': azure_endpoint,
            'azure_deployment': azure_deployment,
            'api_version': api_version
        }
        
        # Initialize Azure OpenAI Client
        client = AzureOpenAI(**azure_config)
        
        # Prompt
        sub_gen_prompt = """
            # Rules
            As a marketing expert, you are an expert in email marketing. Your task is to craft a tailored email headline based on the given email contents.

            1. **Understand and Clarify**: Make sure you understand the email content.
            2. **Subject Guides**: Keep the subject short and catchy
            3. **Avoid Promotion&Spam**: Make sure subject line is not labeled as promotion or spam.

            # Output 
            Directly output the subject lines based on the given email content

            # Format 
            XXX # the subject line 
            """
        
        # Chat History
        chat_hist = [
            {'role': 'system', 'content': sub_gen_prompt},
            {'role': 'user', 'content': email_content}
        ]
        
        try:
            # Generate Response
            response = client.chat.completions.create(
                model='gpt-4',
                messages=chat_hist,
                temperature=0.5,
                n=n_subject
            )
            
            # Extract and Display Results
            res_txts = [x.message.content for x in response.choices]
            st.subheader("Generated Subject Lines")
            for i, subject in enumerate(res_txts, 1):
                st.write(f"{i}. {subject}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
