import streamlit as st
from openai import AzureOpenAI
# Streamlit App
st.title("Email Subject Line Generator")

# User Inputs
st.sidebar.header("Azure Configuration")
api_key = st.sidebar.text_input("Azure API Key", type="password")
azure_endpoint = "https://iegg-gpt.openai.azure.com/"
azure_deployment = "iegg-gpt-4"
api_version = "2023-12-01-preview"

st.header("Email Content")
email_content = st.text_area("Enter the email content:")
email_subject_line = st.text_input("Provide an existing email subject line (optional):")
user_guidance = st.text_area("Add guidance for the AI (optional):", placeholder="E.g., Make the title formal, catchy, or casual.")

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
            3. **Avoid Promotion or Spam Keywords**: Avoid keywords that are likely to be labeled as promotion or spam.
            """
          # Adding user guidance and existing subject line to the prompt if provided
        if user_guidance:
            sub_gen_prompt += f"\n# Additional Guidance\n{user_guidance}"
        sub_gen_prompt += "\n# Output\nDirectly output the subject lines based on the given email content.\n"
        
        if email_subject_line:
            email_content = f"\n# Existing Subject Line\n{email_subject_line}\n# Email Content\n{email_content}"

        
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
                temperature=1,
                n=n_subject
            )
            
            # Extract and Display Results
            res_txts = [x.message.content for x in response.choices]
            st.subheader("Generated Subject Lines")
            for i, subject in enumerate(res_txts, 1):
                st.write(f"{i}. {subject}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
