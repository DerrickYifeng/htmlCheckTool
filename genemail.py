import streamlit as st
from openai import OpenAI
import time

# Streamlit App
st.title("Email Subject Line Generator")

# User Inputs
st.sidebar.header("API Configuration")
api_key = st.sidebar.text_input("API Key", type="password")

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

def generate(query, tasks_prompt, user_api_key, max_retries=3):
    # Format API key correctly - avoid duplicate @smartlink
    formatted_api_key = user_api_key
    if not user_api_key.endswith('@smartlink'):
        formatted_api_key = f'{user_api_key}@smartlink'
    
    # Initialize OpenAI client with proxy settings and timeout
    client = OpenAI(
        api_key=formatted_api_key,
        base_url='http://gptproxy.ai.levelinfinite.com/gpt',
        timeout=60.0  # 60 seconds timeout
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": tasks_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.8,
                max_tokens=2048,
            )
            
            res = response.choices[0].message.content
            if res.startswith('```json'):
                res = res[8:-3]
            return res
            
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed, retrying... ({str(e)[:100]})")
                time.sleep(2)  # Wait 2 seconds before retry
                continue
            else:
                # Last attempt failed
                raise e

# Generate Button
if st.button("Generate Subject Lines"):
    if not api_key or not email_content:
        st.error("Please provide the API key and email content.")
    else:
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

        try:
            # Show progress
            progress_bar = st.progress(0)
            st.info("Generating subject lines...")
            
            # Generate multiple subject lines
            res_txts = []
            for i in range(n_subject):
                progress_bar.progress((i + 1) / n_subject)
                result = generate(email_content, sub_gen_prompt, api_key)
                res_txts.append(result)
            
            progress_bar.empty()
            
            # Display Results
            st.subheader("Generated Subject Lines")
            for i, subject in enumerate(res_txts, 1):
                st.write(f"{i}. {subject}")
                
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                st.error("⚠️ **Connection Timeout**: The proxy server is not responding. This often happens on Streamlit Cloud due to network restrictions.")
                st.info("💡 **Suggestions:**\n- Try running locally instead of Streamlit Cloud\n- Contact your IT team about proxy server accessibility\n- Consider using a different API endpoint")
            elif "401" in error_msg:
                st.error("🔑 **Authentication Error**: Invalid API key format.")
                st.info("Please check your API key and try again.")
            else:
                st.error(f"❌ **Error occurred**: {error_msg}")
                st.info("Please try again or contact support if the issue persists.")
