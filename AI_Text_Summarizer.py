import streamlit as st
import openai
from openai import OpenAI
import time
import fitz  # type: ignore # PyMuPDF for reading PDFs

# Get your secret API key from Streamlit Secrets
api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the client with the secret key
client = OpenAI(api_key=api_key)

# Page setup
st.set_page_config(page_title="AI-Powered Text Summarizer: Supply Chain Industry", layout="centered")
st.title("📄 AI-Powered Text Summarizer")
st.write("Upload a PDF or paste text to generate an instant summary using GPT-4.")

# Summary level selection
summary_type = st.radio("Select summary type:", ["Short", "Medium", "Detailed"])
input_option = st.selectbox("Choose input method:", ["Paste text", "Upload PDF"])

text = ""

# Input: Paste text
if input_option == "Paste text":
    text = st.text_area("Enter your text here:", height=250)

# Input: Upload PDF
elif input_option == "Upload PDF":
    #uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt", "jpg", "docx"])
    st.write("File type received: ", uploaded_file.type if uploaded_file else "No file uploaded.")

    if uploaded_file:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

# Function to summarize using GPT-4
def get_summary(text, summary_type):
    prompt_styles = {
        "Short": "Summarize the following text in 2 bullet points only.",
        "Medium": "Write a 3-4 sentence executive summary of the following text.",
        "Detailed": "Write a detailed summary covering all key supply chain insights, logistics information, and action points."
    }

    prompt = f"{prompt_styles[summary_type]}\n\n{text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()

# Summarize button
if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter or upload text to summarize.")
    else:
        with st.spinner("Generating your summary..."):
            summary = get_summary(text, summary_type)

        st.success("Summary generated:")

        # Typing animation effect
        placeholder = st.empty()
        animated_text = ""
        for char in summary:
            animated_text += char
            placeholder.markdown(f"```markdown\n{animated_text}\n```")
            time.sleep(0.01)