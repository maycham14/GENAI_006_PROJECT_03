# AI-Powered Text Summarizer

## Project Overview
AI Text Summarizer is an advanced web application built using Streamlit and powered by OpenAI's GPT models. The app provides a comprehensive interface where users can upload various document formats or paste raw text to generate high-quality, customizable summaries. The application extends beyond basic summarization to offer domain-specific templates, document comparison, and detailed analytics.

## Features
- **Multiple Input Options**: Upload documents (PDF, TXT, DOCX, CSV) or paste text directly for summarization
- **Advanced AI Summarization**: Utilizes OpenAI's GPT-3.5 and GPT-4 models for generating high-quality summaries
- **Customizable Summary Types**: Choose between Short, Medium, or Detailed summaries based on your needs
- **Domain-Specific Templates**: Specialized templates for Data Science, Health, Legal, Finance, and more
- **Content Validation**: Automatic verification that templates match document content
- **Document Comparison**: Compare two documents side-by-side with similarity analysis
- **Dark/Light Mode**: User-selectable theme with full application styling
- **History Tracking**: Save and reuse previous summaries
- **Analytics Dashboard**: Track summary statistics and processing metrics
- **Export Options**: Download summaries and original text in various formats
- **User-Friendly Interface**: Intuitive interface with responsive design

## How It Works
1. **Select Summary Type**: Choose between Short (1-3 bullet points), Medium (3-4 sentence executive summary), or Detailed (comprehensive analysis)
2. **Input Method**: Upload a document (PDF, TXT, DOCX, CSV) or paste text directly
3. **Customize (Optional)**: Select a domain-specific template or create a custom prompt
4. **Generate Summary**: Press the "Generate Summary" button to create an AI-powered summary
5. **View & Export**: Review your summary, download it, or save it to your history
6. **Compare Documents**: Upload two documents to generate side-by-side summaries with comparison analysis

## Prerequisites
Before you start, ensure you have the following installed on your machine:
- **Python 3.8 or higher**
- **OpenAI API Key**: You can obtain your API key from [OpenAI](https://platform.openai.com/account/api-keys)

## Installation

### 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/maycham14/GENAI_006_PROJECT_03.git
cd AI_Text_Summarizer
```

### 2. Create a Virtual Environment
Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Required Packages
Install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Set up your API key:
- Create a folder named `KEYS` in the root directory
- Create a file named `secrets.toml` inside the `KEYS` folder
- Add your OpenAI API key to the file:
```toml
OPEN_API_KEY = "your-api-key-here"
```

### 5. Run the Application
Start the application:
```bash
streamlit run app.py
```

## Advanced Features

### Summary Types
- **Short**: 2-3 bullet points highlighting critical information
- **Medium**: 3-4 sentence executive summary covering main points
- **Detailed**: Comprehensive summary with all key points and insights

### Domain-Specific Templates
- **Default**: General-purpose summarization
- **Data Science**: Focuses on methodologies, data points, and statistical insights
- **Health**: Emphasizes patient data, medical insights, and treatments
- **Legal**: Highlights key clauses, terms, and legal implications
- **Finance**: Concentrates on market trends, revenue, and economic implications
- **Custom**: Create your own specialized prompt

### Document Comparison
The application allows you to:
1. Upload two documents
2. Select comparison detail level (Brief or Detailed)
3. View side-by-side summaries and detailed comparison analysis

### Theme Selection
The application supports both Light and Dark themes, with complete styling of all UI elements for improved user experience.

### Analytics
Track summary statistics including:
- Processing time
- Document sizes
- Compression ratios
- Historical usage patterns

## Technical Details
- **Frontend**: Streamlit
- **AI Backend**: OpenAI API (GPT-3.5 & GPT-4)
- **Document Processing**: PyMuPDF, python-docx, pandas
- **Text Processing**: NLTK, regex
- **Data Visualization**: Plotly

## Requirements
Main dependencies:
- streamlit
- openai
- fitz (PyMuPDF)
- python-docx
- pandas
- nltk
- plotly
- hashlib
- toml

For a complete list, see the requirements.txt file.

## Acknowledgments
- OpenAI for providing the powerful GPT models
- Streamlit for the interactive web application framework
- The open-source community for document processing libraries

Repository: https://github.com/maycham14/GENAI_006_PROJECT_03