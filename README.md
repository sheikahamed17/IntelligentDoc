# Intelligent Document Q&A

This project is a web-based application built with Streamlit that enables you to have intelligent conversations with your documents. It leverages a powerful Retrieval-Augmented Generation (RAG) pipeline to provide accurate answers based on the content of your knowledge base.

## Key Features
*   **Interactive Chat Interface:** Ask questions in natural language and get answers from your documents, complete with source tracking.
*   **Flexible RAG Backends:** Choose between using Google's state-of-the-art Generative AI models or a local Ollama instance for processing and generation.
*   **Multiple Document Sources:** Build your knowledge base by uploading PDFs, text files, or adding content directly from URLs.
*   **Knowledge Base Management:** Easily view the documents in your vector database and clear the knowledge base through the UI.

## Tech Stack
*   **Frontend:** Streamlit
*   **LLM Providers:** Google AI (Gemini), Ollama
*   **Vector Database:** ChromaDB
*   **Text Processing:** Langchain

## Getting Started

### Prerequisites
*   Python 3.7+
*   Google API Key (if using the Google AI pipeline)

### Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/sheikahamed17/IntelligentDoc.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Set up your environment variables. If you are using the Google AI pipeline, you will need to set your `GOOGLE_API_KEY`.
    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

### Running the Application
1.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
2.  Open your browser and navigate to the URL provided by Streamlit.
