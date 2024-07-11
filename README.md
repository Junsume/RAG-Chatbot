# MindMend Chatbot

Welcome to MindMend, an empathetic chatbot designed to provide support to individuals facing mental trauma. This project leverages Pinecone for vector storage, Langchain for chaining models, and Streamlit and Chainlit for the user interface. 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [Streamlit](#streamlit)
  - [Chainlit](#chainlit)
- [Configuration](#configuration)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Pinecone API Key
- Hugging Face Hub API Token

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/mindmend-chatbot.git
   cd mindmend-chatbot
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file in the root directory and add your API keys:

   ```
   PINECONE_API_KEY=your_pinecone_api_key
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_hub_api_token
   ```

## Usage

### Streamlit

To run the chatbot using Streamlit:

1. **Run the Streamlit app:**

   ```bash
   streamlit run streamlit.py
   ```

2. **Open your browser:**

   The Streamlit app will typically run on `http://localhost:8501`.

### Chainlit

To run the chatbot using Chainlit:

1. **Run the Chainlit app:**

   ```bash
   chainlit run chainlit.py
   ```

2. **Open your browser:**

   The Chainlit app will typically run on `http://localhost:8000`.

## Configuration

The configuration of the chatbot is managed through environment variables and the `main.py` file. Key parameters include:

- `PINECONE_API_KEY`: Your Pinecone API key.
- `HUGGINGFACEHUB_API_TOKEN`: Your Hugging Face Hub API token.
- `index_name`: The name of the Pinecone index (default: `langchain-fdemo`).
- `namespace`: The namespace within the Pinecone index (default: `english`).

## Project Structure

```bash
mindmend-chatbot/
├── main.py            # Main logic for the chatbot
├── streamlit.py       # Streamlit interface
├── chainlit.py        # Chainlit interface
├── requirements.txt   # List of dependencies
├── .env               # Environment variables
└── README.md          # This readme file
```


## Detailed Explanation

### main.py

This file contains the core logic for the RAGChatbot. The chatbot uses Pinecone for vector storage and retrieval and HuggingFace for embeddings and language models.

- **RAGChatbot Class:**
  - `__init__`: Initializes the Pinecone client, embeddings, and QA chain.
  - `initialize_index`: Checks if the specified Pinecone index exists; if not, it creates one.
  - `initialize_qa_chain`: Sets up the Hugging Face endpoint and prompt template for the QA chain.
  - `chat`: Invokes the QA chain with the user's query.

### streamlit.py

This file provides a user interface for the chatbot using Streamlit.

- Displays a sidebar with the title "MindMend".
- Initializes the chatbot and handles user input and responses.
- Stores chat messages in `st.session_state` and displays them in a chat format.

### chainlit.py

This file provides a user interface for the chatbot using Chainlit.

- Defines `on_chat_start` to send a welcome message when the chat starts.
- Defines `on_message` to handle incoming messages and send responses using the chatbot.

By following the instructions above, you can set up and run the MindMend Chatbot to provide empathetic support to users facing mental trauma.
