# 🤖 Conversational AI Chatbot with Llama-3.1 & LangChain

This project is a web-based conversational AI chatbot built with a powerful tech stack. It leverages the **NVIDIA Llama-3.1 Nemotron 70B** model via Hugging Face for text generation, **LangChain** for managing conversation flow and memory, and **Streamlit** for creating an interactive user interface.

The chatbot is designed to maintain the context of the conversation, allowing for more natural and coherent interactions.

## ✨ Key Features

  - **Powerful Language Model**: Powered by the state-of-the-art `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` model.
  - **Conversational Memory**: Remembers previous parts of the conversation using LangChain's `ConversationBufferMemory`.
  - **Interactive UI**: A clean, user-friendly chat interface built with Streamlit, featuring styled chat bubbles.
  - **Modular and Extensible**: The use of LangChain's `ConversationChain` makes it easy to modify prompts, memory types, or even swap out the LLM.
  - **Clear Chat History**: A simple button to clear the conversation and reset the chatbot's memory.

-----

## 🛠️ Tech Stack

  - **Web Framework**: [Streamlit](https://streamlit.io/)
  - **Orchestration Framework**: [LangChain](https://www.langchain.com/)
  - **Model Hub & Library**: [Hugging Face Transformers](https://huggingface.co/transformers)
  - **Core LLM**: [NVIDIA Llama-3.1 Nemotron 70B](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF)

-----

## 🚀 How It Works

The application's architecture is straightforward:

1.  **Model Loading**: A text-generation pipeline is initialized using the `transformers` library to load the Llama-3.1 model from Hugging Face.
2.  **LangChain Integration**: The Hugging Face pipeline is wrapped in LangChain's `HuggingFacePipeline` class, making it compatible with the LangChain ecosystem.
3.  **Memory Management**: An instance of `ConversationBufferMemory` is created to automatically store and append conversation history to the prompt.
4.  **Conversation Chain**: A `ConversationChain` is constructed, linking the **LLM**, **memory**, and a **prompt template**. This chain handles the logic of passing the user's input along with the chat history to the model.
5.  **Streamlit UI**: The user interface is built with Streamlit. It captures user input, sends it to the `ConversationChain` for processing, and displays the user's message and the AI's response in a styled chat format.

-----

## 🖥️ Setup and Installation

To run this application locally, follow these steps.

**Prerequisites**:

  - Python 3.8+
  - `pip` package manager
  - A powerful GPU with sufficient VRAM is **highly recommended** to run the 70B parameter model effectively.

<!-- end list -->

1.  **Clone the Repository**

    ```bash
    git clone https://your-repository-url.git
    cd your-repository-directory
    ```

2.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:

    ```
    streamlit
    langchain
    transformers
    torch
    accelerate
    ```

    Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: You may need to install a specific version of PyTorch that matches your CUDA version for GPU support.*

3.  **Run the Application**
    Execute the following command in your terminal:

    ```bash
    streamlit run app.py
    ```

    The application will open in your default web browser.
