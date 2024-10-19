import streamlit as st
from transformers import pipeline
from langchain import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

# Initialize the pipeline for text generation using Hugging Face
pipe = pipeline("text-generation", model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")

# Wrap the Hugging Face pipeline with LangChain's HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create memory to hold conversation history
memory = ConversationBufferMemory()

# Create a ConversationChain with the LLM and memory
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=PromptTemplate.from_template("{history}\nHuman: {input}\nAI:"),
)

# Streamlit layout enhancements
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

# Sidebar information
st.sidebar.title("🤖 Chatbot with Llama-3.1 & LangChain")
st.sidebar.markdown("""
Welcome to the AI chatbot built using the Llama-3.1 model from Hugging Face and LangChain.
- Ask me anything!
- This chatbot remembers the conversation context using LangChain memory.
""")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main interface design
st.title("🧠 AI Chatbot")
st.markdown("Ask me anything! The chatbot remembers the conversation context.")

# Custom style for chat bubbles
def render_message(role, content):
    if role == "user":
        st.markdown(f"""
        <div style='background-color:#e6f7ff; padding:10px; border-radius:10px; margin-bottom:10px; width:fit-content'>
        <strong>You:</strong> {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background-color:#f1f1f1; padding:10px; border-radius:10px; margin-bottom:10px; width:fit-content'>
        <strong>AI:</strong> {content}
        </div>
        """, unsafe_allow_html=True)

# User input field with button
user_input = st.text_input("Type your message here:", placeholder="Ask the AI...", key="input")

# Process user input and get model's response using LangChain
if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate a response using LangChain's conversation chain
    response = conversation.run(input=user_input)

    # Add bot message to session state
    st.session_state.messages.append({"role": "bot", "content": response})

# Display chat history in styled bubbles
for message in st.session_state.messages:
    render_message(message["role"], message["content"])

# Button to clear chat
if st.button("Clear Chat"):
    st.session_state.messages = []
    conversation.memory.clear()  # Clear memory as well
    st.experimental_rerun()

# Footer
st.markdown("""
    <hr style="border-top: 1px solid #e6e6e6;">
    <footer style="text-align: center;">
    Built with ❤️ using LangChain, Hugging Face, and Streamlit.
    </footer>
""", unsafe_allow_html=True)
