import chainlit as cl
from main import RAGChatbot

# Initialize the RAGChatbot
chatbot = RAGChatbot()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Welcome, I'm here to listen and support you. How can I help you today?").send()

@cl.on_message
async def on_message(message: cl.Message):
    response = chatbot.chat(query=message.content)
    response = response["result"]
    await cl.Message(content=response).send()
