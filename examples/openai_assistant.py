#!/usr/bin/env python
"""
Advanced example integrating Llamate's memory capabilities with OpenAI's ChatCompletion API.
This example shows how to use retrieved memories as context for the OpenAI model.
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Add the parent directory to sys.path for direct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamate import MemoryAgent
from llamate.vectorstore_postgres import PostgresVectorStore
from llamate.embedder import OpenAIEmbedder

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("LLAMATE_OPENAI_API_KEY"))
    
    # Initialize Llamate components
    user_id = "advanced_user"
    embedder = OpenAIEmbedder()
    vectorstore = PostgresVectorStore(user_id=user_id, embedder=embedder)
    
    # Create memory agent
    agent = MemoryAgent(
        user_id=user_id,
        vectorstore=vectorstore,
        embedder=embedder
    )
    
    # System prompt that instructs the model how to use retrieved memories
    system_prompt = """
    You are a helpful assistant with access to the user's previous conversations.
    When responding to the user, incorporate relevant information from their memory
    if it's available. If you use remembered information, subtly acknowledge this
    by saying something like "as you mentioned before" or "based on our previous conversation".
    """
    
    # Message history for OpenAI API
    messages = [{"role": "system", "content": system_prompt}]
    
    print("🦙✨ Llamate OpenAI Memory-Augmented Assistant")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            print("\nGoodbye! 👋")
            break
        
        # Store the user message in memory
        vectorstore.add(user_input, embedder)
        
        # Retrieve relevant memories
        memories = vectorstore.search(user_input, top_k=3)
        
        # If we have relevant memories, add them as context
        context = ""
        if memories:
            context = "Here are some relevant memories from previous conversations:\n"
            context += "\n".join([f"- {memory}" for memory in memories])
            context += "\n\nUse this context to inform your response if relevant."
        
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # If we have context from memory, add it as a system message
        if context:
            messages.append({"role": "system", "content": context})
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo",  # Or any other available model
            messages=messages,
            max_tokens=500
        )
        
        assistant_message = response.choices[0].message.content
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": assistant_message})
        
        # Also store the assistant's response in memory
        vectorstore.add(assistant_message, embedder)
        
        print(f"\nAssistant: {assistant_message}")
        
        # Keep the conversation history manageable by retaining only the last few exchanges
        if len(messages) > 10:
            # Keep system prompt and last 3 exchanges (6 messages)
            messages = [messages[0]] + messages[-6:]

if __name__ == "__main__":
    main()
