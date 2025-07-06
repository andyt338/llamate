#!/usr/bin/env python
"""
Simple chatbot example using Llamate for long-term memory.
"""
import os
import sys
from dotenv import load_dotenv

# Add the parent directory to sys.path for direct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamate import MemoryAgent

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the memory agent with a unique user ID
    user_id = "example_user"
    agent = MemoryAgent(user_id=user_id)
    
    print("🦙 Llamate Memory-Augmented Chatbot")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            print("\nGoodbye! 👋")
            break
        
        # Get a response from the agent with memory retrieval
        response = agent.chat(user_input)
        
        print(f"\nBot: {response}")

if __name__ == "__main__":
    main()
