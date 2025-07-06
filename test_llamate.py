from llamate import MemoryAgent, get_vectorstore_from_env
import os

# Set user ID
user_id = "test_user"

# Initialize components
vectorstore = get_vectorstore_from_env(user_id=user_id)
agent = MemoryAgent(user_id=user_id, vectorstore=vectorstore)

# Add memories
agent.chat("The capital of France is Paris.")
agent.chat("The Eiffel Tower is 324 meters tall.")
agent.chat("Python is a programming language created by Guido van Rossum.")

# Test retrieval
response = agent.chat("Tell me about Paris.")
print("Response:", response)
