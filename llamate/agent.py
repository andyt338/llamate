from llamate.embedder import OpenAIEmbedder
from llamate.vectorstore import FAISSVectorStore

class MemoryAgent:
    def __init__(self, user_id, embedder=None, vectorstore=None, model="gpt-4"):
        self.user_id = user_id
        self.model = model
        self.embedder = embedder or OpenAIEmbedder()
        self.vectorstore = vectorstore or FAISSVectorStore(user_id)

    def chat(self, user_input: str) -> str:
        memories = self.vectorstore.search(user_input)
        memory_context = "\n".join(memories)
        prompt = f"Memory:\n{memory_context}\n\nUser: {user_input}"

        response = self.embedder.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant with long-term memory."},
                {"role": "user", "content": prompt}
            ]
        )

        assistant_reply = response.choices[0].message.content
        self.vectorstore.add(f"User: {user_input}", self.embedder)
        self.vectorstore.add(f"Assistant: {assistant_reply}", self.embedder)
        return assistant_reply
