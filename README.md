# 🦙 Llamate

Llamate is a memory-augmented agent framework for Large Language Models (LLMs) that provides persistent, retrievable memory for AI conversations.

## What is Llamate?

Llamate solves a fundamental limitation of current LLMs: their inability to remember past conversations beyond a single context window. It creates a vector database of memories that can be semantically searched and retrieved during conversations, allowing LLMs to maintain continuity and context over extended interactions.

## How It Works

1. **Memory Storage**: Llamate stores important pieces of conversation as vector embeddings in a database (PostgreSQL is the only supported DB).
2. **Semantic Retrieval**: When new queries come in, Llamate searches for semantically relevant past memories.
3. **Memory Filtering**: The system automatically filters out the current query from search results to prevent echo effects.
4. **Context Enhancement**: Retrieved memories are injected into the conversation context, allowing the LLM to access and utilize past information.
5. **User Identification**: Each user gets a unique memory space, ensuring personalized conversation history.

## Key Features

- **Backend Support**: Works with PostgreSQL (with pgvector)
- **Persistence**: Memories remain available between sessions and application restarts
- **Simple API**: Easy-to-use Python interface that works with any LLM
- **CLI Interface**: Command-line tool for quick testing and interaction
- **Production Ready**: Designed for both development and production environments


## Quick Start

#### 1. Install Package

```bash
pip install llamate
```

#### 2. OpenAI API Requirements

Llamate requires access to the following OpenAI models in your account:

- **Embedding models** (at least one of):
  - `text-embedding-3-small` (default, 1536 dimensions) - Faster, smaller embeddings, cost-effective
  - `text-embedding-3-large` (3072 dimensions) - Higher accuracy, larger embeddings
- `gpt-4` - Recommended for high-quality responses

Make sure these models are enabled in your OpenAI account.

#### 3. Environment Variables

Llamate is configured primarily through environment variables, making it easy to integrate with any backend deployment. The following environment variables are supported:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMATE_OPENAI_API_KEY` | None (Required) | Your OpenAI API key |
| `LLAMATE_DATABASE_URL` | None (Required) | PostgreSQL connection string (when using postgres backend) |
| `LLAMATE_VECTOR_BACKEND` | `postgres` (Required) | Vector store backend (`postgres`) |
| `LLAMATE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model to use (`text-embedding-3-small` or `text-embedding-3-large`) |

Example configuration for production deployment:

```bash
# Required
LLAMATE_OPENAI_API_KEY=sk-your-api-key
LLAMATE_DATABASE_URL=postgresql://user:password@your-db-host:5432/dbname
LLAMATE_VECTOR_BACKEND=postgres

# Optional overrides
LLAMATE_EMBEDDING_MODEL=text-embedding-3-large
```

#### 4. Example Integration

```python
from llamate import MemoryAgent, get_vectorstore_from_env
import os

# In production, set environment variables directly in your deployment platform
# os.environ["LLAMATE_OPENAI_API_KEY"] = "your-key-here" # Set in platform instead
# os.environ["LLAMATE_DATABASE_URL"] = "connection-string" # Set in platform instead

def create_llamate_agent(user_id):
    """Factory function to create a memory-augmented agent for a specific user"""
    vectorstore = get_vectorstore_from_env(user_id=user_id)
    return MemoryAgent(user_id=user_id, vectorstore=vectorstore)

# Example API endpoint
def handle_chat_request(user_id, user_message):
    agent = create_llamate_agent(user_id)
    return agent.chat(user_message)
```

## Local Development

The following steps guide you through setting up Llamate for local development:

1. Create a local Docker container
```bash
docker run --name llamate-postgres -e POSTGRES_USER=llamate -e POSTGRES_PASSWORD=llamate -e POSTGRES_DB=llamate -p 5432:5432 -d ankane/pgvector
```

2. In a separate terminal, initialize llamate

```bash
llamate --init
# Select 'postgres' as your vector store backend
# Enter connection string: postgresql://llamate:llamate@localhost:5432/llamate
```

> **Note:** While you can use `llamate --init` for local development to generate a `.env` file, in production environments you should configure these variables directly in your deployment platform.

3. Now test the package in a python terminal or script
```python
from llamate import MemoryAgent, get_vectorstore_from_env

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
```

To view the data in the local PostgreSQL container, connect to the database:

```bash
docker exec -it llamate-postgres psql -U llamate -d llamate
```

List tables to find your memory table (it will use your user_id):

```sql
\dt
```

View table structure:

```sql
\d memory_test_user
```

Display memory records (omitting the large vector field):

```sql
SELECT id, text FROM memory_test_user;
```

Count records:

```sql
SELECT COUNT(*) FROM memory_test_user;
```

Query specific memories (using text search):

```sql
SELECT id, text FROM memory_test_user WHERE text LIKE '%Paris%';
```

Delete test memories (if needed):

```sql
DELETE FROM memory_test_user WHERE text LIKE '%test%';
```

Exit the PostgreSQL shell:

```sql
\q
```

## How to create Postgres DB in AWS:

First, create an EC2 instance in AWS.

* OS: Ubuntu
* Type: t3.micro, 30 GB general purpose SSD (free tier limit)
* Create new keypair, store it securely somewhere
* Create new VPC and subnet if you need to. Enable public IPs in the subnet if it asks.
* Create new security group, allow port 22 from your IP address, port 5432 from 0.0.0.0/0
* Select your new security group in the dropdown
* Launch instance

1. SSH into instance:

```
chmod 400 ~/Downloads/my-keypair.pem
ssh -i ~/Downloads/my-keypair.pem ubuntu@44.203.101.127
```

Local .pem file name and public IP of the VM will be different. Username will be ubuntu.

2. Install Postgres on the VM:

```
sudo apt update
sudo apt install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
sudo systemctl status postgresql

sudo -i -u postgres
psql
```

3. Create DB, user, and vector extension
```
CREATE DATABASE mydb;
CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';

-- Connect to the newly created database
\c mydb

-- Create the pgvector extension in this specific database
CREATE EXTENSION vector;

-- Grant database privileges to the myuser user
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;

-- Grant schema privileges to the myuser user
GRANT ALL PRIVILEGES ON SCHEMA public TO myuser;

-- If tables already exist, grant privileges on those too
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO myuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO myuser;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO myuser;

-- Allow the user to create new tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO myuser;

-- Verify extension is installed
\dx

\q

exit  # to return to VM
```

4. Configure Postgres service

```
sudo nano /etc/postgresql/16/main/postgresql.conf
# set:
# listen_addresses = '*'

sudo nano /etc/postgresql/16/main/pg_hba.conf
# add:
# host    all             all             0.0.0.0/0               md5

sudo systemctl restart postgresql
sudo systemctl status postgresql
```
