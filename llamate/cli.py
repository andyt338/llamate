import argparse
from llamate.agent import MemoryAgent

def main():
    parser = argparse.ArgumentParser(description="LLAMate - Memory agent CLI")
    parser.add_argument("--user", type=str, required=True, help="User ID")
    args = parser.parse_args()

    agent = MemoryAgent(user_id=args.user)
    print("\nLLAMate CLI: Type 'exit' to quit\n")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = agent.chat(prompt)
        print("LLAMate:", response)

if __name__ == "__main__":
    main()
