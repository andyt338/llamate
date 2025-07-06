from llamate.agent import MemoryAgent


def test_chat_response():
    agent = MemoryAgent(user_id="test_user")
    response = agent.chat("hello")
    assert isinstance(response, str)
    assert len(response) > 0
