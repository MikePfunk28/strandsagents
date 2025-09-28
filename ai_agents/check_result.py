from strands import Agent
import asyncio

async def check_result():
    agent = Agent()
    result = await agent.invoke_async('hello')
    print(f"Type: {type(result)}")
    print(f"Dir: {[attr for attr in dir(result) if not attr.startswith('_')]}")
    print(f"Result: {result}")

    # Try to access content
    if hasattr(result, 'content'):
        print(f"Content: {result.content}")
    if hasattr(result, 'text'):
        print(f"Text: {result.text}")
    if hasattr(result, 'message'):
        print(f"Message: {result.message}")

asyncio.run(check_result())