"""
Memory Assistant - A complete memory management system using Strands Agents

This class provides comprehensive memory operations including storage, retrieval,
search, and answer generation based on stored memories.
"""

from typing import Dict, List, Any, Optional
from strands import Agent
from strands_tools import mem0_memory, use_llm


class MemoryAssistant:
    """
    A memory assistant that helps store and retrieve relevant information.

    Features:
    - Store memories with automatic relevance scoring
    - Retrieve memories based on semantic similarity
    - Generate contextual answers from retrieved memories
    - List and manage stored memories
    - Search through memory collections
    """

    MEMORY_SYSTEM_PROMPT = """You are a memory assistant that helps store and retrieve relevant information.
    You can store new memories, retrieve relevant memories based on queries, and help manage memory collections.
    Always provide accurate and helpful responses based on the stored information."""

    ANSWER_SYSTEM_PROMPT = """You are an assistant that creates helpful responses based on retrieved memories.
    Use the provided memories to create a natural, conversational response to the user's question.
    If memories are provided, base your answer on that information. If no relevant memories are found,
    acknowledge this and provide a general helpful response."""

    def __init__(self, user_id: str = "demo_user"):
        """
        Initialize the Memory Assistant.

        Args:
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.agent = Agent(
            system_prompt=self.MEMORY_SYSTEM_PROMPT,
            tools=[mem0_memory, use_llm],
        )

    def store_memory(self, content: str) -> Dict[str, Any]:
        """
        Store a new memory.

        Args:
            content: The memory content to store

        Returns:
            Dictionary containing the storage result
        """
        try:
            result = self.agent.tool.mem0_memory(
                action="store",
                content=content,
                user_id=self.user_id
            )
            return result
        except Exception as e:
            return {"error": f"Failed to store memory: {str(e)}"}

    def retrieve_memories(self, query: str, min_score: float = 0.3, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories based on a query.

        Args:
            query: Search query to find relevant memories
            min_score: Minimum relevance score (0.0 to 1.0)
            max_results: Maximum number of results to return

        Returns:
            List of relevant memories with their metadata
        """
        try:
            result = self.agent.tool.mem0_memory(
                action="retrieve",
                query=query,
                min_score=min_score,
                max_results=max_results
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    def list_all_memories(self) -> List[Dict[str, Any]]:
        """
        List all stored memories for the user.

        Returns:
            List of all memories with their metadata
        """
        try:
            result = self.agent.tool.mem0_memory(
                action="list",
                user_id=self.user_id
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            print(f"Error listing memories: {e}")
            return []

    def search_memories(self, query: str) -> List[Dict[str, Any]]:
        """
        Search memories with a custom query.

        Args:
            query: Search query

        Returns:
            List of matching memories
        """
        try:
            result = self.agent.tool.mem0_memory(
                action="search",
                query=query,
                user_id=self.user_id
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            print(f"Error searching memories: {e}")
            return []

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """
        Delete a specific memory.

        Args:
            memory_id: ID of the memory to delete

        Returns:
            Dictionary containing the deletion result
        """
        try:
            result = self.agent.tool.mem0_memory(
                action="delete",
                memory_id=memory_id,
                user_id=self.user_id
            )
            return result
        except Exception as e:
            return {"error": f"Failed to delete memory: {str(e)}"}

    def generate_answer_from_memories(self, query: str, memories: List[Dict[str, Any]]) -> str:
        """
        Generate an answer based on retrieved memories.

        Args:
            query: The user's question
            memories: List of relevant memories

        Returns:
            Generated answer based on the memories
        """
        if not memories:
            return "I don't have any relevant memories for that question. Could you provide more context or ask me to store some information first?"

        try:
            # Format memories for the prompt
            memories_text = ""
            for i, memory in enumerate(memories, 1):
                content = memory.get('content', 'No content')
                score = memory.get('score', 'N/A')
                memories_text += f"{i}. {content} (relevance: {score})\n"

            # Create a comprehensive prompt
            prompt = f"""Based on the following memories:

{memories_text}

Please provide a helpful, accurate response to this question: {query}

If the memories are relevant, use them to inform your answer. If they're not directly relevant, acknowledge this and provide a general helpful response."""

            # Use the use_llm tool to generate the response
            response = self.agent.tool.use_llm(prompt=prompt)

            return str(response)

        except Exception as e:
            return f"I encountered an error while generating a response: {str(e)}"

    def process_input(self, user_input: str) -> str:
        """
        Main processing method that handles user input intelligently.

        Args:
            user_input: The user's message or question

        Returns:
            Appropriate response based on the input
        """
        # Check if user wants to store information
        store_keywords = ['remember', 'store', 'save', 'keep', 'note']
        if any(keyword in user_input.lower() for keyword in store_keywords):
            # Extract content to store (remove the command words)
            content = user_input
            for keyword in store_keywords:
                if keyword in content.lower():
                    content = content.lower().replace(keyword, '').strip()
                    break

            if content:
                result = self.store_memory(content)
                if 'error' not in result:
                    return f"✅ I've stored that information in your memory: '{content}'"
                else:
                    return f"❌ Failed to store memory: {result.get('error', 'Unknown error')}"
            else:
                return "What would you like me to remember?"

        # Check if user wants to list memories
        list_keywords = ['list', 'show', 'all', 'everything']
        if any(keyword in user_input.lower() for keyword in list_keywords):
            memories = self.list_all_memories()
            if memories:
                response = f"I have {len(memories)} memories stored:\n\n"
                for i, memory in enumerate(memories[:10], 1):  # Show first 10
                    content = memory.get('content', 'No content')[:100]  # Truncate long content
                    response += f"{i}. {content}...\n"
                if len(memories) > 10:
                    response += f"\n... and {len(memories) - 10} more memories."
                return response
            else:
                return "You don't have any memories stored yet."

        # Check if user wants to search memories
        search_keywords = ['search', 'find', 'look for']
        if any(keyword in user_input.lower() for keyword in search_keywords):
            # Extract search query
            query = user_input
            for keyword in search_keywords:
                if keyword in query.lower():
                    query = query.lower().replace(keyword, '').strip()
                    break

            memories = self.search_memories(query)
            if memories:
                return self.generate_answer_from_memories(query, memories)
            else:
                return f"No memories found for: '{query}'"

        # Default: retrieve relevant memories and generate answer
        memories = self.retrieve_memories(user_input)
        return self.generate_answer_from_memories(user_input, memories)

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored memories.

        Returns:
            Dictionary with memory statistics
        """
        try:
            memories = self.list_all_memories()
            return {
                "total_memories": len(memories),
                "user_id": self.user_id,
                "oldest_memory": memories[0] if memories else None,
                "newest_memory": memories[-1] if memories else None
            }
        except Exception as e:
            return {"error": f"Failed to get stats: {str(e)}"}

    def clear_all_memories(self) -> Dict[str, Any]:
        """
        Clear all memories for the user.

        Returns:
            Dictionary containing the result
        """
        try:
            memories = self.list_all_memories()
            deleted_count = 0

            for memory in memories:
                memory_id = memory.get('id')
                if memory_id:
                    self.delete_memory(memory_id)
                    deleted_count += 1

            return {"message": f"Successfully deleted {deleted_count} memories"}
        except Exception as e:
            return {"error": f"Failed to clear memories: {str(e)}"}


def main():
    """Example usage of the MemoryAssistant."""
    print("🧠 Memory Assistant Demo")
    print("=" * 50)

    # Create memory assistant
    assistant = MemoryAssistant(user_id="demo_user")

    # Example interactions
    print("\n1. Storing some memories...")
    assistant.store_memory("I like to play tennis on weekends")
    assistant.store_memory("My favorite programming language is Python")
    assistant.store_memory("I work as a software engineer")

    print("\n2. Asking a question...")
    response = assistant.process_input("What are my hobbies?")
    print(f"Response: {response}")

    print("\n3. Listing all memories...")
    response = assistant.process_input("list all memories")
    print(f"Response: {response}")

    print("\n4. Searching for specific information...")
    response = assistant.process_input("find programming")
    print(f"Response: {response}")

    print("\n5. Getting memory statistics...")
    stats = assistant.get_memory_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
