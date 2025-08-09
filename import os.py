import os
import re
import google.generativeai as genai

class GeminiAgent:
    """
    A simple agent that uses the Gemini API to choose between a set of tools
    to answer a user's question.
    """

    # The prompt template instructs the LLM on how to behave and what format to use.
    PROMPT_TEMPLATE = """
You are a helpful assistant that can use tools. You have access to the following tools:

1. search(query): Use this to find information about people, places, or facts.
2. calculator(expression): Use this to solve math problems.

To use a tool, respond with the exact format:
TOOL: tool_name(argument)

If you have the final answer, respond with:
ANSWER: your final answer

Here is the user's question:
{user_question}
"""

    def __init__(self):
        """
        Initializes the agent by configuring the Gemini API.
        """
        try:
            api_key = os.environ["GOOGLE_API_KEY"]
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini Agent initialized successfully.")
        except KeyError:
            raise RuntimeError(
                "Error: GOOGLE_API_KEY environment variable not set. "
                "Please get your key from Google AI Studio and set the environment variable."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {e}")

    def _search_tool(self, query: str) -> str:
        """A mock search tool. In a real app, this would call a real search API."""
        print(f"--- Calling Search Tool with query: '{query}' ---")
        query = query.lower()
        if "capital of france" in query:
            return "The capital of France is Paris."
        if "fastest land animal" in query:
            return "The fastest land animal is the cheetah."
        return f"I couldn't find information on '{query}'."

    def _calculator_tool(self, expression: str) -> str:
        """A calculator tool that safely evaluates a mathematical expression."""
        print(f"--- Calling Calculator Tool with expression: '{expression}' ---")
        try:
            # A simple, safer way to evaluate math expressions without using eval()
            allowed_chars = "0123456789+-*/(). "
            if not all(char in allowed_chars for char in expression):
                return "Sorry, the calculation contains invalid characters."
            result = eval(expression)
            return f"The result of the calculation is {result}."
        except Exception as e:
            return f"Sorry, I couldn't calculate that. Error: {e}"

    def _query_llm(self, prompt: str) -> str:
        """Sends a prompt to the Gemini API and gets the response."""
        print("--- Sending prompt to Gemini API ---")
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise ConnectionError(f"Gemini API Request failed: {e}")

    def run(self, user_question: str):
        """
        The main loop that runs the agent for a single user question.
        """
        print(f"\n🤔 AGENT'S GOAL: {user_question}\n")

        prompt = self.PROMPT_TEMPLATE.format(user_question=user_question)
        llm_response = self._query_llm(prompt)
        print(f"🤖 LLM THINKS: {llm_response.strip()}")

        # Use regex to see if the LLM wants to use a tool.
        tool_match = re.search(r"TOOL: (\w+)\((.*?)\)", llm_response, re.DOTALL)

        if tool_match:
            tool_name = tool_match.group(1).strip()
            tool_arg = tool_match.group(2).strip()

            if tool_name == "search":
                tool_result = self._search_tool(tool_arg)
            elif tool_name == "calculator":
                tool_result = self._calculator_tool(tool_arg)
            else:
                tool_result = f"Unknown tool: {tool_name}"
            
            print(f"\n✅ FINAL ANSWER: {tool_result}")

        elif "ANSWER:" in llm_response:
            final_answer = llm_response.split("ANSWER:", 1)[1].strip()
            print(f"\n✅ FINAL ANSWER: {final_answer}")
        else:
            # If the LLM didn't follow the format, just show its raw response.
            print(f"\n✅ FINAL ANSWER (Direct from LLM): {llm_response.strip()}")

def main():
    """Main function to create and run the agent."""
    try:
        agent = GeminiAgent()
        agent.run("What is the capital of France?")
        agent.run("What is (15 + 30) / 5?")
        agent.run("What is the fastest land animal?")
    except RuntimeError as e:
        print(e)

if __name__ == "__main__":
    main()