from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langchain_core.runnables import Runnable

load_dotenv()

from langchain_openai import ChatOpenAI

from tools import get_weather

tools = [get_weather]

llm: ChatOpenAI = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
llm_with_tools: Runnable = llm.bind_tools(tools)

system_message = SystemMessage(
    content='You are a helpful assistant that can answer questions and perform tasks using tools. Always start your answer with "Aye Captain!" and then provide the answer or the result of the tool usage.'
)

def run_once(user_input: str) -> str:
    # Invoke the LLM with the user input and system message
    # The LLM will decide if it needs to call any tools
    first = llm_with_tools.invoke([HumanMessage(content=user_input), system_message])

    # Evaluate if the LLM response contains tool calls
    if getattr(first, 'tool_calls', None):
        tool_outputs = []

        # Iterate over the tool calls and invoke each tool with the provided arguments
        for call in first.tool_calls:
            name = call['name']
            args = call['args']

            result = next(t for t in tools if t.name == name).invoke(args)
            tool_outputs.append(ToolMessage(content=result, tool_call_id=call['id']))

        # Create a formatted response using the LLM in combination with the tool outputs, the original user input and the system message
        final = llm_with_tools.invoke([HumanMessage(content=user_input), first, *tool_outputs, system_message])
        return final.content

    # Return the LLM's response directly if no tool calls were made
    return first.content

def main():
    while True:
        try:
            user_input = input('> ')
            if user_input.lower() in ['exit', 'quit']:
                break
            print(run_once(user_input))
        except KeyboardInterrupt:
            break
    
    print('Bye...')


if __name__ == '__main__':
    main()
