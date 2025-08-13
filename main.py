import uuid
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableWithMessageHistory, RunnableLambda
from langchain_openai import ChatOpenAI

from memory import get_history
from tools import get_weather, calculator

load_dotenv()

tools = [get_weather, calculator]

llm: ChatOpenAI = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
llm_with_tools: Runnable = llm.bind_tools(tools)

system_message = SystemMessage(
    content='You are a helpful ai assistant that can answer questions and perform tasks using tools.'
            'You speak british english and you are very polite.'
            'Most of the time you will be used to perform small tasks'
            'Your developer is NoOneHardy and most of the time it will be him who will speak to you.'
)


def call_tools(msg: AIMessage) -> List[ToolMessage]:
    tool_outputs = []

    if getattr(msg, 'tool_calls', None):
        # Iterate over the tool calls and invoke each tool with the provided arguments
        for call in msg.tool_calls:
            name = call['name']
            args = call['args']

            result = next(t for t in tools if t.name == name).invoke(args)
            tool_outputs.append(ToolMessage(content=result, tool_call_id=call['id']))

    return tool_outputs


def call_llm(input_messages: List[HumanMessage]) -> AIMessage:
    # Invoke the LLM with the user input and system message
    # The LLM will decide if it needs to call any tools
    first = llm_with_tools.invoke([*input_messages, system_message])

    tool_outputs = call_tools(first)

    if len(tool_outputs) > 0:
        # Create a formatted response using the LLM in combination with the tool outputs, the original user input and the system message
        final = llm_with_tools.invoke([*input_messages, first, *tool_outputs, system_message])
        return final

    return first


agent_core = RunnableLambda(call_llm)
agent_with_memory = RunnableWithMessageHistory(
    agent_core,
    get_history
)


def chat(user_input: str, session_id: str) -> str:
    result = agent_with_memory.invoke(
        [HumanMessage(content=user_input)],
        config={'configurable': {'session_id': session_id}}
    )
    return result.content


def main():
    session_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input('> ')
            if user_input.lower() in ['exit', 'quit']:
                break
            print(chat(user_input, session_id))
        except KeyboardInterrupt:
            break

    print('Bye...')


if __name__ == '__main__':
    main()
