# LangChain Expression Language (LCEL) Guide

## Table of Contents
- [LangChain Expression Language (LCEL) Guide](#langchain-expression-language-lcel-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Key Features](#key-features)
    - [1. First-class Streaming Support](#1-first-class-streaming-support)
    - [2. Async Support](#2-async-support)
    - [3. Optimized Parallel Execution](#3-optimized-parallel-execution)
    - [4. Retries and Fallbacks](#4-retries-and-fallbacks)
    - [5. Access to Intermediate Results](#5-access-to-intermediate-results)
    - [6. Input and Output Schemas](#6-input-and-output-schemas)
    - [7. Seamless LangSmith Tracing](#7-seamless-langsmith-tracing)
  - [Getting Started](#getting-started)
  - [Advanced LCEL Patterns](#advanced-lcel-patterns)
    - [Map-Reduce](#map-reduce)
    - [Agents](#agents)
  - [Error Handling](#error-handling)
  - [Integration with Other LangChain Components](#integration-with-other-langchain-components)
  - [Performance Optimization](#performance-optimization)
  - [Testing LCEL Chains](#testing-lcel-chains)
  - [Use Cases and Real-World Examples](#use-cases-and-real-world-examples)
    - [1. Question Answering System](#1-question-answering-system)
    - [2. Multi-step Data Analysis Pipeline](#2-multi-step-data-analysis-pipeline)
  - [LCEL Best Practices](#lcel-best-practices)
  - [Customization](#customization)
  - [Versioning and Reproducibility](#versioning-and-reproducibility)
  - [Glossary](#glossary)

## Introduction

LangChain Expression Language (LCEL) is a declarative way to chain LangChain components. It's designed to support putting prototypes into production without code changes, from simple "prompt + LLM" chains to complex chains with hundreds of steps.

## Key Features

### 1. First-class Streaming Support

LCEL provides the best possible time-to-first-token for chains, allowing for streaming of parsed, incremental chunks of output at the same rate as the LLM provider outputs raw tokens.

**Example:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatOpenAI()
output_parser = StrOutputParser()

chain = prompt | model | output_parser

for chunk in chain.stream({"topic": "programming"}):
    print(chunk, end="", flush=True)
```

### 2. Async Support

Any chain built with LCEL can be called with both synchronous and asynchronous APIs, enabling the same code to be used for prototypes and production with great performance and the ability to handle many concurrent requests.

**Example:**

```python
import asyncio

async def main():
    async for chunk in chain.astream({"topic": "programming"}):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### 3. Optimized Parallel Execution

LCEL automatically executes steps that can be run in parallel for the smallest possible latency, both in sync and async interfaces.

**Example:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableParallel

model = ChatOpenAI()

chain = RunnableParallel(
    joke=ChatPromptTemplate.from_template("tell me a joke about {topic}") | model,
    fact=ChatPromptTemplate.from_template("tell me a fact about {topic}") | model
)

result = chain.invoke({"topic": "programming"})
print(result["joke"])
print(result["fact"])
```

### 4. Retries and Fallbacks

You can configure retries and fallbacks for any part of your LCEL chain, improving reliability at scale.

**Example:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableWithFallbacks

model = ChatOpenAI()
fallback_model = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | RunnableWithFallbacks(model, fallbacks=[fallback_model])

result = chain.invoke({"topic": "programming"})
print(result)
```

### 5. Access to Intermediate Results

LCEL allows access to results of intermediate steps before the final output is produced, which is useful for debugging and enhancing user experience.

**Example:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence

model = ChatOpenAI()

chain = RunnableSequence(
    ChatPromptTemplate.from_template("tell me a joke about {topic}"),
    model,
    lambda x: f"The model's response was: {x.content}"
)

for event in chain.stream_events({"topic": "programming"}):
    if event.event == "on_chat_model_start":
        print("Model started thinking...")
    elif event.event == "on_chat_model_end":
        print("Model finished!")
    elif event.event == "on_chain_end":
        print("Final output:", event.data["output"])
```

### 6. Input and Output Schemas

LCEL chains have Pydantic and JSONSchema schemas inferred from the chain structure, which can be used for validation of inputs and outputs and is an integral part of LangServe.

**Example:**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field

class JokeOutput(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic} with a setup and punchline.")

chain = prompt | model | JokeOutput.from_response()

result = chain.invoke({"topic": "programming"})
print(f"Setup: {result.setup}")
print(f"Punchline: {result.punchline}")
```

### 7. Seamless LangSmith Tracing

All steps in LCEL chains are automatically logged to LangSmith for maximum observability and debuggability.

## Getting Started

To get started with LCEL, follow these steps:

1. Install LangChain:
   ```
   pip install langchain
   ```

2. Set up your environment variables for your LLM provider (e.g., OpenAI):
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```

3. Create a simple chain:
   ```python
   from langchain.chat_models import ChatOpenAI
   from langchain.prompts import ChatPromptTemplate
   from langchain.schema.output_parser import StrOutputParser

   prompt = ChatPromptTemplate.from_template("tell me a {adjective} joke about {topic}")
   model = ChatOpenAI()
   output_parser = StrOutputParser()

   chain = prompt | model | output_parser

   result = chain.invoke({"adjective": "funny", "topic": "programming"})
   print(result)
   ```

4. Experiment with different components and features of LCEL to build more complex chains and applications.

## Advanced LCEL Patterns

### Map-Reduce

Map-Reduce is a powerful pattern for processing large datasets. Here's an example of how to implement it using LCEL:

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

# Define the map function
map_prompt = ChatPromptTemplate.from_template("Summarize this text: {text}")
map_chain = map_prompt | ChatOpenAI(temperature=0)

# Define the reduce function
reduce_prompt = ChatPromptTemplate.from_template("Combine these summaries: {summaries}")
reduce_chain = reduce_prompt | ChatOpenAI(temperature=0)

# Create the map-reduce chain
map_reduce_chain = (
    RunnableMap({
        "summaries": RunnablePassthrough.assign(text=lambda x: x["texts"]) | map_chain.map(),
    })
    | reduce_chain
)

# Use the chain
texts = ["Text 1", "Text 2","Text 3"]
result = map_reduce_chain.invoke({"texts": texts})
print(result)
```

### Agents

LCEL can be used to build flexible agents. Here's a simple example:

```python
from langchain.agents import AgentExecutor, Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Use the following tools to answer the user's question: {tool_names}"),
    ("human", "{input}")
])

llm = ChatOpenAI(temperature=0)

agent = (
    {
        "input": lambda x: x["input"],
        "tool_names": lambda x: ", ".join([tool.name for tool in x["tools"]])
    }
    | prompt
    | llm
    | AgentExecutor(tools=tools)
)

result = agent.invoke({
    "input": "What's the latest news about artificial intelligence?",
    "tools": tools
})
print(result)
```

## Error Handling

LCEL provides robust error handling capabilities. Here's an example of how to implement retries and fallbacks:

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableWithFallbacks

# Define primary and fallback models
primary_model = ChatOpenAI(temperature=0)
fallback_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Create a prompt
prompt = ChatPromptTemplate.from_template("Explain {concept} in simple terms.")

# Create a chain with fallback
chain_with_fallback = RunnableWithFallbacks(
    prompt | primary_model,
    fallbacks=[prompt | fallback_model]
)

# Add retry logic
chain_with_retry = chain_with_fallback.with_retry(
    max_attempts=3,
    wait_exponential_jitter=True
)

# Use the chain
try:
    result = chain_with_retry.invoke({"concept": "quantum computing"})
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")
```

## Integration with Other LangChain Components

LCEL can be easily integrated with other LangChain components like retrievers, memory, and tools. Here's an example using a retriever and memory:

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Create a vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(["LangChain is awesome", "LCEL is powerful"], embeddings)
retriever = vectorstore.as_retriever()

# Create a memory
memory = ConversationBufferMemory(return_messages=True)

# Create the chain
template = """Answer the question based on the following context and chat history:
Context: {context}
Chat History: {chat_history}
Human: {question}
AI: """

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

chain = (
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "chat_history": lambda x: memory.load_memory_variables({})["history"],
        "question": lambda x: x["question"]
    }
    | prompt
    | model
)

# Use the chain
result = chain.invoke({"question": "What is LangChain?"})
print(result)

# Update memory
memory.save_context({"input": "What is LangChain?"}, {"output": result.content})

# Ask another question
result = chain.invoke({"question": "What is LCEL?"})
print(result)
```

## Performance Optimization

To optimize LCEL chains for better performance, consider the following tips:

1. Use Async Operations: Leverage async methods for I/O-bound operations.

```python
import asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

async def generate_responses(questions):
    model = ChatOpenAI()
    prompt = ChatPromptTemplate.from_template("{question}")
    chain = prompt | model

    async_results = await asyncio.gather(*[chain.ainvoke({"question": q}) for q in questions])
    return async_results

questions = ["What is AI?", "What is machine learning?", "What is deep learning?"]
results = asyncio.run(generate_responses(questions))
for result in results:
    print(result)
```

2. Implement Caching: Use LangChain's built-in caching to avoid redundant computations.

```python
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache

set_llm_cache(InMemoryCache())

# Now, repeated calls with the same input will use cached results
```

3. Batch Processing: Use the `batch` method for processing multiple inputs efficiently.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("Tell me a fact about {topic}")
chain = prompt | model

topics = ["AI", "machine learning", "deep learning"]
results = chain.batch([{"topic": t} for t in topics])
for result in results:
    print(result)
```

## Testing LCEL Chains

Testing LCEL chains is crucial for ensuring reliability and consistency. Here are some best practices:

1. Unit Testing: Test individual components of your chain.

```python
import unittest
from langchain.prompts import ChatPromptTemplate

class TestPrompt(unittest.TestCase):
    def test_prompt_template(self):
        prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
        result = prompt.format(topic="AI")
        self.assertEqual(result, "Tell me about AI")

if __name__ == '__main__':
    unittest.main()
```

2. Integration Testing: Test the entire chain end-to-end.

```python
import unittest
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class TestChain(unittest.TestCase):
    def test_chain_output(self):
        prompt = ChatPromptTemplate.from_template("Tell me a fact about {topic}")
        model = ChatOpenAI()
        chain = prompt | model
        result = chain.invoke({"topic": "AI"})
        self.assertIsNotNone(result.content)
        self.assertIn("AI", result.content)

if __name__ == '__main__':
    unittest.main()
```

3. Mock Testing: Use mocks to test chains without calling actual APIs.

```python
import unittest
from unittest.mock import Mock
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class TestChainWithMock(unittest.TestCase):
    def test_chain_with_mock(self):
        mock_model = Mock(spec=ChatOpenAI)
        mock_model.return_value.invoke.return_value = "Mocked response about AI"

        prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
        chain = prompt | mock_model

        result = chain.invoke({"topic": "AI"})
        self.assertEqual(result, "Mocked response about AI")

if __name__ == '__main
    unittest.main()
```

## Use Cases and Real-World Examples

LCEL can be applied to a wide range of use cases. Here are a few real-world examples:

### 1. Question Answering System

This example demonstrates a question answering system that uses a retriever to fetch relevant information and then answers questions based on that context.

```python
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.schema.runnable import RunnablePassthrough

# Create a vector store with some sample data
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    ["LangChain is a framework for developing applications powered by language models.",
     "LCEL allows for easy composition of LangChain components.",
     "Vector stores are used for semantic search in LangChain."],
    embeddings
)
retriever = vectorstore.as_retriever()

# Create the chain
template = """Answer the question based on the following context:
Context: {context}
Question: {question}
Answer: """

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()

chain = (
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": lambda x: x["question"]
    }
    | prompt
    | model
)

# Use the chain
result = chain.invoke({"question": "What is LangChain used for?"})
print(result.content)
```

### 2. Multi-step Data Analysis Pipeline

This example shows how to create a multi-step data analysis pipeline using LCEL.

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Sample data
data = {
    "sales": [100, 200, 150, 300, 250],
    "expenses": [80, 120, 100, 150, 130]
}

# Step 1: Calculate total sales and expenses
def calculate_totals(data):
    return {
        "total_sales": sum(data["sales"]),
        "total_expenses": sum(data["expenses"])
    }

# Step 2: Calculate profit
def calculate_profit(data):
    return {
        "profit": data["total_sales"] - data["total_expenses"]
    }

# Step 3: Generate report
report_template = """Based on the following financial data:
Total Sales: ${total_sales}
Total Expenses: ${total_expenses}
Profit: ${profit}

Provide a brief financial analysis and recommendations for the business."""

report_prompt = ChatPromptTemplate.from_template(report_template)
model = ChatOpenAI()

# Create the multi-step chain
analysis_chain = (
    RunnablePassthrough.assign(totals=calculate_totals)
    | RunnablePassthrough.assign(profit=calculate_profit)
    | report_prompt
    | model
)

# Run the analysis
result = analysis_chain.invoke(data)
print(result.content)
```

## LCEL Best Practices

When working with LCEL, consider the following best practices:

1. **Modular Design**: Break down complex chains into smaller, reusable components. This makes your code more maintainable and easier to test.

2. **Use Type Hints**: Leverage Python's type hinting system to make your code more readable and catch potential errors early.

```python
from typing import Dict, List
from langchain.schema.runnable import RunnablePassthrough

def process_data(data: Dict[str, List[int]]) -> Dict[str, int]:
    return {
        "total": sum(data["values"])
    }

chain = RunnablePassthrough.assign(processed=process_data) | ...
```

3. **Error Handling**: Implement proper error handling and fallback mechanisms to make your chains more robust.

4. **Streaming for Large Outputs**: Use streaming for chains that may produce large outputs to improve responsiveness.

5. **Leverage Async**: Use async methods when dealing with I/O-bound operations to improve performance.

6. **Monitor and Log**: Use LangSmith or custom logging to monitor your chains' performance and debug issues.

7. **Version Control**: Keep track of your chain versions, especially when deploying to production.

## Customization

LCEL allows for extensive customization through custom Runnable components. Here's an example of creating a custom Runnable:

```python
from langchain.schema.runnable import Runnable
from typing import Any, Dict

class CustomProcessor(Runnable):
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        # Custom processing logic here
        return {"processed_" + k: v.upper() if isinstance(v, str) else v for k, v in input.items()}

# Use the custom processor in a chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

custom_processor = CustomProcessor()
prompt = ChatPromptTemplate.from_template("Processed input: {processed_input}")
model = ChatOpenAI()

chain = custom_processor | prompt | model

result = chain.invoke({"input": "Hello, World!"})
print(result.content)
```

## Versioning and Reproducibility

Versioning and ensuring reproducibility are crucial for maintaining and scaling LCEL applications. Here are some strategies:

1. **Use LangSmith**: LangSmith provides versioning and experiment tracking for LangChain applications.

2. **Git Version Control**: Use Git to version control your LCEL code and configurations.

3. **Environment Management**: Use tools like `poetry` or `conda` to manage your project's dependencies and environment.

4. **Configuration Files**: Store chain configurations in separate files (e.g., YAML) and version control them.

```python
import yaml
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create chain based on configuration
model = ChatOpenAI(**config["model"])
prompt = ChatPromptTemplate.from_template(config["prompt_template"])
chain = prompt | model

# Use the chain
result = chain.invoke({"input": config["test_input"]})
print(result.content)
```

5. **Seed Random Number Generators**: If your chains involve any randomness, set and record random seeds for reproducibility.

## Glossary

- **LCEL**: LangChain Expression Language, a declarative way to chain LangChain components.
- **Runnable**: The base interface for LCEL components, defining methods like `invoke`, `stream`, and `batch`.
- **RunnableSequence**: A series of Runnables executed in order.
- **RunnableParallel**: A set of Runnables executed in parallel.
- **RunnablePassthrough**: A Runnable that passes its input through unchanged, often used with `assign` for adding new keys.
- **RunnableWithFallbacks**: A Runnable with defined fallback options in case of failure.
- **LangSmith**: A platform for debugging, testing, evaluating, and monitoring LLM applications.
- **Retriever**: A component that fetches relevant documents based on a query.
- **Vector Store**: A database optimized for storing and querying vector embeddings.
- **Embedding**: A numerical representation of text that captures semantic meaning.
- **Prompt Template**: A template for generating prompts with variable inputs.
- **Output Parser**: A component that structures the raw output from an LLM into a desired format.

This comprehensive guide should provide a solid foundation for understanding and
working with LCEL. As you build more complex applications, refer back to this
guide and the official LangChain documentation for more detailed information on
specific components and advanced usage patterns.
