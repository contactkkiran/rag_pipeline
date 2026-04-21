# README for `agentic_rag_multi_tool.py`

## Overview

`agentic_rag_multi_tool.py` is a small agentic RAG demo that combines three behaviors in one script:

- `SEARCH` for answers from local PDF documents
- `WEB` for fresh or real-time information using DuckDuckGo
- `ANSWER` for general knowledge questions using the LLM directly

The script loads PDFs from the `docs/` folder, chunks them, creates embeddings with OpenAI, builds a FAISS vector store in memory, and then uses simple routing logic to decide which tool path to use for each user question.

## How It Works

For every question, the script chooses one of these paths:

1. `SEARCH`
   Uses the local PDF knowledge base and FAISS similarity search.

2. `WEB`
   Runs a DuckDuckGo search for recent or current information.

3. `ANSWER`
   Sends the question directly to the chat model for a normal response.

### Routing Rules

The script first applies rule-based routing:

- Questions containing words like `loan`, `service`, `policy`, `product`, or `offer` go to `SEARCH`
- Questions containing words like `latest`, `news`, `today`, `current`, or `weather` go to `WEB`

If no rule matches, the LLM acts as a fallback tool selector and returns one of:

- `ANSWER`
- `SEARCH`
- `WEB`

## Project Inputs

The script expects:

- a `docs/` folder containing PDF files
- an OpenAI API key in environment variables
- internet access for OpenAI API calls and DuckDuckGo web search

Current PDFs in this project:

- `docs/account_opening.pdf`
- `docs/bank_services.pdf`
- `docs/customer_support.pdf`
- `docs/digital_banking.pdf`
- `docs/loan_policies.pdf`

## Requirements

Recommended Python version:

- Python 3.10+

Install dependencies:

```bash
pip install python-dotenv langchain-community langchain-openai langchain-text-splitters pymupdf faiss-cpu duckduckgo-search
```

If you prefer, you can also install `langchain` itself:

```bash
pip install langchain
```

## Environment Setup

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

The script already calls `load_dotenv()`, so the key will be loaded automatically from `.env`.

## How to Run

From the project root:

```bash
python agentic_rag_multi_tool.py
```

You will get an interactive prompt like:

```text
Ask (type 'exit' to quit):
```

Example questions:

```text
What loan services are available?
What is the latest banking news today?
What is compound interest?
```

## Script Flow

At startup, the script:

1. loads all PDF pages from `docs/`
2. splits them into chunks using `CharacterTextSplitter`
3. creates embeddings with `OpenAIEmbeddings`
4. builds a FAISS vector store from the chunks
5. starts an interactive question loop

At query time:

1. it decides between `SEARCH`, `WEB`, or `ANSWER`
2. it executes the selected path
3. it sends the final prompt to `gpt-4o-mini`
4. it prints the response in the terminal

## Key Settings in the Script

Current chunking settings:

```python
chunk_size=500
chunk_overlap=50
```

Current chat model:

```python
ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

Current web search behavior:

- DuckDuckGo search
- top 3 results
- uses the result `body` text only

## What Each Tool Path Does

### `SEARCH`

- runs `vectorstore.similarity_search(query, k=5)`
- prints retrieved chunks in the terminal
- builds a prompt using only retrieved context
- instructs the model to say `"I don't know"` if the answer is not found

### `WEB`

- calls DuckDuckGo search
- collects brief snippets from top results
- asks the LLM to answer using that web data

### `ANSWER`

- directly asks the LLM to answer clearly
- does not use local documents or web search

## Limitations

- The FAISS index is rebuilt every time the script starts.
- The vector store is not loaded from the existing `faiss_index/` folder.
- Routing is keyword-based first, so some questions may be forced into the wrong path.
- Web search results are short snippets, not full page retrieval.
- Responses from the `WEB` path do not include source links.
- Document answers do not include file names or page citations.

## Suggested Improvements

Good next upgrades for this script:

- persist and reuse the FAISS index instead of rebuilding it each run
- add source citations with file name and page number
- move routing logic into a dedicated function
- use a better text splitter such as `RecursiveCharacterTextSplitter`
- add exception handling for missing API keys or network failures
- return structured outputs for tool selection instead of free-text parsing
- add a `requirements.txt`

## File Reference

Main script:

- [agentic_rag_multi_tool.py](/Users/kirankumar/Documents/rag_pipeline/agentic_rag_multi_tool.py)

## Quick Summary

Use `agentic_rag_multi_tool.py` when you want a simple demo of:

- local PDF RAG
- web lookup for recent information
- direct LLM answers for general questions

all combined into one interactive command-line workflow.
