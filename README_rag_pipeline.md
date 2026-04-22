# RAG Pipeline with LangChain, OpenAI, and FAISS

This project builds a simple Retrieval-Augmented Generation (RAG) pipeline over local PDF documents.

It:

- loads PDF files from the `docs/` folder
- splits them into small text chunks
- generates embeddings with OpenAI
- stores vectors in a local FAISS index
- retrieves relevant chunks for a query
- sends the retrieved context to an OpenAI chat model for a final answer

## Project Structure

```text
rag_pipeline/
├── main.py
├── README.md
├── docs/
│   ├── account_opening.pdf
│   ├── bank_services.pdf
│   ├── customer_support.pdf
│   ├── digital_banking.pdf
│   └── loan_policies.pdf
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

## Requirements

- Python 3.10+
- An OpenAI API key

Install the Python packages used by `main.py`:

```bash
pip install python-dotenv langchain langchain-community langchain-openai langchain-text-splitters pymupdf faiss-cpu
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## How to Run

Run:

```bash
python main.py
```

## What `main.py` Does

1. Reads all PDF files from the `docs/` directory.
2. Loads each PDF with `PyMuPDFLoader`.
3. Splits document text into chunks using:
   - `chunk_size=100`
   - `chunk_overlap=20`
4. Creates embeddings with `OpenAIEmbeddings`.
5. Builds and saves a FAISS index in `faiss_index/`.
6. Runs a sample query:

```text
What loan services are available?
```

7. Retrieves the top 3 similar chunks.
8. Sends the retrieved context to `gpt-4o-mini`.
9. Prints the final answer in the terminal.

## Notes

- The script currently uses a hardcoded query inside `main.py`.
- The FAISS index is rebuilt each time you run the script.
- If the `docs/` folder is missing, the script exits with an error.
- `load_dotenv` is imported, but the current script does not call `load_dotenv()`. If your API key is stored in `.env`, add this near the top of `main.py`:

```python
load_dotenv()
```

## Example Use Case

This project is useful for:

- internal document question-answering
- banking or policy document search
- learning the basics of a RAG workflow with LangChain

## Future Improvements

- accept user queries from input instead of hardcoding them
- separate indexing and querying into different scripts
- add better chunking strategy
- avoid rebuilding the index on every run
- include citations/source pages in answers
- add a `requirements.txt`
