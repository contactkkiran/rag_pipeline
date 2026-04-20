from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader


DATA_PATH = "docs"

all_docs = []

# Check folder exists
if not os.path.exists(DATA_PATH):
    print("❌ Folder not found. Create 'docs' and add PDFs.")
    exit()

# Load PDFs
for file in os.listdir(DATA_PATH):
    if file.lower().endswith(".pdf"):
        full_path = os.path.join(DATA_PATH, file)
        print("Loading:", full_path)

        loader = PyMuPDFLoader(full_path)
        docs = loader.load()

        all_docs.extend(docs)

print("\n✅ Total pages loaded:", len(all_docs))


# CharacterTextSplitter is a simple text splitter that splits text based 
# on character count. It allows you to specify the chunk size and the overlap between chunks. This is useful for processing large documents in smaller, manageable pieces while retaining some context between chunks.
from langchain_text_splitters import CharacterTextSplitter


splitter = CharacterTextSplitter(
    chunk_size=100,     # smaller
    chunk_overlap=20
)
# Create chunks from the loaded documents
docs = splitter.split_documents(all_docs)

print("Chunks:", len(docs))

# This converts text → numbers (vectors)

from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

# Test embedding on one chunk
vector = embeddings.embed_query(docs[0].page_content)
print("Vector:", vector)
print("Vector length:", len(vector))

from langchain_community.vectorstores import FAISS

# Create FAISS index from chunks
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")  # Save the index locally
print("✅ FAISS index created")


# LLM generates answer from retrieved chunks
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",   # fast + cost effective
    temperature=0
)

query = "What loan services are available?"

results = vectorstore.similarity_search(query, k=3)

# NEXT STEP → Send to LLM 
# Step1 create context from retrieved chunks
context = "\n\n".join([r.page_content for r in results]) # Combine retrieved chunks into context

# Step 2: Create prompt
prompt = f"""
Answer the question based ONLY on the context below.

Context:
{context}

Question:
{query}
"""

# Step 3: Call LLM

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = llm.invoke(prompt)

# Step 4: Print final answer
print("\n🤖 FINAL ANSWER:\n")
print(response.content)