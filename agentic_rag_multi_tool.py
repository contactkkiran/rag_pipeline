from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from duckduckgo_search import DDGS

# =========================
# 🔐 Load ENV
# =========================
load_dotenv()

DATA_PATH = "docs"

# =========================
# 📄 LOAD DOCUMENTS
# =========================
all_docs = []

if not os.path.exists(DATA_PATH):
    print("❌ Folder not found. Create 'docs' and add PDFs.")
    exit()

for file in os.listdir(DATA_PATH):
    if file.lower().endswith(".pdf"):
        path = os.path.join(DATA_PATH, file)
        loader = PyMuPDFLoader(path)
        all_docs.extend(loader.load())

print("✅ Loaded pages:", len(all_docs))

# =========================
# ✂️ SPLIT (IMPROVED)
# =========================
splitter = CharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
)
docs = splitter.split_documents(all_docs)

print("✅ Chunks:", len(docs))

# =========================
# 🔢 EMBEDDINGS + FAISS
# =========================
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

print("✅ FAISS ready")

# =========================
# 🌐 WEB SEARCH
# =========================
def web_search(query):
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        return "\n".join([r["body"] for r in results])


# =========================
# 🤖 LLM
# =========================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =========================
# 🚀 MAIN FUNCTION
# =========================
def ask_question(query: str):

    query_lower = query.lower()

    # =========================
    # 🔥 HYBRID ROUTING (RULES FIRST)
    # =========================

    # Force SEARCH for business/domain queries
    if any(word in query_lower for word in ["loan", "service", "policy", "product", "offer"]):
        decision = "SEARCH"

    # Force WEB for real-time queries
    elif any(word in query_lower for word in ["latest", "news", "today", "current", "weather"]):
        decision = "WEB"

    else:
        # =========================
        # 🧠 LLM DECISION (FALLBACK)
        # =========================
        decision_prompt = f"""
You are a strict tool selector.

Choose:
- ANSWER → general knowledge
- SEARCH → document/business data
- WEB → latest info

Return ONLY:
ANSWER or SEARCH or WEB

Question: {query}
"""
        decision_raw = llm.invoke(decision_prompt).content.strip()
        decision = decision_raw.upper().split()[0]

    print("\n🧠 Decision:", decision)

    # =========================
    # 🔀 EXECUTION
    # =========================

    if decision == "SEARCH":
        results = vectorstore.similarity_search(query, k=5)

        print("\n🔍 Retrieved Chunks:\n")
        for r in results:
            print(r.page_content[:200], "\n---")

        context = "\n\n".join([r.page_content for r in results])

        prompt = f"""
You are a helpful assistant.

Use ONLY the context below.
If answer is not found, say "I don't know".

Context:
{context}

Question:
{query}
"""
        response = llm.invoke(prompt)

    elif decision == "WEB":
        web_data = web_search(query)

        prompt = f"""
Answer using this web data:

{web_data}

Question:
{query}
"""
        response = llm.invoke(prompt)

    elif decision == "ANSWER":
        response = llm.invoke(f"Answer clearly:\n{query}")

    else:
        print("⚠️ Unknown → fallback SEARCH")

        results = vectorstore.similarity_search(query, k=5)
        context = "\n\n".join([r.page_content for r in results])
        response = llm.invoke(context + "\n\n" + query)

    return response.content


# =========================
# 🧪 TEST LOOP
# =========================
while True:
    query = input("\nAsk (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    answer = ask_question(query)
    print("\n🤖 ANSWER:\n", answer)