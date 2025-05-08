import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

COLLECTION_NAME = "interview_context"
# Qdrant client setup
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = QdrantVectorStore.from_texts(
    texts=[],
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    collection_name=COLLECTION_NAME,
    distance=models.Distance.COSINE 
)
# ——————————————————————————————
# 2. LOAD + CHUNK + STORE DOCUMENTS
# ——————————————————————————————

def ingest(path: str, doc_id_prefix: str):
    loader = PyPDFLoader(path) if path.lower().endswith(".pdf") \
            else UnstructuredWordDocumentLoader(path)

    text = "\n".join([p.page_content for p in loader.load()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"✅ chunks created: {len(chunks)}")

    metadatas = [{"source": f"{doc_id_prefix}-{i}"} for i in range(len(chunks))]
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)

ingest("full-stack.pdf", doc_id_prefix="resume")

# ——————————————————————————————
# 3. INTERVIEW LOOP
# ——————————————————————————————
# Gemini LLM for chat
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,    
)

SYSTEM_PROMPT = """You are an AI interviewer for an entry-level Full Stack Developer candidate.
Use the retrieved resume & job-description snippets to craft a single, clear next question.
Keep it formal, focused, and one at a time."""


def get_next_question(chat_history):
    query = " ".join([m.content for m in chat_history if isinstance(m, HumanMessage)])
    if not query.strip():
        query = "Start the interview"

    docs = vectorstore.similarity_search(query, k=5)
    context = "\n---\n".join(d.page_content for d in docs)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"Context:\n{context}"),
        HumanMessage(content=query),
    ] + chat_history

    return llm.invoke(messages) 


def main():
    chat_history = []
    print("🤖 AI Interviewer: Let's get started!")
    while True:
        ai_resp = get_next_question(chat_history)
        print("AI Interviewer:", ai_resp.content)
        chat_history.append(ai_resp)

        answer = input("You: ").strip()
        if answer.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        chat_history.append(HumanMessage(content=answer))

if __name__ == "__main__":
    main()