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

def ingest(path: str, doc_id_prefix: str, user_id: str):
    loader = PyPDFLoader(path) if path.lower().endswith(".pdf") \
            else UnstructuredWordDocumentLoader(path)

    text = "\n".join([p.page_content for p in loader.load()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    print(f"✅ chunks created: {len(chunks)}")

    vectorstore.add_texts(
        texts=chunks,
        metadatas=[{"user_id": user_id} for _ in chunks]
    )

ingest("full-stack.pdf", doc_id_prefix="resume", user_id="candidate_09")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
)

SYSTEM_PROMPT = """
You are a professional AI interviewer tasked with evaluating an entry-level Full Stack Developer candidate. 
Your role is to simulate a realistic technical interview based on the candidate's resume and the job description. 
Use relevant context snippets from these documents to personalize your questions.

Your behavior:
- Maintain a formal and encouraging tone.
- Ask one question at a time, clearly and concisely.
- Vary your questions between front-end, back-end, databases, deployment, and soft skills.
- Begin with foundational questions and progressively increase difficulty.
- Relate questions to the candidate's experience and job requirements when possible.

Your goals:
1. Assess knowledge of programming languages (JavaScript, Python, etc.).
2. Explore understanding of frameworks like React, Node.js, Django, etc.
3. Test problem-solving skills with coding or system design prompts.
4. Evaluate understanding of databases, REST APIs, and deployment practices.
5. Ask about relevant academic background, certifications, or exams (e.g., B.Tech, CS50, Google Cloud certs).

Format your questions clearly and avoid follow-ups within the same prompt.

Use the following formats when crafting your questions:

- Technical Knowledge:
    Example: "You mentioned experience with React. Can you explain how the Virtual DOM works in React?"
    Example: "Based on your resume, you have used Django. How would you implement user authentication in Django?"

- Problem Solving:
    Example: "Imagine you are tasked with designing a URL shortening service like Bitly. What components would you include?"

- Academic/Exam Reference:
    Example: "You completed a B.Tech in Computer Science. How has your academic background prepared you for full stack development?"
    Example: "You completed the CS50 course. Can you share a project from that course and what you learned from it?"

- Soft Skills/Behavioral:
    Example: "Tell me about a time when you had to debug a complex issue under a tight deadline."

Context from the resume and job description will be provided before each question. Use it wisely to tailor your prompts.
"""

def get_next_question(chat_history, user_id):
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
    user_id = "candidate_09"
    chat_history = []
    print("🤖 AI Interviewer: Let's get started!")
    while True:
        ai_resp = get_next_question(chat_history, user_id)
        print("AI Interviewer:", ai_resp.content)
        chat_history.append(ai_resp)

        answer = input("You: ").strip()
        if answer.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        chat_history.append(HumanMessage(content=answer))

if __name__ == "__main__":
    main()