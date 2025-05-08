import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from mem0 import Memory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

QUADRANT_HOST = "localhost"

NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "aWUmTLIcezhT_UZ8iuq-76Q7LqCUqOijEY23k93ZgY4"

config = {
    "version": "v1.1",
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
        }
    },
    "llm": {
        "provider": "gemini",
        "config": {
            "model": "gemini-1.5-flash-latest",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "test",
            "host": "localhost",
            "port": 6333,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {     
            "url": NEO4J_URL,
            "username": NEO4J_USERNAME,
            "password": NEO4J_PASSWORD
        },
    },
}

memory = Memory.from_config(config)

class State(TypedDict):
    messages: Annotated[list, add_messages]

class ResumeProcessor:
    def __init__(self):
        self.memory = memory
        self.chunks = []

    def create_chunks(self, text: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.chunks = splitter.split_text(text)
        return self.chunks

    def load_documents(self, resume_path: str, jd_path: str = ""):
        try:
            if resume_path.endswith(".pdf"):
                resume_loader = PyPDFLoader(resume_path)
            else:
                resume_loader = UnstructuredLoader(resume_path)
            resume_text = resume_loader.load()[0].page_content

            jd_text = ""
            if jd_path:
                if jd_path.endswith(".pdf"):
                    jd_loader = PyPDFLoader(jd_path)
                else:
                    jd_loader = UnstructuredLoader(jd_path)
                jd_text = jd_loader.load()[0].page_content

            return resume_text, jd_text
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return "", ""

    
    def process_and_store(self, resume_text, jd_text, user_id):
        chunks = self.create_chunks(resume_text)
        formatted = [{"role": "user", "content": chunk} for chunk in chunks]
        self.memory.add(formatted, user_id=user_id)         

class InterviewChatbot:
    def __init__(self): 
        self.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
        self.memory = memory
        self.base_system_prompt = """You are a Professional AI Interviewer for a Full Stack Developer position, interviewing a student with 0 years of professional experience.
- Use the resume to tailor your interview questions.
- Ask one question at a time.
- Focus on assessing understanding of core web development concepts (HTML, CSS, JavaScript, React, Node.js, databases).
- Ask about projects, internships, coursework, or personal learning.
- Stay formal, objective, and encouraging.
- Avoid jargon-heavy or advanced corporate process questions."""    

    def run(self, state: State, user_id: str):  
        query = state["messages"][-1]["content"].strip()
        if not query:
            return {"messages":[{"role":"assistant",
                                "content":"Please ask a question or type ‘exit’ to quit."}]}

        print(f"🧠 User query: {query}")

        try:
            mem_result = self.memory.search(query=query, user_id=user_id)
            print(f"🔍 Memory search results: {mem_result}")
        except Exception as e:
            print(f"❌ Error during memory search: {e}")
            mem_result = {}

        memories = "\n".join([m["memory"] for m in mem_result.get("results", [])])

        if memories:
            print("📎 Using retrieved memory context.")
            enhanced_system_prompt = f"{self.base_system_prompt}\nRelevant context:\n{memories}"
        else:
            print("📎 No relevant memory found.")
            enhanced_system_prompt = self.base_system_prompt

        messages = [
            { "role": "system", "content": enhanced_system_prompt },
            { "role": "user", "content": query }
        ]

        try:
            response = self.llm.invoke(messages)
            print(f"🗣️ LLM response: {response.content}")
        except Exception as e:
            print(f"❌ Error during LLM response: {e}")
            return {"messages": [{"role": "assistant", "content": "Sorry, something went wrong."}]}

        interaction = [
            { "role": "user", "content": query },
            { "role": "assistant", "content": response.content }
        ]

        try:
            self.memory.add(interaction, user_id=user_id)
            print("🧠 Interaction stored successfully in memory")
        except Exception as e:
            print(f"❌ Failed to store interaction in memory: {e}")

        return {"messages": [response]}

if __name__ == "__main__":
    user_id = "candidate_1234"

    processor = ResumeProcessor()
    resume_path = "full-stack.pdf"
    jd_path = "job_description.pdf"

    resume_text, _ = processor.load_documents(resume_path, jd_path="")
    processor.process_and_store(resume_text, jd_text="", user_id=user_id)

    chatbot = InterviewChatbot()
    print("\n🤖 AI Interviewer is ready. Start your mock interview.")

    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break

        state = {
            "messages": [{"role": "user", "content": user_input}]
        }

        result = chatbot.run(state, user_id)
        print("Bot:", result["messages"][-1]["content"])