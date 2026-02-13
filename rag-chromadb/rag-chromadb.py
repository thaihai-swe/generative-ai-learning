import os
import json
import logging
from pprint import pprint
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from wikipediaapi import Wikipedia

# Load environment variables
load_dotenv()

# Configuration
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY", "lm-studio")
OPEN_AI_API_BASE_URL = os.getenv("OPEN_AI_API_BASE_URL", "http://127.0.0.1:1234/v1")
OPEN_AI_MODEL = os.getenv("OPEN_AI_MODEL", "meta-llama-3.1-8b-instruct")
MAX_TOOL_CALLS = 10  # Prevent infinite loops

client = OpenAI(base_url=OPEN_AI_API_BASE_URL, api_key=OPEN_AI_API_KEY)




db_client = chromadb.PersistentClient(path="./chroma_db")
db_client.heartbeat()
embedding_function = embedding_functions.DefaultEmbeddingFunction()

# Set a proper user agent for Wikipedia API
USER_AGENT = "generative-ai-learning/1.0 (contact: your-email@example.com)"
wiki = Wikipedia(user_agent=USER_AGENT, language="en")


def load_wikipedia_page(page_name):
    """Load a Wikipedia page and add it to the collection."""
    try:
        print(f"\n📖 Loading Wikipedia page: {page_name}...")
        page = wiki.page(page_name)

        if not page.exists():
            print(f"❌ Page '{page_name}' not found on Wikipedia. Please try another page.")
            return None

        docs = page.text.split("\n\n")  # Split into paragraphs

        if not docs or docs[0].strip() == "":
            print(f"❌ Could not retrieve content from '{page_name}'.")
            return None

        # Use sanitized page name for collection
        collection_name = page_name.replace(" ", "_").lower()

        # Create or get collection
        collection = db_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        # Add documents to collection
        collection.add(
            ids=[f"{collection_name}_{i}" for i in range(len(docs))],
            documents=docs,
            metadatas=[{"source": page_name, "index": i} for i in range(len(docs))]
        )

        print(f"✅ Successfully loaded {len(docs)} paragraphs from '{page_name}'")
        print(f"📄 Summary: {docs[0][:150]}...\n")

        return collection

    except Exception as e:
        print(f"❌ Error loading page: {str(e)}")
        return None


def ask_question(collection, question, n_results=2):
    """Query the collection and get an answer."""
    try:
        print(f"\n🔍 Querying with: {question}")

        query = collection.query(query_texts=[question], n_results=n_results)

        if not query['documents'] or not query['documents'][0]:
            print("❌ No relevant documents found.")
            return None

        # Build context from retrieved documents
        CONTEXT = ""
        for doc in query['documents'][0]:
            CONTEXT += doc + "\n"

        print("✅ Retrieved relevant context")

        # Get answer from LLM
        response = client.chat.completions.create(
            model=OPEN_AI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides accurate information based on the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context: {CONTEXT}\n\nQuestion: {question}\n\nAnswer the question based on the context provided."
                }
            ],
            temperature=0.2,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"❌ Error processing question: {str(e)}")
        return None


def main():
    """Main interactive loop."""
    print("\n" + "="*60)
    print("🤖 RAG-based Q&A System with Wikipedia and ChromaDB")
    print("="*60)

    current_collection = None
    current_page = None

    while True:
        if current_collection is None:
            # Get Wikipedia page from user
            page_name = input("\n📝 Enter Wikipedia page name (or 'quit' to exit): ").strip()

            if page_name.lower() == 'quit':
                print("👋 Goodbye!")
                break

            if not page_name:
                print("❌ Please enter a valid page name.")
                continue

            collection = load_wikipedia_page(page_name)
            if collection:
                current_collection = collection
                current_page = page_name
            else:
                continue

        else:
            # Ask question about current page
            print(f"\n📚 Currently exploring: {current_page}")
            question = input("❓ Ask your question (or 'change' to switch page, 'quit' to exit): ").strip()

            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break

            if question.lower() == 'change':
                current_collection = None
                current_page = None
                continue

            if not question:
                print("❌ Please enter a valid question.")
                continue

            answer = ask_question(current_collection, question)
            if answer:
                print(f"\n💡 Answer:\n{answer}")


if __name__ == "__main__":
    main()
