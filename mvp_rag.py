"""
MVP RAG System - Simple Text File Q&A
====================================

A minimal viable product for Retrieval Augmented Generation
that works with local text files.

Usage:
    python mvp_rag.py
"""

import os
import getpass
from langrag import RAGSystem

def main():
    print("MVP RAG System - Text File Q&A")
    print("=" * 50)
    
    print("Initializing RAG system...")
    rag = RAGSystem()
    
    txt = "doc.txt"
    if not os.path.exists(txt):
        print(f"❌ Error: {txt} not found!")
        print("Please make sure you have a text file to index.")
        return
    
    print(f" Found text file: {txt}")
    
    print("\n Indexing document...")
    try:
        rag.index_docs(src=txt, src_type="text")
        print(" Indexing completed!")
    except Exception as e:
        print(f" Error during indexing: {e}")
        return
    
    print(" Building RAG graph...")
    rag.build_graph()
    print(" RAG system ready!")
    
    print("\n Testing with sample questions...")
    print("=" * 50)
    
    print(f"\n Interactive Mode")
    print("Ask questions about your document!")
    print("(Type 'quit', 'exit', or 'q' to stop)")
    print("=" * 50)
    
    while True:
        try:
            q = input("\n❓ Your question: ").strip()
            
            if q.lower() in ['quit', 'exit', 'q', '']:
                print(" Goodbye!")
                break
                
            if q:
                print("\n Thinking...")
                result = rag.query_with_sources(q)
                
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f" Error: {e}")
            print("Please try again with a different question.")

if __name__ == "__main__":
    print("Checking requirements...")
    
    try:
        import langchain
        import faiss
        print(" Dependencies OK")
    except ImportError as e:
        print(f" Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        exit(1)
    
    main()
