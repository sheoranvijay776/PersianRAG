import os
import getpass
import bs4
from typing import List, TypedDict

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

from langgraph.graph import START, StateGraph

def setup_metis_api():
    if not os.environ.get("METIS_API_KEY"):
        key = "METIS_API_KEY"
        os.environ["METIS_API_KEY"] = key
    else:
        key = os.environ["METIS_API_KEY"]
    
    url = "https://api.metisai.ir/openai/v1"
    
    return key, url

class RAGSystem:
    
    def __init__(self):
        self.key, self.url = setup_metis_api()
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=self.key,
            base_url=self.url
        )
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.key,
            base_url=self.url,
            temperature=0
        )
        
        self.vs = None
        
        try:
            self.prompt = hub.pull("rlm/rag-prompt")
        except:
            from langchain_core.prompts import ChatPromptTemplate
            self.prompt = ChatPromptTemplate.from_template(
                """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}
Answer:"""
            )
        
        self.graph = None
    
    def load_documents(self, src=None, src_type="auto"):
        docs = []
        
        if src_type == "auto":
            if src is None:
                src_type = "web"
            elif isinstance(src, str):
                if src.startswith(("http://", "https://")):
                    src_type = "web"
                else:
                    src_type = "text"
            elif isinstance(src, list):
                if all(s.startswith(("http://", "https://")) for s in src):
                    src_type = "web"
                else:
                    src_type = "text"
        
        if src_type == "web":
            docs = self._load_web_docs(src)
        elif src_type == "text":
            docs = self._load_text_docs(src)
        else:
            raise ValueError(f"Unsupported src_type: {src_type}")
        
        print(f"Loaded {len(docs)} doc(s)")
        print(f"Total chars: {sum(len(doc.page_content) for doc in docs)}")
        
        return docs
    
    def _load_web_docs(self, urls):
        if urls is None:
            urls = ["https://lilianweng.github.io/posts/2023-06-23-agent/"]
        
        print(f"Loading docs from {len(urls)} URL(s)...")
        
        strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        
        loader = WebBaseLoader(
            web_paths=urls,
            bs_kwargs={"parse_only": strainer},
        )
        
        return loader.load()
    
    def _load_text_docs(self, paths):
        if isinstance(paths, str):
            paths = [paths]
        
        print(f"Loading docs from {len(paths)} text file(s)...")
        
        docs = []
        for path in paths:
            try:
                loader = TextLoader(path, encoding='utf-8')
                file_docs = loader.load()
                docs.extend(file_docs)
                print(f"✓ Loaded: {path}")
            except Exception as e:
                print(f"✗ Error loading {path}: {e}")
        
        return docs
    
    def split_docs(self, docs: List[Document], size: int = 1000, overlap: int = 200):
        print(f"Splitting docs into chunks (size: {size}, overlap: {overlap})...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            add_start_index=True,
        )
        
        splits = splitter.split_documents(docs)
        print(f"Split into {len(splits)} chunks")
        
        return splits
    
    def create_vs(self, docs: List[Document]):
        print("Creating vector store and generating embeddings...")
        
        self.vs = FAISS.from_documents(
            documents=docs,
            embedding=self.embeddings
        )
        
        print(f"Vector store created with {len(docs)} docs")
        
        return self.vs
    
    def index_docs(self, src=None, src_type="auto", size: int = 1000, overlap: int = 200):
        print("=" * 50)
        print("INDEXING PIPELINE")
        print("=" * 50)
        
        docs = self.load_documents(src, src_type)
        
        splits = self.split_docs(docs, size, overlap)
        
        self.create_vs(splits)
        
        print("Indexing completed successfully!")
        print("=" * 50)
    
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
    
    def retrieve(self, state: State):
        if self.vs is None:
            raise ValueError("Vector store not initialized. Please run index_docs() first.")
        
        docs = self.vs.similarity_search(state["question"], k=4)
        return {"context": docs}
    
    def generate(self, state: State):
        content = "\n\n".join(doc.page_content for doc in state["context"])
        msgs = self.prompt.invoke({"question": state["question"], "context": content})
        resp = self.llm.invoke(msgs)
        return {"answer": resp.content}
    
    def build_graph(self):
        builder = StateGraph(self.State)
        
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("generate", self.generate)
        
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        
        self.graph = builder.compile()
        
        print("RAG graph built successfully!")
        return self.graph
    
    def query(self, q: str):
        if self.graph is None:
            self.build_graph()
        
        if self.vs is None:
            raise ValueError("Vector store not initialized. Please run index_docs() first.")
        
        print(f"\nQuestion: {q}")
        print("-" * 50)
        
        result = self.graph.invoke({"question": q})
        
        print(f"Answer: {result['answer']}")
        print(f"\nUsed {len(result['context'])} context docs")
        
        return result
    
    def query_with_sources(self, q: str):
        result = self.query(q)
        
        print("\nSource Documents:")
        print("=" * 50)
        for i, doc in enumerate(result['context'], 1):
            print(f"\nSource {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                print(f"Source: {doc.metadata['source']}")
        
        return result

def main():
    print("RAG System Example")
    print("=" * 50)
    
    rag = RAGSystem()
    
    rag.index_docs()
    
    rag.build_graph()
    
    questions = [
        "What is Task Decomposition?",
        "What are the challenges in building LLM-powered autonomous agents?",
        "How does Chain of Thought prompting work?",
        "What is the difference between AutoGPT and BabyAGI?"
    ]
    
    print("\nRunning example queries...")
    print("=" * 50)
    
    for q in questions:
        try:
            result = rag.query(q)
            print("\n" + "="*50)
        except Exception as e:
            print(f"Error processing question '{q}': {e}")
    
    print("\nInteractive Mode (type 'quit' to exit)")
    print("=" * 50)
    
    while True:
        try:
            q = input("\nEnter your question: ").strip()
            if q.lower() in ['quit', 'exit', 'q']:
                break
            if q:
                rag.query_with_sources(q)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
