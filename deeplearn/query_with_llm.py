"""Query the vector store and forward the context to a local Ollama model `deepseek-coder:6.7b`.

Example:
    python deeplearn\query_with_llm.py --collection deeplearn_collection --query "How does auth work?"
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import chromadb
from chromadb.config import Settings


def get_top_chunks(collection_name: str, query: str, persist_path: str = './deeplearn_chroma_db', k: int = 4):
    client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=str(Path(persist_path).absolute())))
    coll = client.get_collection(name=collection_name)
    res = coll.query(query_texts=[query], n_results=k)
    docs = []
    for docs_list in res.get('documents', []):
        docs = docs_list
    return docs


def call_ollama(prompt: str, model: str = 'deepseek-coder:6.7b') -> str:
    # Try to use `ollama` CLI. This requires Ollama to be installed and model present locally.
    try:
        p = subprocess.run(['ollama', 'run', model, '--prompt', prompt], capture_output=True, text=True, check=True)
        return p.stdout
    except Exception as e:
        return f'Ollama call failed: {e}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', required=True)
    parser.add_argument('--query', required=True)
    parser.add_argument('--persist_path', default='./deeplearn_chroma_db')
    args = parser.parse_args()

    top_chunks = get_top_chunks(args.collection, args.query, persist_path=args.persist_path)
    context = '\n\n---\n\n'.join(top_chunks)

    prompt = f"You are a code assistant. Use the following context from a repository to answer the user question.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{args.query}\n\nAnswer concisely and reference file paths when possible."

    print('Calling local Ollama model...')
    out = call_ollama(prompt)
    print(out)


if __name__ == '__main__':
    main()
