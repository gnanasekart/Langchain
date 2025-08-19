"""Chunk repository files, compute embeddings, and store in Chroma.

Example:
    python deeplearn\vectorize_and_store.py --path ../someproject --collection my_collection
"""
import argparse
import os
from pathlib import Path
from typing import List

from langchain.text_splitter import CharacterTextSplitter
import chromadb
from chromadb.config import Settings

from deeplearn.load_repo import gather_files
from deeplearn.embeddings import embed_texts


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True, help='Local folder or git URL to index')
    parser.add_argument('--collection', default='deeplearn_collection', help='Chroma collection name')
    parser.add_argument('--persist_path', default='./deeplearn_chroma_db', help='Chroma DB folder')
    args = parser.parse_args()

    print('Gathering files...')
    files = gather_files(args.path)
    print(f'Found {len(files)} files')

    # Prepare Chroma client
    client = chromadb.Client(Settings(chroma_db_impl='duckdb+parquet', persist_directory=str(Path(args.persist_path).absolute())))
    collection = client.get_or_create_collection(name=args.collection)

    ids = []
    metadatas = []
    documents = []

    for relpath, text in files:
        chunks = chunk_text(text)
        for i, c in enumerate(chunks):
            ids.append(f"{relpath}__{i}")
            metadatas.append({'source': relpath})
            documents.append(c)

    print(f'Embedding {len(documents)} chunks...')
    embeddings = embed_texts(documents)

    print('Upserting into Chroma...')
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings.tolist())
    client.persist()
    print('Done.')


if __name__ == '__main__':
    main()
