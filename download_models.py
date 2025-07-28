#!/usr/bin/env python3
"""
Script to download models during Docker build
"""
import logging

logging.basicConfig(level=logging.INFO)

def download_models():
    print('Downloading sentence transformer model for embedding...')
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('intfloat/e5-base-v2')
    
    print('Downloading cross encoder model for reranking...')
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    
    print('Models downloaded successfully!')

if __name__ == "__main__":
    download_models() 
