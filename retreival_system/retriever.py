import faiss

import typing
import os 
import sys
from datetime import datetime

import torch
import pickle

import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
from colbert import Indexer, Searcher

import sys
sys.path.append("./")
import retreival_system.bm25 as bm25 

import spacy


INDEX_PATH = "indices/coversations.index"

CONVERATIONS_DIR = "conversations/"

COLBERT_MODEL_PATH = "../llm_models/colbertv2.0"
COLBERT_INDEX_PATH = "indices/coversations.colbert.index"

BM25_INDEX_PATH = "indices/coversations.bm25.index"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class JinaRetriever:
    def __init__(self, dim=768) -> None:
        if os.path.exists(INDEX_PATH):
            self.index = faiss.read_index(INDEX_PATH)
        else:
            self.dim = dim
            self.index = faiss.IndexFlatL2(dim)
        self.tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en')
        self.model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
        self.load_new_docs(path = CONVERATIONS_DIR)
    
    def get_vector(self, doc : str):
        encoded_input = self.tokenizer([doc], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def add_document(self, doc : str):
        self.index.add(self.get_vector(doc))
        print(f"There are {self.index.ntotal} docs in the index")
        return self.index.ntotal
    
    def chat_to_passage(self, chat):
        ret_str = ""
        for elem in chat:
            role, content = elem["role"], elem["content"]
            ret_str += role + ":" + content + "\n"

        return ret_str            

    def load_new_docs(self, path):
        self.files = sorted(os.listdir(CONVERATIONS_DIR))
        residual_files = sorted(self.files)[self.index.ntotal:]

        print(f"Loading {len(residual_files)} docs...")
        for f in residual_files:
            with open(os.path.join(CONVERATIONS_DIR, f), "rb") as f:
                chat = pickle.load(f)
            self.add_document(self.chat_to_passage(chat))
        self.save_index(path=INDEX_PATH)
        
    def retrieve_k(self, query_vec : np.ndarray, k : int = 3):
        D, I = self.index.search(query_vec, k)
        return D[0], I[0]
    
    def save_index(self, path : str = "./index"):
        faiss.write_index(self.index, INDEX_PATH)
        return True

    def get_retrieved_docs(self, query:str, k:int = 3):
        D, I = self.retrieve_k(self.get_vector(query), k)
        ret_str = ""
        for i, idx in enumerate(I) :
            with open(os.path.join(CONVERATIONS_DIR, self.files[idx]), "rb") as f:
                chat = pickle.load(f)
            ret_str += f"Conversation {i} : \n"
            ret_str += self.chat_to_passage(chat)
        
        return ret_str

class ColBertRetriever:
    def __init__(self, dim=768) -> None:
        self.collections = []
        self.load_new_docs(path=CONVERATIONS_DIR)
    
    def load_and_save_index(self):
        cher = Searcher(index=COLBERT_INDEX_PATH, collection=self.colbert_collection)
    
    def chat_to_passage(self, chat):
        ret_str = ""
        for elem in chat:
            role, content = elem["role"], elem["content"]
            ret_str += role + ":" + content + "\n"

        return ret_str            

    def load_new_docs(self, path=CONVERATIONS_DIR):
        self.files = sorted(os.listdir(path))

        print(f"Loading {len(self.files)} docs...")
        for f in self.files:
            with open(os.path.join(CONVERATIONS_DIR, f), "rb") as f:
                chat = pickle.load(f)
            self.collections.append(self.chat_to_passage(chat))
        self.colbert_collection = Collection(path=CONVERATIONS_DIR, data=self.collections)
        self.config = ColBERTConfig(doc_maxlen=512, nbits=2)
        self.load_and_save_index()


    def get_retrieved_docs(self, query:str, k:int = 3):
        I, R, S = self.searcher.search(query, k)
        print(I, R, S)
        ret_str = ""
        for i, idx in enumerate(I) :
            ret_str += f"Conversation {i} : \n"
            ret_str += self.chat_to_passage(self.collections[idx])
        
        return ret_str
    
    
class BM25ColbertRetriever:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
        self.model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
        self.nlp = spacy.load("en_core_web_sm")
        self.bm_tokenizer = lambda x : [token.lemma_.lower() for token in self.nlp(x) if not token.is_stop and not token.is_punct]
        self.load_new_docs(path = CONVERATIONS_DIR)
    
    def load_index(self):
        if os.path.exists(BM25_INDEX_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
            if self.index.corpus_size < len(self.corpus):
                print("Adding new documents")
                self.index.add_documents(self.corpus[self.index.corpus_size:])
                self.save_index(path=BM25_INDEX_PATH)
        else:
            self.index = bm25.BM25Okapi([self.bm_tokenizer(doc) for doc in self.corpus])
            self.save_index(path=BM25_INDEX_PATH)
            
    def get_vector(self, doc : str):
        encoded_input = self.tokenizer([doc], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def chat_to_passage(self, chat):
        ret_str = ""
        for elem in chat:
            role, content = elem["role"], elem["content"]
            ret_str += role + " : " + content + "\n"

        return ret_str            

    def load_new_docs(self, path):
        self.files = sorted(os.listdir(CONVERATIONS_DIR))
        self.corpus = []
        print(f"Loading {len(self.files)} chats...")
        for f in self.files:
            with open(os.path.join(CONVERATIONS_DIR, f), "rb") as f:
                chat = pickle.load(f)
            self.corpus.append(self.chat_to_passage(chat))
        print(f"Loaded {len(self.corpus)} chats")
        self.load_index()
    
    def save_index(self, path : str = "./index"):
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        return True
    
    scores = []

    # Function to compute MaxSim
    def maxsim(self, query_embedding, document_embedding):
        ## creds : https://gist.github.com/virattt/b140fb4bf549b6125d53aa153dc53be6
        # Expand dimensions for broadcasting
        # Query: [batch_size, query_length, embedding_size] -> [batch_size, query_length, 1, embedding_size]
        # Document: [batch_size, doc_length, embedding_size] -> [batch_size, 1, doc_length, embedding_size]
        expanded_query = query_embedding.unsqueeze(2)
        expanded_doc = document_embedding.unsqueeze(1)

        # Compute cosine similarity across the embedding dimension
        sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)

        # Take the maximum similarity for each query token (across all document tokens)
        # sim_matrix shape: [batch_size, query_length, doc_length]
        max_sim_scores, _ = torch.max(sim_matrix, dim=2)

        # Average these maximum scores across all query tokens
        avg_max_sim = torch.mean(max_sim_scores, dim=1)
        return avg_max_sim

    def reranker(self, I, query):
        # Encode the query
        scores = []
        query_encoding = self.tokenizer(query, return_tensors='pt')
        query_embedding = self.model(**query_encoding).last_hidden_state.mean(dim=1)

        # Get score for each document
        for idx in I:
            document = self.corpus[idx]
            document_encoding = self.tokenizer(document, return_tensors='pt', truncation=True, max_length=512)
            document_embedding = self.model(**document_encoding).last_hidden_state
            print("Document embedding shape :", document_embedding.shape)
            print("Document query shape :", query_embedding.shape)
            
            # Calculate MaxSim score
            score = self.maxsim(query_embedding.unsqueeze(0), document_embedding)
            scores.append((score.item(), idx))
        
        scores = sorted(scores)
        return [i[1] for i in scores]
    

    def get_retrieved_docs(self, query:str, k:int = 3):
        I = self.index.get_top_n(self.bm_tokenizer(query), n=3*k)
        print("Retrieved indices :", I)
        if len(I) > k:
            print("Starting reranking of documents")
            print("Before reranking", I)
            I = self.reranker(I, query)[:k]
            print("After reranking", I)
            
        ret_str = ""
        for idx in I :
            ret_str += f"Conversation on {str(datetime.fromtimestamp(int(self.files[idx].split('.')[0])).strftime('%B %d, %Y %I:%M'))} : \n"
            ret_str += self.corpus[idx]
        
        return ret_str


if __name__ == "__main__":
    rag = BM25ColbertRetriever()
    x = rag.get_retrieved_docs("electrodynamics")
    