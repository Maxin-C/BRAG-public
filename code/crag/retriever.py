import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import faiss

class Retriever():
    def __init__(self) -> None:
        self.kb = None
        pass
    
    def load_kb(self):
        pass

    def retrieve_knowledge(self, query, top_k):
        return None

class TextEncoder(nn.Module):
    def __init__(self, text_model) -> None:
        super(TextEncoder, self).__init__()
        self.text_model = text_model
        self.output_dim = text_model.encoder.layers[-1].norm2.normalized_shape[0]
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text):
        model_output = self.text_model(**text)
        embeddings = self.mean_pooling(model_output, text['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

class RetNomic(Retriever):
    def __init__(self, chunks_path, embeddings_path, model_dict_path=None, device="cuda:0") -> None:
        with open(chunks_path, "r") as file:
            self.chunks = json.load(file)
        with open(embeddings_path, "r") as file:
            self.chunk_embeddings = json.load(file)

        dim = len(self.chunk_embeddings[0])
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.chunk_embeddings = np.array(self.chunk_embeddings)
        self.faiss_index.add(self.chunk_embeddings)

        text_model = AutoModel.from_pretrained("/mnt/pvc-data.common/ChenZikang/huggingface/nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/pvc-data.common/ChenZikang/huggingface/nomic-ai/nomic-embed-text-v1.5")
        self.text_encoder = TextEncoder(text_model)
        if model_dict_path != None:
            self.text_encoder.load_state_dict(torch.load(model_dict_path, map_location=device))
        self.text_encoder.to(device)

        self.device = device
    
    def retrieve_knowledge(self, query, top_k):
        chunk_tokenized = self.tokenizer(
            query,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors='pt').to(self.device)
        with torch.no_grad():
            embedding = self.text_encoder(chunk_tokenized)

        _, chunk_index = self.faiss_index.search(np.array(embedding.cpu()), k=top_k)

        chunks = []
        for i in range(len(query)):
            query_chunks = []
            for j in chunk_index[i]:
                query_chunks.append(self.chunks[j])
            chunks.append(query_chunks)
        return chunks
