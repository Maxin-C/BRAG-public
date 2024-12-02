import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import argparse

class ChunkSet(Dataset):
    def __init__(self, chunks) -> None:
        super(ChunkSet).__init__()
        self.chunks = chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, index):
        return self.chunks[index]


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

if __name__=="__name__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_path", type=str, default='', required=True)
    parser.add_argument("--model_dict_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--model_dict_path", type=str, default=None)
    parser.add_argument("--embedding_path", type=str, default='', required=True)

    parser.add_argument("--device", type=str, default='cuda:0')

    args = parser.parse_args()

    torch.cuda.empty_cache()

    text_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
    device = args.device
    text_encoder = TextEncoder(text_model)
    if args.model_dict_path != None:
        text_encoder.load_state_dict(torch.load(args.model_dict_path, map_location=device))
    else:
        text_encoder.to(device)

    chunks = json.load(open(args.chunk_path, "r"))
    chunk_set = ChunkSet(chunks)
    chunk_loader = DataLoader(chunk_set, batch_size=args.batch_size, shuffle=False)

    local_embedding = []
    for chunk_batch in tqdm(chunk_loader):
        chunk_tokenized = tokenizer(
            chunk_batch,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors='pt').to(device)
        with torch.no_grad():
            embedding = text_encoder(chunk_tokenized)
        local_embedding.extend(embedding.to("cpu").tolist())

    with open(args.embedding_path, "w", encoding="utf-8") as file:
        json.dump(local_embedding, file)
