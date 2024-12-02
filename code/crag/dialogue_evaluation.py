import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from openai import OpenAI

from eval_matrix import EvalMatrixSentence
from eval_model import EvalModel, EvalQwen, EvalGPT4, EvalHuatuo
from retriever import RetNomic

class EvalQwen(EvalQwen):
    def _gen_input(self, data):
        messages = [
            {"role": "user", "content": data['question']}
        ]
        return messages

    def _gen_rag_input(self, data, top_k):
        query = data['question']
        
        chunk_list = self.retriever.retrieve_knowledge([data['question']], top_k)[0]
        context = "请参考以下信息完成用户指定任务：\n"
        for i in chunk_list:
            context += i
            context += "\n"

        messages  = [
            {"role": "system", "content": context},
            {'role': 'user', 'content': query}]

        return messages

class EvalHuatuo(EvalHuatuo):
    def _gen_input(self, data):
        query = data['question']
        messages  = [{'role': 'user', 'content': query}]
        return messages
    
    def _gen_rag_input(self, data, top_k):
        query = data['question']
        
        chunk_list = self.retriever.retrieve_knowledge([data['question']], top_k)[0]
        context = "请参考以下信息完成用户指定任务：\n"
        for i in chunk_list:
            context += i
            context += "\n"

        messages  = [
            {"role": "system", "content": "You are a helpful assistant."},
            {'role': 'user', 'content': f"上下文信息：\n{context}\n{query}"}
            ]

        return messages

class EvalGPT4(EvalGPT4):
    def _gen_input(self, data):
        query = data['question']
        messages  = [{'role': 'user', 'content': query}]
        return messages
        
    def _gen_rag_input(self, data, top_k):
        query = data['question']
        
        chunk_list = self.retriever.retrieve_knowledge([data['question']], top_k)[0]
        context = "请参考以下信息完成用户指定任务：\n"
        for i in chunk_list:
            context += i
            context += "\n"

        messages  = [
            {"role": "system", "content": context},
            {'role': 'user', 'content': query}]

        return messages

def eval_models(eval_model:EvalModel, dataset, use_rag, batch_size, top_k, max_new_tokens, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    eval_model.clear_pre_ref()
    eval_model.inference_batch(dataset, batch_size=batch_size, use_rag=use_rag, top_k=top_k, max_new_tokens=max_new_tokens)
    eval_model.evaluate(is_save=True, result_path=output_path, use_rag=use_rag)
    eval_model.save_pre_ref(output_path, use_rag=use_rag)

if __name__=="__name__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--generator_model_path", default="Qwen/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--openai_key", default=None, type=str)
    parser.add_argument("--retriever_model_dict_path", default=None, type=str)

    parser.add_argument("--use_rag", default=False, type=bool)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--chunks_path", type=str, required=True)
    parser.add_argument("--embeddings_path", type=str, help="Embedding of the chunks knowledge base", required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()

    dataset = json.load(open(args.dataset_path, "rb"))

    if "gpt" in args.generator_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.generator_model_path, trust_remote_code=True, torch_dtype=torch.float16).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path, trust_remote_code=True, padding_side='left')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.generator_model_path, trust_remote_code=True, torch_dtype=torch.float16).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path, trust_remote_code=True, padding_side='left')


    retriver = RetNomic(
        chunks_path=args.chunks_path,
        embeddings_path=args.embeddings_path,
        model_dict_path=args.retriever_model_dict_path,
        device=args.device
    )

    if args.openai_key != None:
        model = OpenAI(api_key=args.openai_key)
        tokenizer = []
        eval_crag = EvalGPT4(model=model, tokenizer=tokenizer, device=args.device, eval_matrix=EvalMatrixSentence(), retriever=None)
    elif "Qwen" in args.generator_model_path or "Llama" in args.generator_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.generator_model_path, trust_remote_code=True, torch_dtype=torch.float16).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path, trust_remote_code=True, padding_side='left')
        eval_crag = EvalQwen(model=model, tokenizer=tokenizer, eval_matrix=EvalMatrixSentence(device=args.device), device=args.device, retriever=retriver)
    elif "HuatuoGPT" in args.generator_model_path:
        model = AutoModelForCausalLM.from_pretrained(args.generator_model_path, trust_remote_code=True, torch_dtype=torch.float16).to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path, trust_remote_code=True, padding_side='left')
        eval_crag = EvalHuatuo(model=model, tokenizer=tokenizer, eval_matrix=EvalMatrixSentence(device=args.device), device=args.device, retriever=retriver)

    eval_models(eval_crag, dataset, use_rag=args.use_rag, batch_size=args.batch_size, top_k=args.top_k, max_new_tokens=args.max_new_tokens, output_path=args.output_path)


