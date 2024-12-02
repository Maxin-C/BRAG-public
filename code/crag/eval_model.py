import torch
import json
from tqdm import tqdm
import warnings
from PIL import Image
from eval_matrix import EvalMatrix
from retriever import Retriever
from copy import deepcopy
from transformers.generation.utils import GenerationConfig
from load_lora import LoadLoraModel
    
class EvalModel():
    def __init__(self, model=None, tokenizer=None, device="cuda:0", predictions:list=[], references:list=[], adapter_path=None, eval_matrix:EvalMatrix=None,retriever:Retriever=None):
        if predictions != [] and len(predictions) != len(references):
            warnings.warn("The length of predictions is not equal to references.")
        self.predictions = predictions
        self.references = references
        self.cot = []
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.result = {}
        self.retriever = retriever
        self.rag_content = []
        self.eval_matrix = eval_matrix

        if adapter_path is not None:
            load_lora_model = LoadLoraModel(model, adapter_path)
            self.model = load_lora_model.load_lora_adapter()
    
    def _gen_input(self, data):
        # Placeholder for actual implementation
        return None
    
    def _gen_rag_input(self, data, top_k):
        # Placeholder for actual implementation
        # Update self.rag_content here
        return None
    
    def _gen_pred_input(self, data):
        return None

    def _get_response(self, input_data, max_new_tokens):
        # Placeholder for actual implementation
        return None

    def _get_response_batch(self, input_data_batch, max_new_tokens, **kwargs):
        # Placeholder for actual implementation
        return None
    
    def _gen_pre_ref(self, response, data):
        self.references.append(data['answer'])
        self.predictions.append(response)
        return None
    
    def _gen_pre_ref_batch(self, response_batch, data_batch):
        self.references.extend([data["answer"] for data in data_batch])
        self.predictions.extend(response_batch)
        return None

    def check_model_and_tokenizer(self):
        if self.model is None or self.tokenizer is None:
            print("No model or tokenizer loaded.")
            return False
        return True
    
    def inference(self, dataset, max_new_tokens=1024, use_rag:bool=False, top_k:int=5):
        self.rag_content = []
        if not self.check_model_and_tokenizer():
            return None, None
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if use_rag:
                if self.retriever == None:
                    warnings("Please use load_retriever() to load RAG retriever. LLM is inferencing without RAG.")
                    input_data = self._gen_input(data)
                else:
                    input_data = self._gen_rag_input(data, top_k)
            else:
                input_data = self._gen_input(data)
            response = self._get_response(input_data, max_new_tokens=max_new_tokens)

            self._gen_pre_ref(response, data)
        return self.predictions, self.references
    
    def inference_batch(self, dataset, max_new_tokens=1024, batch_size=8, use_rag:bool=False, top_k:int=5, **kwargs):
        self.rag_content = []
        if not self.check_model_and_tokenizer():
            return None, None
        for data_id in tqdm(range(0, len(dataset), batch_size)):
            data_batch = dataset[data_id:data_id+batch_size]
            input_data_batch = []
            for data in data_batch:
                if use_rag:
                    if self.retriever == None:
                        warnings("Please use load_retriever() to load RAG retriever. LLM is inferencing without RAG.")
                        input_data_batch.append(self._gen_input(data))
                    else:
                        input_data_batch.append(self._gen_rag_input(data, top_k))
                else:
                    input_data_batch.append(self._gen_input(data))
            response_batch = self._get_response_batch(input_data_batch, max_new_tokens=max_new_tokens, **kwargs)
            if not isinstance(response_batch, list):
                raise TypeError("Response batch must be a list")

            self._gen_pre_ref_batch(response_batch, data_batch)
    
    def evaluate(self, is_save:bool=False, result_path:str=None, use_rag=False):
        if len(self.predictions) != len(self.references):
            warnings.warn("The length of predictions is not equal to references.")
            return None
        if self.eval_matrix == None:
            warnings.warn("Please initilize an evaluation matrix in self.eval_matrix.")
            return None
        self.result = self.eval_matrix.compute(self.predictions, self.references)
        if is_save and result_path is not None:
            prefix = "rag_" if use_rag else ""
            self.save_eval_result(result_path=f"{result_path}/{prefix}eval_result.json")
        return self.result 
    
    def get_eval_result(self):
        return self.result

    def load_pre_ref(self, pre_path, ref_path):
        try:
            with open(pre_path, "rb") as file:
                self.predictions = json.load(file)
            with open(ref_path, "rb") as file:
                self.references = json.load(file)
            if len(self.predictions) != len(self.references):
                warnings.warn("The length of predictions is not equal to references.")
            self.rag_content = []
        except Exception as e:
            print(f"Failed to load predictions and references: {e}")

    def load_model(self, model):
        self.model = model

    def load_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def load_retriever(self, retriever:Retriever):
        self.retriever = retriever

    def clear_pre_ref(self):
        self.predictions = []
        self.references = []
        self.rag_content = []
        self.cot = []
    
    def save_pre_ref(self, save_path, use_rag:bool=False):
        try:
            prefix = ""
            if use_rag:
                prefix = "rag_"
                with open(f"{save_path}/rag_content.json", "w", encoding="utf-8") as file:
                    json.dump(self.rag_content, file, ensure_ascii=False)
            with open(f"{save_path}/{prefix}predictions.json", "w", encoding="utf-8") as file:
                json.dump(self.predictions, file, ensure_ascii=False)
            with open(f"{save_path}/{prefix}references.json", "w", encoding="utf-8") as file:
                json.dump(self.references, file, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save predictions and references: {e}")
    
    def save_eval_result(self, result_path):
        try:
            with open(result_path, "w", encoding="utf-8") as file:
                json.dump(self.result, file, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to save evaluation result: {e}")

    def cal_single_multi(self, save_path, use_rag:bool=False):
        assert len(self.predictions) == len(self.references) != 0
        
        predictions_single = []
        references_single = []
        predictions_multi = []
        references_multi = []
        for i in range(len(self.references)):
            if len(self.references[i]) == 1:
                predictions_single.append(self.predictions[i])
                references_single.append(self.references[i])
            else:
                predictions_multi.append(self.predictions[i])
                references_multi.append(self.references[i])

        acc_single = self.eval_matrix.compute(predictions_single, references_single)
        acc_multi = self.eval_matrix.compute(predictions_multi, references_multi)

        prefix = "rag_" if use_rag else ""
        
        with open(f"{save_path}/{prefix}predictions_single.json", "w", encoding="utf-8") as file:
            json.dump(predictions_single, file, ensure_ascii=False)

        with open(f"{save_path}/{prefix}references_single.json", "w", encoding="utf-8") as file:
            json.dump(references_single, file, ensure_ascii=False)

        with open(f"{save_path}/{prefix}eval_result_single.json", "w", encoding="utf-8") as file:
            json.dump(acc_single, file, ensure_ascii=False)

        with open(f"{save_path}/{prefix}predictions_multi.json", "w", encoding="utf-8") as file:
            json.dump(predictions_multi, file, ensure_ascii=False)

        with open(f"{save_path}/{prefix}references_multi.json", "w", encoding="utf-8") as file:
            json.dump(references_multi, file, ensure_ascii=False)

        with open(f"{save_path}/{prefix}eval_result_multi.json", "w", encoding="utf-8") as file:
            json.dump(acc_multi, file, ensure_ascii=False)

# --------------------------------------------------------------------------------------------------
# Local LLM
    
class EvalQwen(EvalModel):
    def _get_response(self, input_data, max_new_tokens=1024):
        text = self.tokenizer.apply_chat_template(
            input_data,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def _get_response_batch(self, input_data_batch, max_new_tokens=1024):
        text_batch = []
        for input_data in input_data_batch:
            text_batch.append(self.tokenizer.apply_chat_template(
                input_data,
                tokenize=False,
                add_generation_prompt=True,
                padding = True,
                truncation=True,
                max_length=2048
            ))
        model_inputs = self.tokenizer(text_batch, padding=True, truncation=True, max_length=2048, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return response

class EvalHuatuo(EvalModel):
    def build_chat_input(self, model, tokenizer, messages, max_new_tokens: int=0):
        def _parse_messages(messages, split_role="user"):
            system, rounds = "", []
            round = []
            for i, message in enumerate(messages):
                # if message["role"] == "system":
                #     assert i == 0
                #     system = message["content"]
                #     continue
                if message["role"] == split_role and round:
                    rounds.append(round)
                    round = []
                round.append(message)
            if round:
                rounds.append(round)
            return system, rounds
        
        max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
        max_input_tokens = model.config.model_max_length - max_new_tokens
        system, rounds = _parse_messages(messages, split_role="user")
        max_history_tokens = max_input_tokens
        roles = ('<问>：','<答>：')
        sep = '\n'

        history_tokens = []
        for round in rounds[::-1]:
            round_tokens = []
            for message in round:
                message["content"]
                if message["role"] == "user":
                    round_tokens.extend(tokenizer.encode(roles[0]+message["content"]+sep))
                else:
                    round_tokens.extend(tokenizer.encode(roles[1]+message["content"]+sep))
            if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
                history_tokens = round_tokens + history_tokens  # concat left
                if len(history_tokens) < max_history_tokens:
                    continue
            break

        input_tokens = history_tokens
        if messages[-1]["role"] != "assistant":
            input_tokens.extend(tokenizer.encode(roles[1]))
        # debug
        input_tokens = input_tokens[-max_input_tokens:]  # truncate left
        # print(tokenizer.decode(input_tokens),flush=True)
        return torch.LongTensor([input_tokens]).to(model.device)
    
    def _get_response(self, input_data, max_new_tokens=1024):
        self.model.generation_config = GenerationConfig.from_dict({
            "max_new_tokens": max_new_tokens,
        })
        response = self.model.HuatuoChat(self.tokenizer, input_data)
        return response
    
    def _get_response_batch(self, input_data_batch, max_new_tokens=1024):
        model_inputs = []
        generation_config = GenerationConfig.from_dict({
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "max_new_tokens": max_new_tokens,
            "temperature": 0.8,
            "top_k": 3,
            "top_p": 0.7,
            "repetition_penalty": 1.1,
            "do_sample": True
        })
        self.model.generation_config = generation_config

        model_inputs = []
        for input_data in input_data_batch:
            model_input = self.build_chat_input(self.model, self.tokenizer, input_data, generation_config.max_new_tokens)
            model_inputs.append(model_input)

        generated_ids = self.model.generate(
            model_inputs,
            generation_config
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return response

# --------------------------------------------------------------------------------------------------
# API Calling LLM
class EvalGPT4(EvalModel):
    def _get_response(self, input_data, max_new_tokens=1024):
        response = self.model.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=input_data,
                max_tokens = max_new_tokens
            )
        return response.choices[0].message.content
    
    def _get_response_batch(self, input_data_batch, **kwargs):
        warnings("no response batch function can be used!")
        return super()._get_response_batch(input_data_batch, **kwargs)
    
    def inference(self, dataset, use_rag:bool=False, top_k:int=5, max_new_tokens=1024):
        if not self.check_model_and_tokenizer():
            return None, None
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if use_rag:
                if self.retriever == None:
                    warnings("Please use load_retriever() to load RAG retriever. LLM is inferencing without RAG.")
                    input_data = self._gen_input(data)
                else:
                    input_data = self._gen_rag_input(data, top_k)
            else:
                input_data = self._gen_input(data)
            response = "error"

            for _ in range(5):
                try:
                    response = self._get_response(input_data, max_new_tokens)
                except Exception as e:
                    continue
                else:
                    break

            self.predictions.append(response)
            self.references.append(data["answer"])
        return self.predictions, self.references
    
    def inference_batch(self, dataset, max_new_tokens=1024, batch_size=8, use_rag: bool = False, top_k: int = 5, **kwargs):
        if not self.check_model_and_tokenizer():
            return None, None
        for i in tqdm(range(len(dataset))):
            data = dataset[i]
            if use_rag:
                if self.retriever == None:
                    warnings("Please use load_retriever() to load RAG retriever. LLM is inferencing without RAG.")
                    input_data = self._gen_input(data)
                else:
                    input_data = self._gen_rag_input(data, top_k)
            else:
                input_data = self._gen_input(data)
            response = "error"

            for _ in range(5):
                try:
                    response = self._get_response(input_data, max_new_tokens)
                except Exception as e:
                    continue
                else:
                    break

            self.predictions.append(response)
            self.references.append(data["answer"])
        return self.predictions, self.references