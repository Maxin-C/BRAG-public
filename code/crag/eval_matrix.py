import numpy as np
import evaluate
from nltk.tokenize import word_tokenize
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
from rouge_chinese import Rouge
import jieba
import nltk
from bert_score import score
nltk.download('punkt')
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class EvalMatrix():
    def __init__(self) -> None:
        pass

    def compute(self, predictions, references):
        result = {}
        return result

class EvalMatrixSentence(EvalMatrix):
    """
    一个用于多种文本评价指标计算的类，包括ROUGE、BLEU、BERTScore和嵌入平均值。
    """

    class EmbedAve():
        """
        用于计算句子嵌入平均值的内部类。
        """
        def __init__(self, model_path:str, device:str) -> None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path).to(device)
            self.device = device

        def _encode(self, sentences:list, batch_size:int=256):
            with torch.no_grad():
                embeddings_collection = []
                # 分批次处理句子以获得嵌入表示
                for sentence_id in tqdm(range(0, len(sentences), batch_size), desc='Extract embeddings'):
                    sentence_batch = sentences[sentence_id:sentence_id+batch_size]

                    inputs = self.tokenizer(sentence_batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs_on_device, return_dict=True)

                    embeddings = outputs.last_hidden_state[:, 0]
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                    embeddings_collection.append(embeddings.cpu())
                embeddings = torch.cat(embeddings_collection, dim=0)
            embeddings = embeddings.numpy()
            return embeddings
        
        def compute(self, predictions: list, references: list):
            assert len(predictions) == len(references), "the length of predictions must be equal to the length of references"
            predictions = [' '.join(jieba.cut(pre)) for pre in predictions]
            references = [' '.join(jieba.cut(ref)) for ref in references]
            hypo_embed = self._encode(predictions)
            ref_embed = self._encode(references)
            cos_sim = np.sum(hypo_embed*ref_embed, axis=1)
            return {"cosine similarity": cos_sim.tolist(), "average": np.average(cos_sim).tolist(), "variance": np.var(cos_sim).tolist()}
        

    def __init__(self, device="cuda:0") -> None:
        self.rouge = Rouge()
        self.bleu = evaluate.load("bleu")
        self.bertscore = score
        # self.embed_ave = self.EmbedAve(model_path="/mnt/pvc-data.common/ChenZikang/huggingface/maidalun1020/bce-embedding-base_v1", device=device)
        print("Loaded evaluation matrix: rouge, bleu, bertscore, embedding average")

    def compute(self, predictions, references):
        result = {}

        def tokenize(sentence):
            return ' '.join(jieba.cut(sentence))
        tokenized_references = [tokenize(ref) for ref in references]
        tokenized_predictions = [tokenize(pred) for pred in predictions]
        
        result["rouge"] = self.rouge.get_scores(tokenized_predictions, tokenized_references, avg=True)

        bleu_score = []
        for i in range(len(predictions)):
            bleu_score.append(round(sentence_bleu([list(references[i])], list(predictions[i]), smoothing_function=SmoothingFunction().method3) * 100, 4))
        result["bleu"] = sum(bleu_score)/len(bleu_score)

        p, r, f1 = self.bertscore(predictions, references, lang='zh', verbose=True)
        # result["bertscore"] = self.bertscore.compute(predictions=predictions, references=references, model_type="bert-base-uncased")
        result['bertscore'] =  torch.mean(f1, dim=-1).item()
        # {
        #     'p-average': torch.mean(p, dim=-1).item(),
        #     'r-average': torch.mean(r, dim=-1).item(),
        #     'f1-average': torch.mean(f1, dim=-1).item(),
        #     'p': p.tolist(),
        #     'r': r.tolist(),
        #     'f1': f1.tolist()
        # }

        # result["embedding average"] = self.embed_ave.compute(predictions=predictions, references=references)
        return result
    
class EvalMatrixChoice(EvalMatrix):
    def _text_2_list(self, text):
        upper_letters = [char for char in text if char.isupper()]

        return set(upper_letters)

    def _cal_acc(self, pred, ref):

        pred = self._text_2_list(pred)
        ref = self._text_2_list(ref)
        
        if pred == ref:
            return 1
        elif pred.issubset(ref):
            return 0.5
        else:
            return 0

    def compute(self, predictions, references):
        ref_len = len(references)

        if ref_len > 0:
            acc_count = 0
            for i in range(ref_len):
                try:
                    acc_count += self._cal_acc(predictions[i], references[i])
                except:
                    print(predictions[i], references[i])
            acc = acc_count/ref_len
        else:
            acc = 0

        return {
            "accuracy": acc
        }