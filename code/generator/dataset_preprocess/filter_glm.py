import json
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
from tqdm import tqdm
    
def load_model(peft_model_path):
    config = PeftConfig.from_pretrained(peft_model_path)
    q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float32)

    base_model = AutoModel.from_pretrained(config.base_model_name_or_path,
                                           quantization_config=q_config,
                                           trust_remote_code=True,
                                           device_map='auto')
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
    return model, tokenizer

def gen_context(data):
    dialogue_text = ""
    for d in data['dialogue']:
        dialogue_text += f"{d['speaker']}：{d['speech']}\n"
    context = f'''
判断对话是否符合以下要求：
1.	与患者对话，明确询问与乳腺癌相关的问题；
2.	对话有一个语义完整的回应；
3.	回答可以回答上述问题；
4.	不包括挂号预约、治疗费用等医疗程序相关内容；
5.	不包括需要检查材料和图像等附加信息的对话。
注意：输出仅限于True或False

语料内容：
医疗背景：{data["description"]}
对话内容：{dialogue_text}
'''
    max_len = 12000
    if len(context)>max_len:
        context = context[:max_len]
    return context

def annote_dataset(model, tokenizer, dataset):
    annotted_dataset = []
    for i in tqdm(range(len(dataset))):
        instruction = gen_context(dataset[i])
        response, history = model.chat(tokenizer=tokenizer, query=instruction)
        if response == "True":
            annotted_dataset.append(dataset[i])
    return annotted_dataset

def main():
    parser = argparse.ArgumentParser(description="Automativally annote medDialogue corpus")
    parser.add_argument("--peft_model_path", type=str, default="output/qlora/medDialogue")
    parser.add_argument("--file_path", type=str, default="dataset/medDialogue/medDialogueBC.json")

    arg = parser.parse_args()
    with open(arg.file_path, "r") as file:
        dataset = json.load(file)

    model, tokenizer = load_model(arg.peft_model_path)

    check_point = int(len(dataset)/20)
    for i in range(19):
        splitted_dataset = dataset[check_point*(i+1):check_point*(i+2)]
        annotted_dataset = annote_dataset(model, tokenizer, splitted_dataset)
        with open(f"dataset/medDialogue/annotted_20_{i+2}.json", "w") as file:
            json.dump(annotted_dataset, file, ensure_ascii=False, indent=2)
        print(f"Annotted dataset in stored in dataset/medDialogue/annotted_{20}_{i+2}.json")

if __name__ == "__main__":
    main()