
from openai import OpenAI
from pydantic import BaseModel
import json
from tqdm import tqdm
import argparse

def gen_prompt(data):
    prompt = f"请将下面内容总结为医患的单轮问答，必须包括所有信息，医生和患者只说一次话。\n背景描述：{data['description']}"

    for d in data['dialogue']:
        prompt += f"\n{d['speaker']}：{d['speech']}"
    
    message = [{'role': 'user', 'content': prompt}]

    return message

class Response(BaseModel):
    background: str
    patient: str
    doctor: str

def call_model(client, messages):
    completion = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            response_format=Response
        )
    return completion.choices[0].message.parsed

def get_response(client, data):
    try:
        response = call_model(client, gen_prompt(data))
        response = response.model_dump_json()
    except Exception as e:
        return None
    else:
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    
    args = parser.parse_args()

    client = OpenAI(api_key = args.api_key)

    dataset = json.load(open(args.dataset_path, 'r'))

    failed_dataset = []
    summarized_dataset = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        response = get_response(client, data)
        if response == None:
            failed_dataset.append(data)
        else:
            summarized_dataset.append(response)

    json.dump(summarized_dataset, open(f"{args.output_path}/medDialog-BC.json", 'w'), ensure_ascii=False)
    json.dump(failed_dataset, open(f"{args.output_path}/failed_dataset.json", 'w'), ensure_ascii=False)