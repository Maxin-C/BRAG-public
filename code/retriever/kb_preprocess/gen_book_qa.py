
from openai import OpenAI
import json
from tqdm import tqdm
import re
import json

client = OpenAI(
    api_key="",
)

def gen_qa(text, model_name, qa_num):
    system_message = f"请根据用户输入的文本信息，基于该信息生成{qa_num}个不同的问答对。必须满足以下要求:"
    system_message += '''
            1. 问答内容完全基于输入文本，并且回答能够从文本中找到对应的依据。
            2. 问答内容不需要额外信息。
            3. 最终生成的问答对尽可能提及文本中包含的所有信息。
            4. 每条问答对字符数不少于400。问题字符数不少于80。
            5. 每条问答对提出的问题彼此独立，无概念交集。
            6. 不要使用省略号代替文本内容。
            7. 问答对输出格式为json组成的list，具体格式为{'question':'', 'answer': ''}。'''
    
    user_message = f'''文本信息为：
    {text}
请根据文本信息按照格式生成问答对'''
    
    for _ in range(5):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                model=model_name,
                temperature=0.3
            )
        except:
            continue
        else:
            return chat_completion.choices[0].message.content
    return ""

if __name__ == "__main__":
    with open("ocr.json", "r") as file:
        dataset = json.load(file)
    for i in tqdm(range(len(dataset)), desc="gpt qa generation"):
        data = dataset[i]
        dataset[i]['qa'] = gen_qa(data['ocr'], "gpt-4o", int(len(data['ocr'])/200))


    with open("book_qa.json", "w", encoding="utf-8") as file:
        json.dump(dataset, file, ensure_ascii=False)