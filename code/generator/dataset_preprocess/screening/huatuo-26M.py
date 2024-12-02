import json

def append_data(filename):
    with open(filename, 'r') as file:
        dataset = file.readlines()
    corpus = []
    for data in dataset:
        data = json.loads(data)
        if "乳腺癌" in data['question'] or "乳腺炎" in data['question'] or "乳头" in data['question'] or "导管原位癌" in data['question']:
            corpus.append(data)

    return corpus

corpus = append_data('dataset.jsonl')

with open('huatuo-BC.json', 'w', encoding='utf-8') as file:
    json.dump(corpus, file, ensure_ascii=False)