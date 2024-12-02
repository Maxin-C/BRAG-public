import json

def append_data(file_name):
    corpus = []
    with open(file_name, 'r') as file:
        dataset = file.readlines()
    for data in dataset:
        data = data.split('\t')
        if "乳腺癌" in data[3] or "乳腺炎" in data[3] or "乳头" in data[3] or "导管原位癌" in data[3]:
            corpus.append(data)
    return corpus

corpus = {
    'train': append_data('medQA.train.txt'),
    'eval': append_data('medQA.valid.txt'),
    'test': append_data('medQA.test.txt')
}

with open('medQABC.json', 'w', encoding='utf-8') as file:
    json.dump(corpus, file, ensure_ascii=False)