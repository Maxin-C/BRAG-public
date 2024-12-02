
import json
import os
import pandas

questions = pandas.read_csv('questions.csv')
answers = pandas.read_csv('answers.csv')

corpus = []
for index, row in questions.iterrows():
    if "乳腺癌" in row['content'] or "乳腺炎" in row['content'] or "乳头" in row['content'] or "导管原位癌" in row['content']:
        que_id = row['que_id']
        try:
            corpus.append({
                "question": row['content'],
                "answer": answers[answers['que_id']==que_id]['content'].values[0],
                "big_cate": row['big_cate'],
                "small_cate": row['small_cate']
            })
        except:
            print(que_id)

with open('cMedQABC.json', 'w', encoding='utf-8') as file:
    json.dump({'all':corpus}, file, ensure_ascii=False)