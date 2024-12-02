
import re
import json
import os
import argparse

def extract_by_keywords(text, start, end):
    match_text = re.search(rf"{start}(.*?)(?:{end}|$)", text, re.DOTALL)
    if match_text:
        return match_text.group(1).strip()
    else:
        return ""

def gen_json(sample):
    description = extract_by_keywords(sample, "Description", "\n\s*\n")
    consult = extract_by_keywords(sample, "希望获得的帮助： ", "患病多久：")
    summary = extract_by_keywords(sample, "病情摘要及初步印象：", "总结建议：")
    advice = extract_by_keywords(sample, "总结建议：", "\n\s*\n")

    matches_dialogue = re.finditer(r'Dialogue(.*?)(?:\n\s*\n|$)', sample, re.DOTALL)
    dialogues_list = []
    if consult != "":
        dialogues_list.append({"speaker": "病人", "speech": consult})
    for match_dialogue in matches_dialogue:
        dialogue_text = match_dialogue.group(1).strip()
        dialogues = re.findall(r'(医生|病人)：([\s\S]*?)(?=医生|病人|$)', dialogue_text)
        dialogues_list.extend([{"speaker": speaker.strip(), "speech": speech.strip()} for speaker, speech in dialogues])

    return {
        "description": description,
        "dialogue": dialogues_list,
        "summary": summary,
        "advice": advice
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    
    args = parser.parse_args()

    current_directory = os.getcwd()
    files = os.listdir(args.dataset_dir)
    txt_files = [file for file in files if file.endswith(".txt")]

    corpus_json = []
    for txt_file in txt_files:
        file_path = os.path.join(current_directory, txt_file)
        with open(file_path, 'r') as file:
            content = file.read()
            corpus_splited = content.split("id")
            for sample in corpus_splited:
                if "乳腺癌" in sample or "乳腺炎" in sample or "乳头" in sample or "导管原位癌" in sample:
                    corpus_json.append(gen_json(sample))

    for c in corpus_json:
        dialogue = []
        temp = {
            "speaker": "",
            "speech": ""
        }
        for d in c["dialogue"]:
            if temp["speaker"] == d["speaker"]:
                temp["speech"] += d["speech"]
            else:
                if temp["speaker"] != "":
                    dialogue.append(temp)
                temp = d
        dialogue.append(temp)
        c["dialogue"] = dialogue

    with open(args.output_path, "w", encoding="utf-8") as file:
        json.dump(corpus_json, file, ensure_ascii=False, indent=2)