import os
import re
import json
import argparse

def split_chunk(text, sentence_pattern, max_len=450):

    chunks = []
    start = 0
    for match in re.finditer(sentence_pattern, text):
        end = match.end()
        if end - start > max_len:
            chunks.append(text[start:end])
            start = end
        elif end == len(text):
            chunks.append(text[start:])
    return chunks

def text2chunk(file_path, max_len):
    with open(file_path, 'r') as file:
        text = file.read()
    pattern = r"!\[(\d+)_image_(\d+)\.png]\(\1_image_\2\.png\)"
    text = re.sub(pattern, "", text)

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    pattern = r"#+"
    chunks_temp = re.split(pattern, text, flags=re.MULTILINE)

    chunks = []
    for chunk in chunks_temp:
        if chunk.strip:
            language_pattern = r'[\u4e00-\u9fa5]'
            if bool(re.search(language_pattern, file_path)):
                chunks.extend(split_chunk(chunk, sentence_pattern = r"。|;|；", max_len=max_len))
            else:
                chunks.extend(split_chunk(chunk, sentence_pattern = r"\.|;", max_len=max_len))
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markdown_path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='')
    parser.add_argument("--chunk_max_len", type=str, default=100)

    args = parser.parse_args()

    chunks = []
    code_book = {}
    index = 0

    for file_name in os.listdir(args.markdown_path):
        file_path = f"ocr_md/{file_name}"
        split_chunk = text2chunk(file_path, args.chunks_max_len)
        code_book.update({file_name:[index, index+len(split_chunk)]})
        chunks.extend(split_chunk)
        index += len(split_chunk)
    
    with open(f"{args.output_path}/chunks_{args.chunks_max_len}.json", 'w', encoding='utf-8') as file:
        json.dump(chunks, file, ensure_ascii=False)
    with open(f"{args.output_path}/code_book_{args.chunks_max_len}.json", 'w', encoding='utf-8') as file:
        json.dump(code_book, file, ensure_ascii=False)