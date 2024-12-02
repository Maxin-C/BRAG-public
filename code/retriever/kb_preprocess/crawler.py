
import requests
from bs4 import BeautifulSoup
import random
import time
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")
from retrying import retry
import argparse

@retry(stop_max_attempt_number=1, wait_fixed=500)
def proxy_request(url, proxies):
    headers= {
        'User-Agent': 'PostmanRuntime/7.37.0',
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate, br'
    }
    try:
        response = requests.get(url, proxies=proxies, headers=headers, verify=False, timeout=0.5)
    except:
        return None

    # 获取页面内容
    if response.status_code == 200:
        return response.text
    else:
        return None

def get_qa(urls_proxies):

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(proxy_request, urls_proxies)

    questions = []
    answers = []
    for html_content in results:
        if html_content != None:
            soup = BeautifulSoup(html_content, 'html.parser')

            target_div = soup.find('div', class_='content-answer-con')
            if target_div:
                questions.append(target_div.get_text())
            else:
                questions.append('')

            div_con = soup.select_one('body > div.wrapper > div > div.content-left-area > div > div.answer > div > ul > li > div.con')
            if div_con:
                paragraphs = div_con.find_all('p')
                if paragraphs:
                    answers.append(' '.join([p.get_text() for p in paragraphs]) )
                else:
                    answers.append('')
            else:
                answers.append('')
        else:
            questions.append('')
            answers.append('')
    
    return questions, answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--tunnel", type=str, required=True, help="get tunnel on https://www.kuaidaili.com/")
    parser.add_argument("--username", type=str, required=True, help="get username on https://www.kuaidaili.com/")
    parser.add_argument("--password", type=str, required=True, help="get password on https://www.kuaidaili.com/")
    parser.add_argument("--batch_size", type=str, required=True)

    args = parser.parse_args()

    proxies = {
        "http": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": args.username, "pwd": args.password, "proxy": args.tunnel},
        "https": "http://%(user)s:%(pwd)s@%(proxy)s/" % {"user": args.username, "pwd": args.password, "proxy": args.tunnel}
    }

    dataset = json.load(open(args.dataset_path, 'r'))
    corpus = {'train': [], 'eval': [],'test': []}
    failed_data = []

    for _, value in enumerate(corpus):
        for i in tqdm(range(0, len(dataset[value]), args.batch_size), desc=value):
            datas = dataset[value][i: i+args.batch_size]
            urls = [data['answers'][0] for data in datas]
            urls_proxies = zip(urls, [proxies]*len(urls))
            questions, answers = get_qa(urls)
            for j in range(len(datas)):
                if questions[j] != '' or answers[j] != '':
                    corpus[value].append({
                        'title': datas[j]['questions'][0],
                        'question': questions[j],
                        'answer': answers[j],
                        'url': urls[j]
                    })
                else:
                    failed_data.append(datas[j])


    with open(f'{args.output_path}/dataset.json', 'w', encoding='utf-8') as file:
        json.dump(corpus, file, ensure_ascii=False)

    with open(f'{args.output_path}/failed_dataset.json', 'w', encoding='utf-8') as file:
        json.dump(failed_data, file, ensure_ascii=False)
