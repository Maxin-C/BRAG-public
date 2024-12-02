import json
from openai import OpenAI
from tqdm import tqdm
client = OpenAI(api_key = '')

def gen_scene_prompt(medDialogue):
    dialogue_text = ""
    for d in medDialogue['dialogue']:
        dialogue_text += f"{d['speaker']}：{d['speech']}\n"
    prompt = f'''
判断对话是否符合以下要求：
1.	与患者对话，明确询问与乳腺癌相关的问题；
2.	对话有一个语义完整的回应；
3.	回答可以回答上述问题；
4.	不包括挂号预约、治疗费用等医疗程序相关内容；
5.	不包括需要检查材料和图像等附加信息的对话。
注意：输出仅限于True或False

Input: 
医疗背景：
疾病：乳腺纤维瘤 内容：病情描述（发病时间、主要症状、就诊医院等）：患者女，35岁。从25岁起，就确定为乳腺纤维瘤，因为穷，所以没有治疗。这当中也没有什么异常感觉，还生了两个孩子。但最近，乳房有些涨痛。今天做了个彩超，超声提示‘双侧乳腺实性肿物【右乳外侧及内上均见一个类圆形低回声结节，大小分别为，10mmx7mm,10mmx4mm。边界清楚，包膜明显，内部回声欠均匀。左乳内上象限见一个类圆形低回声结节，大小，9mmx6mm，边界清楚，包膜明显，内部回声欠均匀。曾经治疗情况和效果：没有治疗过想得到怎样的帮助:针对目前情况，您能否给一些治疗建议？
对话内容：
医生：建议观察，半年后复查。
Output:
否

Input:
医疗背景：
疾病：乳腺有结快 内容：病情描述（发病时间、主要症状、就诊医院等）：去年6月开始有感觉有个小快过来想看怕变病曾经治疗情况和效果：到嘉兴妇保看过药没吃过说要开刀想得到怎样的帮助:4月30日我来上海看病希望能看到
对话内容：
病人：袁明天我早上4点07分火车嘉善开往上海你医院求诊
医生：你可明日下午一时正到龙华医院乳腺科病房可找到我，我先可以绐你检查一下。如果来晚了我一时半要去开刀了。 袁
病人：好的   那我怎么找你呀   你给个电话号码给我  我正时到袁
医生：我今天没有碰到你。我的手机号，13651833825. 袁
Output:
是

Input:
医疗背景：
{medDialogue['description']}
对话内容：
{dialogue_text}
Output:
'''
    return prompt

def gpt_judger(prompt):
    for _ in range(5):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="gpt-4o-2024-08-06",
                temperature=0.3
            )
        except:
            continue
        else:
            return chat_completion.choices[0].message.content
    return ""

with open("../dataset/medDialogueBC.json", 'r') as file:
    medDialogue = json.load(file)

result = []
for i in tqdm(range(2000), desc="Dataset Pre-Processing", unit="dialogue"):
    scene_prompt = gen_scene_prompt(medDialogue[i])
    response = gpt_judger(scene_prompt) 
    if "True" in response:
        medDialogue[i]["tag"] = False
        result.append(medDialogue[i])
    elif "False" in response:
        medDialogue[i]["tag"] = True
        result.append(medDialogue[i])
    else:
        medDialogue[i]["tag"] = response
        result.append(medDialogue[i])

with open("2000.json", "w", encoding='utf-8') as file:
    json.dump(result, file, ensure_ascii=False)