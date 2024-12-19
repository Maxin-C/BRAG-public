# Breast-CRAG

Github repository for article "Breast-CRAG: A Breast Cancer Large Language Model Leveraging Retrieval-Augmented Generation"

## Directory description

Code: We have open-sourced all the code used in this experiment in the code folder. Please note that some code requires downloading corresponding models or using specific API keys, so please carefully verify before use.

Dataset: Datasets other than Huatuo-BC have been open-sourced in the dataset folder. Huatuo-BC is derived from Huatuo-26M, which only provides original webpage links but not the raw data. To avoid copyright infringement, this study also only provides data links. Users can obtain the required data through certain methods.

Knowledge Base (KB): Considering copyright issues, we have created a JSON file in the kb folder to display the names of the data and literature used in the experiment. Based on the code in the code/retriever/kb_preprocess folder, you can process PDF files in a full pipeline if you can obtain the PDF files.

Model Dictionary: In the model_dict folder, we have shared links to the trained models, which users can obtain from Google Drive. The filter subfolder stores the QLoRA adapter files, the generator subfolder stores the models exported after combining with LoRA adapters (hence very large), and the retriever subfolder stores the fine-tuning results of the 137M model.

## Evluation Result
Our model has demonstrated performance on par with or exceeding gpt-4o-2024-08-06 across four breast cancer question-answer datasets and two breast cancer exam datasets. Detailed results are presented below:

Table 1. Evaluation Result on Breast Cancer Dialogue Dataset
| Evalset        | Model                  | Rouge-1 | Rouge-2 | Rouge-L | Bleu  | Bert-score  |
|----------------|------------------------|---------|---------|---------|-------|-------------|
| Huatuo-BC      | huatuogpt-2            | 0.056   | 0.006   | 0.041   | 0.682 | 0.508       |
|                | llama3-8b-chinese-chat | 0.154   | 0.019   | 0.100   | 1.864 | 0.600       |
|                | qwen2.5-7b-instruct    | 0.144   | 0.021   | 0.071   | 1.568 | 0.612       |
|                | gpt-4-o                | 0.196   | 0.033   | 0.123   | 3.509 | 0.632       |
|                | BRAG                   | 0.231   | 0.059   | 0.193   | 6.355 | 0.653       |
| MedDialogue-BC | huatuogpt-2            | 0.076   | 0.015   | 0.047   | 1.232 | 0.512       |
|                | llama3-8b-chinese-chat | 0.201   | 0.035   | 0.109   | 2.582 | 0.627       |
|                | qwen2.5-7b-instruct    | 0.113   | 0.024   | 0.055   | 1.657 | 0.541       |
|                | gpt-4-o                | 0.261   | 0.065   | 0.149   | 5.395 | 0.667       |
| cMedQA-BC      | huatuogpt-2            | 0.055 | 0.006 | 0.040 | 0.620 | 0.505  |
|                | llama3-8b-chinese-chat | 0.146 | 0.017 | 0.096 | 1.600 | 0.592  |
|                | qwen2.5-7b-instruct    | 0.152 | 0.021 | 0.091 | 1.913 | 0.608  |
|                | gpt-4-o                | 0.194 | 0.032 | 0.121 | 3.385 | 0.630  |
|                | BRAG                   | 0.214 | 0.044 | 0.179 | 4.349 | 0.647  |
| webMedQA-BC    | huatuogpt-2            | 0.059 | 0.007 | 0.043 | 0.842 | 0.512  |
|                | llama3-8b-chinese-chat | 0.146 | 0.017 | 0.098 | 2.026 | 0.591  |
|                | qwen2.5-7b-instruct    | 0.158 | 0.022 | 0.094 | 2.412 | 0.606  |
|                | gpt-4-o                | 0.197 | 0.033 | 0.121 | 4.037 | 0.625  |
|                | BRAG                   | 0.174 | 0.034 | 0.144 | 2.909 | 0.626  |


Table 2. Evaluation Result on Breast Cancer Exam Dataset

| Models                 | Exam-BC (simple set) |        |      | Exam-BC (hard set) |        |      | USMLE-BC  |
|------------------------|----------------------|--------|------|--------------------|--------|------|-----------|
|                        | Single               | Multi. | Ave. | Single             | Multi. | Ave. | Single    |
| HuatuoGPT2-7B          | 0.36                 | 0.36   | 0.36 | 0.26               | 0.24   | 0.25 | 0.48      |
| Llama3-8B-Chinese-Chat | 0.22                 | 0.37   | 0.25 | 0.24               | 0.22   | 0.23 | 0.49      |
| Qwen2.5-7B-Instruct    | 0.35                 | 0.44   | 0.37 | 0.3                | 0.49   | 0.33 | 0.41      |
| GPT-4-o                | 0.73                 | 0.54   | 0.69 | 0.68               | 0.45   | 0.64 | 0.81      |
| Breast-CRAG            | 0.65                 | 0.54   | 0.63 | 0.56               | 0.37   | 0.52 | 0.81      |

# Paper
Our pre-print paper can be accessed at http://ssrn.com/abstract=5052341
