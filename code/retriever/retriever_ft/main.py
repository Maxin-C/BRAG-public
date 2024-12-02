
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, AdamW
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
from datetime import datetime
import argparse

class TextDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()

        self.query = []
        self.text_pos = []
        self.text_neg = []

        for i in tqdm(range(len(dataset)), desc="Load Dataset"):
            data = dataset[i]
            self.query.append(data['qa']['question'])
            self.text_pos.append(data['chunk_positive'])
            self.text_neg.append(data['chunk_negative'])
    
    def __len__(self):
        assert len(self.query)  == len(self.text_pos) == len(self.text_neg)
        return len(self.query)
    
    def __getitem__(self, index):
        return self.query[index], self.text_pos[index], self.text_neg[index]


class TextEncoder(nn.Module):
    def __init__(self, text_model) -> None:
        super(TextEncoder, self).__init__()
        self.text_model = text_model
        self.output_dim = text_model.encoder.layers[-1].norm2.normalized_shape[0]
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, text):
        model_output = self.text_model(**text)
        embeddings = self.mean_pooling(model_output, text['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)


def cosine_similarity_dist_loss(a, b, c, margin=0.1):
    cos_sim_ab = F.cosine_similarity(a, b, dim=-1)
    cos_sim_ac = F.cosine_similarity(a, c, dim=-1)
    loss = torch.clamp(margin - (cos_sim_ab - cos_sim_ac), min=0.0)
    return loss.mean()


def cal_acc(query, text_pos, text_neg, margin=0):
    pred_text_pos = F.cosine_similarity(query, text_pos, dim=1)
    pred_text_neg = F.cosine_similarity(query, text_neg, dim=1)

    diff = pred_text_pos - pred_text_neg > margin
    
    acc = torch.sum(diff).item()
    return acc


def inference(model, query, text_pos, text_neg, tokenizer, max_seq_length, device, is_train=False):
    query = tokenizer(
        list(query),
        max_length=max_seq_length,
        truncation=True,
        padding=True,
        return_tensors='pt').to(device)
    text_pos = tokenizer(
        list(text_pos),
        max_length=max_seq_length,
        truncation=True,
        padding=True,
        return_tensors='pt').to(device)
    text_neg = tokenizer(
        list(text_neg),
        max_length=max_seq_length,
        truncation=True,
        padding=True,
        return_tensors='pt').to(device)

    if is_train:
        query = model(query)
        text_pos = model(text_pos)
        text_neg = model(text_neg)
    else:
        with torch.no_grad():
            query = model(query)
            text_pos = model(text_pos)
            text_neg = model(text_neg)
    
    loss_1 = cosine_similarity_dist_loss(query, text_pos, text_neg, margin=0.3)

    loss = loss_1

    acc = cal_acc(query, text_pos, text_neg, margin=0.3)

    return loss, acc


def train(model, tokenizer, max_seq_length, patience, train_dataset, eval_dataset, test_dataset, epochs, batch_size, optimizer, schedualer, checkpoint, start_epoch, device, save_folder_path):
    assert checkpoint > 0

    train_dataloader, eval_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(eval_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    train_loss_list = []
    eval_loss_list = []
    train_acc_list = []
    eval_acc_list = []

    early_stop_counter = 0
    best_eval_acc = 0.0

    model.to(device)

    for epoch in range(start_epoch, epochs):

        total_loss_train = 0
        total_acc_train = 0
        model.train()
        for train_query, train_text_pos, train_text_neg in tqdm(train_dataloader):
            loss, acc = inference(model, train_query, train_text_pos, train_text_neg, tokenizer, max_seq_length, device, is_train=True)

            total_loss_train += loss
            total_acc_train += acc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(
            f'Epochs: {epoch + 1}: \
            | Train Loss: {total_loss_train / len(train_dataset): .3f} \
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f} '
        )
        train_loss_list.append(total_loss_train)
        train_acc_list.append(total_acc_train)

        total_loss_eval = 0
        total_acc_eval = 0
        model.eval()
        for eval_query, eval_text_pos, eval_text_neg in tqdm(eval_dataloader):
            loss, acc = inference(model, eval_query, eval_text_pos, eval_text_neg, tokenizer, max_seq_length, device, is_train=False)

            total_loss_eval += loss
            total_acc_eval += acc

        print(
            f'Epochs: {epoch + 1}: \
            | Eval Loss: {total_loss_eval / len(eval_dataset): .3f} \
            | Eval Accuracy: {total_acc_eval / len(eval_dataset): .3f} '
        )
        eval_loss_list.append(total_loss_eval)
        eval_acc_list.append(total_acc_eval)

        if total_acc_eval / len(eval_dataset) + 0.1 > best_eval_acc:
            if total_acc_eval / len(eval_dataset) > best_eval_acc:
                best_eval_acc = total_acc_eval / len(eval_dataset)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter > patience:
            break

        schedualer.step()

        if (epoch+1) % checkpoint == 0:
            torch.save(model.state_dict(), f"{save_folder_path}/cp_{epoch+1}.pt")
        

    total_loss_test = 0
    total_acc_test = 0
    model.eval()
    for test_query, test_text_pos, test_text_neg in tqdm(test_dataloader):
        loss, acc = inference(model, test_query, test_text_pos, test_text_neg, tokenizer, max_seq_length, device, is_train=False)

        total_loss_test += loss
        total_acc_test += acc
    print(
            f'| Test Loss: {total_loss_test / len(test_dataset): .3f} \
            | Test Accuracy: {total_acc_test / len(test_dataset): .3f} '
        )
    
    train_loss_list = torch.tensor(train_loss_list,device='cpu')
    train_loss_list = train_loss_list / train_dataset.__len__()
    train_acc_list = torch.tensor(train_acc_list,device='cpu')
    train_acc_list = train_acc_list / train_dataset.__len__()

    eval_loss_list = torch.tensor(eval_loss_list,device='cpu')
    eval_loss_list = eval_loss_list / eval_dataset.__len__()
    eval_acc_list = torch.tensor(eval_acc_list,device='cpu')
    eval_acc_list = eval_acc_list / eval_dataset.__len__()

    total_loss_test = total_loss_test / len(test_dataset)
    total_acc_test = total_acc_test / len(test_dataset)
    
    return epoch, train_loss_list, train_acc_list, eval_loss_list, eval_acc_list, total_loss_test, total_acc_test, model.state_dict()

if __name__=="__name__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default='model_config.json')

    args = parser.parse_args()

    model_config = json.load(open(args.model_config_path))

    with open(model_config['dataset_path'], "r") as file:
        dataset = json.load(file)
    train_dataset = TextDataset(dataset["train"])
    eval_dataset = TextDataset(dataset["eval"])
    test_dataset = TextDataset(dataset["test"])


    torch.cuda.empty_cache()

    text_model = AutoModel.from_pretrained(model_config['local_model_path']+model_config['model']['text_model'], trust_remote_code=True, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_config['local_model_path']+model_config['model']['text_tokenizer'])
    model = TextEncoder(text_model=text_model)

    if model_config['optimizer'] == 'SGD':
        optimizer = SGD(model.parameters(), lr=model_config['learning_rate'], weight_decay=model_config['weight_decay'])
    elif model_config['optimizer'] == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=model_config['learning_rate'], weight_decay=model_config['weight_decay'])
    else:
        Warning("Unimported optimizer. Please check your imported package.")
        optimizer = None

    schedualer = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=model_config['schedualer']['milestones'], gamma=model_config['schedualer']['gamma'])

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_folder_path = model_config['save_model_dict_path'] + timestamp

    epochs = model_config['epochs']

    start_epoch = model_config['start_epoch']
    if start_epoch > 0 and model_config['start_epoch_timestamp']!="":
        model.load_state_dict(torch.load(model_config['local_model_path']+model_config['start_epoch_timestamp'], map_location=model_config['device'])) # type: ignore
        schedualer.last_epoch = start_epoch - 1
    else:
        model.to(model_config['device'])

    if model_config['load_model_dict_path'] != "" and start_epoch == 0:
        model.load_state_dict(torch.load(f"{model_config['load_model_dict_path']}model_state.pt", map_location=model_config['device']))


    train_epochs, train_loss_list, train_acc_list, eval_loss_list, eval_acc_list, test_loss, test_acc, model_state = train(
        model = model,
        tokenizer = tokenizer,
        max_seq_length = model_config['max_seq_length'],
        patience = model_config['patience'],
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        test_dataset = test_dataset,
        epochs = epochs,
        batch_size = model_config['batch_size'],
        optimizer = optimizer,
        schedualer = schedualer,
        checkpoint = model_config['checkpoint'],
        start_epoch = start_epoch,
        device=model_config['device'],
        save_folder_path=save_folder_path)


    torch.save(model_state, f"{save_folder_path}/model_state.pt")

    for filename in os.listdir(save_folder_path):
        if filename.startswith('cp') and filename.endswith('.pt'):
            file_path = os.path.join(save_folder_path, filename)
            os.remove(file_path)

    model_config['train_epochs'] = train_epochs
    with open(f'{save_folder_path}/model_config.json', 'w') as file:
        json.dump(model_config, file)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,20))

    axes[0,0].plot(np.arange(1, train_epochs-start_epoch+2, 1), train_loss_list)
    axes[0,0].set_title('Loss-Epoch Curve(Train)')

    axes[0,1].plot(np.arange(1, train_epochs-start_epoch+2, 1), train_acc_list)
    axes[0,1].set_title('Acc-Epoch Curve(Train)')

    axes[1,0].plot(np.arange(1, train_epochs-start_epoch+2, 1), eval_loss_list)
    axes[1,0].set_title('Loss-Epoch Curve(Eval)')

    axes[1,1].plot(np.arange(1, train_epochs-start_epoch+2, 1), eval_acc_list)
    axes[1,1].set_title('Acc-Epoch Curve(Eval)')

    plt.savefig(f'{save_folder_path}/result.png')
    plt.show()

    train_result = {
        "train_loss" : train_loss_list.tolist(),
        "train_acc" : train_acc_list.tolist(),
        "eval_loss" : eval_loss_list.tolist(),
        "eval_acc" : eval_acc_list.tolist(),
        "test_loss" : test_loss.item(),
        "test_acc" : test_acc
    }

    with open(f'{save_folder_path}/result.json', 'w') as file:
        json.dump(train_result, file)