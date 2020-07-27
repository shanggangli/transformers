#-*- codeing=utf-8 -*-
#@time: 2020/7/21 13:42
#@Author: Shang-gang Lee
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
import pandas as pd
import numpy as np
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data=pd.read_csv(r'C:\untitled\kaggle\tweet-sentiment-extraction\data\train.csv')
test_data=pd.read_csv(r'C:\untitled\kaggle\tweet-sentiment-extraction\data\test.csv')

train_data.dropna(axis=0,how='any',inplace=True) # drop missing values

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

def processing(text,select_text,tokenizer,max_len):
    input_ids=[]
    attention_mask=[]
    token_type_ids=[]
    start_idx=[]
    end_idx=[]
    for i in range(len(text)):
        output=tokenizer.encode_plus(text[i],add_special_tokens=True,return_tensors='pt',max_length=max_len,truncation=True,pad_to_max_length=True)
        input_ids.append(output['input_ids'])
        attention_mask.append(output['attention_mask'])
        token_type_ids.append(output['token_type_ids'])

        select_text_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(select_text[i]))
        for j,x in enumerate(input_ids[-1]):
            for i in x:

                if i > 0:
                    start_idx.append([i for i, k in enumerate(x) if select_text_ids[0] in k])
                    end_idx.append([i for i, k in enumerate(x) if select_text_ids[-1] in k])
                    #print(start_idx[j])
                    try:
                        if start_idx[j] != None and end_idx[j] != None:
                            break

                    except:continue

    for i in range(len(start_idx)):
        if len(start_idx[i]) > 1:
            start_idx[i] = [start_idx[i][0]]

        elif len(end_idx[i]) > 1:
            end_idx[i] = [end_idx[i][0]]

    return input_ids,attention_mask,token_type_ids,start_idx,end_idx

text=train_data.loc[:10,'text'].values
select_text=train_data.loc[:10,'selected_text'].values
max_len=128
input_ids,attention_mask,token_type_ids,start_idx,end_idx=processing(text=text,select_text=select_text,tokenizer=tokenizer,max_len=max_len)
input_ids=[t.numpy() for t in input_ids]
attention_mask=[t.numpy() for t in attention_mask]
token_type_ids=[t.numpy() for t in token_type_ids]

input_ids=torch.tensor(input_ids).to(device,dtype=torch.long)
attention_mask=torch.tensor(attention_mask).to(device,dtype=torch.float)
token_type_ids=torch.tensor(token_type_ids).to(device,dtype=torch.long)
start_idx=torch.tensor(np.array(start_idx)).to(device,dtype=torch.long)
end_idx=torch.tensor(np.array(end_idx)).to(device,dtype=torch.long)

# print(input_ids.shape)
# print(attention_mask.shape)
# print(token_type_ids.shape)
# print(strat_idx.shape)
# print(end_idx.shape)

input_ids=input_ids.view(-1,max_len)
attention_mask=attention_mask.view(-1,max_len)
token_type_ids=token_type_ids.view(-1,max_len)

model=BertForQuestionAnswering.from_pretrained('bert-base-uncased')
loss,start_scores,end_scores=model(input_ids,attention_mask,token_type_ids,
                                   start_positions=start_idx,end_positions=end_idx)

print(loss)
print(start_scores.shape)
print(end_scores.shape)

