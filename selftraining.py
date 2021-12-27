import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_metric,load_dataset,Dataset

import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding,RobertaForSequenceClassification,AdamW,get_scheduler,TrainingArguments,Trainer


import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split,StratifiedKFold
from tqdm.auto import tqdm, trange

import csv
import gc

model_checkpoint = 'roberta-base'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
transformers.logging.set_verbosity(transformers.logging.ERROR)

BATCH_SIZE = 32

def compute_metrics(testing_dataloader):
    metric = load_metric("f1")

    model.eval()
    for batch in testing_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        
    return metric.compute(average='micro')

#load data
data = pd.read_csv('./data/BABE/final_labels_SG2.csv',sep=';')
data = data[['text','label_bias']]
data = data[data['label_bias']!='No agreement']
mapping = {'Non-biased':0, 'Biased':1}
data.replace({'label_bias':mapping},inplace=True)
data.rename(columns={'text':'sentence','label_bias':'label'},inplace=True)

basil_data = pd.read_csv('./data/basil.csv',sep=',')
basil_data_processed = basil_data[basil_data['bias'] == 1][['sentence','bias']]
bd_labels = basil_data_processed['bias']
bd_sentences = basil_data_processed[['sentence']]


#training
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint);
model = RobertaForSequenceClassification.from_pretrained(model_checkpoint);
model.to(device);


training_args = TrainingArguments(
    output_dir='../',
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    warmup_steps=0,  
    logging_steps=50,
    disable_tqdm = True,
    save_total_limit=2,
    weight_decay=5e-5)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenize = lambda data : tokenizer(data['sentence'], truncation=True)

tokenized_data = Dataset.from_pandas(data)
tokenized_data = tokenized_data.map(tokenize,batched=True)
tokenized_data = tokenized_data.remove_columns(['sentence','__index_level_0__'])
tokenized_data.set_format("torch")

tokenized_bd = Dataset.from_pandas(bd_sentences)
tokenized_bd = tokenized_bd.map(tokenize,batched=True)
tokenized_bd = tokenized_bd.remove_columns(['sentence','__index_level_0__'])
#tokenized_bd.set_format("torch")



f1_scores = []
k = 100

for train_index, val_index in skfold.split(data[['sentence']],data[['label']]):

    #split for this whole selftraining iteration
    token_train = Dataset.from_dict(tokenized_data[train_index])
    token_valid = Dataset.from_dict(tokenized_data[val_index])
    token_train.set_format("torch")
    token_valid.set_format("torch")
    
    tokenized_bd = Dataset.from_pandas(bd_sentences)
    tokenized_bd = tokenized_bd.map(tokenize,batched=True)
    tokenized_bd = tokenized_bd.remove_columns(['sentence','__index_level_0__'])
    
    #self training
    while True:
        #print("Iteration :",iterations)
        print("Fitting on ", len(token_train), " data")
        #initial training
        model = RobertaForSequenceClassification.from_pretrained(model_checkpoint);
        trainer = Trainer(model,training_args,train_dataset=token_train,data_collator=data_collator,
                          tokenizer=tokenizer);
        trainer.train()
        
        #making predictions on unlabelled dataset
        unlabelled_dataloader = DataLoader(tokenized_bd, batch_size=BATCH_SIZE, collate_fn=data_collator)

        logits = torch.Tensor().to(device)

        model.eval()
        for batch in unlabelled_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = torch.cat((logits,F.softmax(outputs.logits)))

        
        #stop when there is not enough of resources
        if len(logits[:,0]) < k or len(logits[:,1]) < k:
            break
            
        #indices of the highest probability ranked predictions
        unbiased_topk_indices = torch.topk(logits[:,0],k)[1]
        biased_topk_indices = torch.topk(logits[:,1],k)[1]


        #insert them into training set
        for index in biased_topk_indices:
            item = tokenized_bd[index.item()]
            item['label'] = 1

            token_train = token_train.add_item(item)

        for index in unbiased_topk_indices:
            item = tokenized_bd[index.item()]
            item['label'] = 0

            token_train = token_train.add_item(item)

        #remove them from unlabelled
        all_indices = np.arange(0,len(tokenized_bd))
        all_to_drop = torch.cat((biased_topk_indices,unbiased_topk_indices)).cpu()
        remaining = np.delete(all_indices,all_to_drop)

        tokenized_bd = Dataset.from_dict(tokenized_bd[remaining])
    
    #evaluation
    eval_dataloader = DataLoader(token_valid, batch_size=BATCH_SIZE, collate_fn=data_collator)
    f1_scores.append(compute_metrics(eval_dataloader)['f1'])
    print("\n\nFinished with F1: ", compute_metrics(eval_dataloader)['f1'])


torch.save(model.state_dict(),'/home/horyctom/experimental-tomas-horych/self_training_babe.pth')

with open('/home/horyctom/experimental-tomas-horych/scores.txt','w') as f:
	for num in f1_scores:
		f.write(str(num) + '\n')

