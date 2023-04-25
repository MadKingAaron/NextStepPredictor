import os
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
from random import sample

import datasets
import itertools


class CaptionDataset(Dataset):
    def __init__(self, tokenizer, caption_df:pd.DataFrame, task_prefix:str, prepend_prefix:bool = True) -> None:
        self.captions_df = caption_df
        self.task_prefix = task_prefix
        self.prepend_prefix = prepend_prefix
        self.tokenizer = tokenizer

        super().__init__()
    
    def __len__(self):
        return len(self.captions_df)
    

    def get_indices(self, index):
        sub_df = self.captions_df.iloc[index,:]
        if self.prepend_prefix:
            inputs = [self.task_prefix + x for x in sub_df['input']]
        else:
            inputs = list(sub_df['input'])

        outputs = list(sub_df['label'])

        return inputs, outputs

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        inputs, outputs = self.get_indices(index)

        encoding = self.tokenizer(inputs, padding="max_length", max_length=512, truncation=True, return_tensors='pt')
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        target_encoding = self.tokenizer(outputs, padding="max_length", max_length=128, truncation=True, return_tensors='pt')
        labels = target_encoding.input_ids

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return input_ids, attention_mask, labels

class IdxDataset(Dataset):
    def __init__(self, len:int) -> None:
        self.len = len
        super().__init__()
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.tensor(index)

class LoaderWrapper():
    def __init__(self, loader:DataLoader, ds:Dataset) -> None:
        self.loader = loader
        self.ds = ds
    
    def __call__(self):
        for indices in self.loader:
            yield self.ds[indices]


def train_val_test_split(df:pd.DataFrame, train_split:float, val_split:float, test_split:float):
    train_df, other_df = train_test_split(df, train_size=train_split)

    val_split_norm = val_split/(val_split + test_split)
    val_df, test_df = train_test_split(other_df, train_size=val_split_norm)

    return train_df, val_df, test_df


def get_individual_loader(tokenizer, task_prefix:str, prepend_prefix:bool, df:pd.DataFrame, batch_size:int = 64) -> LoaderWrapper:
    ds = CaptionDataset(tokenizer, df, task_prefix, prepend_prefix)
    idx_ds = IdxDataset(len(ds))
    dl_idx = DataLoader(dataset=idx_ds, batch_size=64, shuffle=True)
    loader = LoaderWrapper(dl_idx, ds)
    return loader

def get_loaders(tokenizer, task_prefix:str, prepend_prefix:bool, df:pd.DataFrame, train_split:float, train_batch:int, val_split:float, val_batch:int, test_split:float, test_batch:int):
    train_df, val_df, test_df = train_val_test_split(df, train_split, val_split, test_split)

    trainloader = get_individual_loader(tokenizer, task_prefix, prepend_prefix, train_df, train_batch)
    valloader = get_individual_loader(tokenizer, task_prefix, prepend_prefix, val_df, val_batch)
    testloader = get_individual_loader(tokenizer, task_prefix, prepend_prefix, test_df, test_batch)

    return trainloader, valloader, testloader


def tokenize_func(examples, tokenizer, prefix:str = 'Predict next step in the sequence for the recipe:\n'):
    ending = ' </s>'
    inputs = ['Given the recipe is '+str(x[1])+'. '+prefix+str(x[0])+ending 
              for x in zip(examples['input'], examples['recipe_type'])]
    labels = [x+ending for x in examples['label']]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
    labels_ids = tokenizer(labels, max_length=512, truncation=True, padding=True)
    model_inputs['labels'] = labels_ids['input_ids']
    return model_inputs




def downsample(lst:list, sample_idx:list)->list:
    one_hot_enc = [0] * len(lst)
    for idx in sample_idx:
        one_hot_enc[idx] = 1

    downsampled_lst = list(itertools.compress(lst, one_hot_enc))
    return downsampled_lst

def downsample_dataset(dataset:dict, downsample_num:float=1.0):
    for set_key in dataset.keys():
        keys = tuple(dataset[set_key].features.keys())
        set_size = len(dataset[set_key][keys[0]])
        sample_idx = list(sample(range(set_size), int(set_size * downsample_num)))
        for key in keys:
            dataset[set_key][key] = downsample(dataset[set_key][key], sample_idx)
    
    return dataset

def get_hf_ds(data_type:str = 'csv', data_files = {'train':'./yc2_captions/train.csv', 'test':'./yc2_captions/test.csv', 'validation':'./yc2_captions/val.csv'}):
    dataset = datasets.load_dataset('csv', data_files=data_files)

    return dataset

def tokenize_ds(dataset, tokenizer, deep_copy:bool=False, prefix:str = 'Predict next step in the sequence for the recipe:\n'):
    preproc_func = lambda x: tokenize_func(x, tokenizer, prefix)
    if deep_copy:
        import copy
        dataset = copy.deepcopy(dataset)
    
    for key in dataset.keys():
        dataset[key] = dataset[key].map(preproc_func, batched=True, remove_columns=dataset[key].column_names)
        dataset[key].set_format('torch')
    
    return dataset
    
def get_hf_dataLoaders(ds,  collator, train_batch:int = 64, val_batch:int = 64, test_batch:int = 64):
    train_loader = DataLoader(ds['train'], shuffle=True, batch_size=train_batch, collate_fn=collator)
    val_loader = DataLoader(ds['validation'], shuffle=True, batch_size=val_batch, collate_fn=collator)
    test_loader = DataLoader(ds['test'], shuffle=True, batch_size=test_batch, collate_fn=collator)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    ds = get_hf_ds()
    print(ds)
    print(ds.keys())
    print(ds['train'].features.keys())
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')

    tokenized_ds = tokenize_ds(ds, tokenizer, deep_copy=True)

    print(tokenized_ds)
    print(tokenized_ds['train'][0])

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

    train_loader, val_loader, test_loader = get_hf_dataLoaders(tokenized_ds, collator=collator)

    for batch in train_loader:
        break

    print({k: v.shape for k, v in batch.items()})




    # dl = DataLoader(dataset=ds, batch_size=24)
    # print(len(ds[idx]))

    # for input_ids, attention_mask, labels in dl:
    #     print(len(input_ids), len(attention_mask), len(labels))
    #     break
            