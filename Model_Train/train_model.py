import CaptionDataset
import train_test_model
import argparse


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import pandas as pd
import check_gpu_mem


def train():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    df = pd.read_csv('./trainVal_samples.csv')


    trainloader, valloader, testloader = CaptionDataset.get_loaders(tokenizer, 'Predict next step in sequence:\n', prepend_prefix=True,
                                                                    df=df, train_split=.6, train_batch=64, val_split=0.3, val_batch=64,
                                                                    test_split=0.1, test_batch=64)

    optimizer, scheduler = train_test_model.get_optimzer(initial_lr=5e-8, model=model)

    model = train_test_model.train_model(optimizer=optimizer, trainloader=trainloader, valloader=valloader, model=model, epochs=10, scheduler=scheduler, device='cuda', tb_comment='LR_5e-8_Epoch_10')
    model.save_pretrained('./flan-t5-small-trained', from_pt=True)


def get_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    return model, tokenizer

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-8)
    parser.add_argument('--checkpt_freq', type=int, default=10)
    parser.add_argument('--tb_comment', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--adamw', action='store_true')

    return parser.parse_args()

def get_best_device(num_devices:int=8):
    best_dev = check_gpu_mem.get_best_free_gpu(num_devices)
    return best_dev

def train_hf_ds(batch_size=64, epochs=100, lr=3e-8, checkpt_freq=10, tb_comment='', adamw = False):
    datafiles = {'train':'./yc2_captions/train_new_split.csv', 'test':'./yc2_captions/test_new_split.csv', 'validation':'./yc2_captions/val_new_split.csv'}
    model, tokenizer = get_model()

    device = check_gpu_mem.get_best_free_gpu(8)
    #device = -1
    print('GPU:', device)
    model = model.to(device)
    
    optimizer, lr_scheduler = train_test_model.get_optimzer(initial_lr=lr, model=model, adamw=adamw)

    dataset = CaptionDataset.get_hf_ds(data_files=datafiles)#(data_files = {'train':'./yc2_captions/train_masked.csv', 'test':'./yc2_captions/test_masked.csv', 'validation':'./yc2_captions/val_masked.csv'})
    
    print(dataset)
    
    tokenized_ds = CaptionDataset.tokenize_ds(dataset, tokenizer, deep_copy=True)#, prefix= 'Predicted masked steps in sequence of steps:\n')
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    train_dataloader, val_dataloader, test_dataloader = CaptionDataset.get_hf_dataLoaders(tokenized_ds, collator, train_batch=batch_size, val_batch=batch_size, test_batch=batch_size)

    model = train_test_model.train_model_hf(optimizer, trainloader=train_dataloader, valloader=val_dataloader, model=model, epochs=epochs, scheduler=lr_scheduler,
                                                   tb_comment=tb_comment, device=device, checkpt_freq=checkpt_freq)
    model.save_pretrained('./flan-t5-small-trained', from_pt=True)

if __name__ == "__main__":
    args = get_args()
    train_hf_ds(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, checkpt_freq=args.checkpt_freq, tb_comment=args.tb_comment, adamw=args.adamw)