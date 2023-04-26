import sys
sys.path.insert(0, "./Model_Train")
sys.path.insert(0, "./SwinBERT")
import Model_Train.CaptionDataset as CaptionDataset
import Model_Train.model_test as model_test
import SwinBERT.get_inferred_ds as get_inferred_ds

import datasets
import pandas as pd
import argparse



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dataset', type=str, default='./test.csv', help='Filepath for test dataset')
    parser.add_argument('--device', type=str, default='cpu', help='PyTorch device - i.e. "cpu", "cuda", "cuda:0", "cuda:1", etc.')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Test batch size for predicting next step')

    return parser.parse_args()

def main():
    args = get_args()
    ds = get_inferred_ds.get_infered_models(test_dataset = args.test_dataset, device=args.device)
    model_test.test_model(test_dataset_type='pandas', test_dataset=ds, device=args.device, batch_size=args.batch_size)


if __name__ == '__name__':
    main()


