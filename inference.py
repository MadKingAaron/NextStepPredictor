import os, sys
import contextlib
sys.path.insert(0, "./Model_Train")
sys.path.insert(0, "./SwinBERT")
import Model_Train.CaptionDataset as CaptionDataset
import Model_Train.model_test as model_test
import SwinBERT.get_inferred_ds as get_inferred_ds


import pandas as pd
import argparse


@contextlib.contextmanager
#temporarily change to a different working directory
def changeCWD(path):
    _oldCWD = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.abspath(path))

    try:
        yield
    finally:
        os.chdir(_oldCWD)


def get_infered_samples(test_dataset:str, device:str, video_dir:str):
    cwd = os.path.dirname(os.path.abspath(__file__))
    full_filepath_ds = os.path.join(cwd, test_dataset)
    full_videopath = os.path.join(cwd, video_dir.replace('./', ''))
    print(device)

    with changeCWD('./SwinBERT'):
        ds = get_inferred_ds.get_infered_models(test_dataset=full_filepath_ds, device=device, video_folder=full_videopath)

    return ds

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_dataset', type=str, default='./vid_split.csv', help='Filepath for test dataset')
    parser.add_argument('--video_dir', type=str, default='./Videos', help='Directory containing all video files')
    parser.add_argument('--device', type=str, default='cuda:6', help='PyTorch device - i.e. "cpu", "cuda", "cuda:0", "cuda:1", etc.')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Test batch size for predicting next step')

    return parser.parse_args()





def main():
    args = get_args()
    #print(args)
    ds = get_infered_samples(test_dataset = args.test_dataset, device=args.device, video_dir=args.video_dir)
    #print('Done')
    ds = pd.read_csv(args.test_dataset)
    model_test.test_model(test_dataset_type='pandas', test_dataset=ds, device=args.device, batch_size=args.batch_size)
    #print(ds)


if __name__ == '__main__':
    main()


