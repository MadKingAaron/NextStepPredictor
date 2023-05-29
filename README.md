# Next Step Predictor

## Getting Started

### Downloading SwinBERT model
* Download the modified fork of the original SwinBERT library from the link [https://github.com/MadKingAaron/SwinBERT](https://github.com/MadKingAaron/SwinBERT)
* Follow the instructions on how to set up the SwinBERT model. The instructions can be found in `./SwinBERT/README.md` or [README.md](https://github.com/MadKingAaron/SwinBERT/blob/main/README.md)

### Setting up language model
* The datasets used for the project can be found in the following shared Google Drive folder: [https://drive.google.com/drive/folders/1kITjPs__-JSP-j1-JyTFMW-Easgr0I0O?usp=sharing](https://drive.google.com/drive/folders/1kITjPs__-JSP-j1-JyTFMW-Easgr0I0O?usp=sharing)
* The models fine-tuned in the paper can be found in the following Google Drive folder: [https://drive.google.com/drive/folders/1NHILzC1Q4YXbJ1mZr9pnD-muaznHlNzW?usp=share_link](https://drive.google.com/drive/folders/1NHILzC1Q4YXbJ1mZr9pnD-muaznHlNzW?usp=share_link)
    * Please place all models downloaded in `./Models`

### Libraries 
To install the libraries, please run `pip install -r ./requirements.txt`
**NOTE: Make sure to run with PyTorch >= 1.6 (any version with AMP integrated into it)**


### Inference Videos
When adding videos for inference, make sure to place the videos in the following format example:

If you are running inference on a video titled `<video_title>.mp4`, make sure to place all segments of `<video_title>.mp4`-- `<video_title>_1.mp4`, `<video_title>_2.mp4`, ..., `<video_title>_n.mp4` -- in `./Videos/<video_title>`

The folder layout should look like this:
```plaintext
.
├── SwinBERT
├── Model_Train
├── Models
└── Videos
    └── <video_title>
        ├── <video_title>_0.mp4
        ├── <video_title>_1.mp4
        ├── <video_title>_2.mp4
        └── <video_title>_3.mp4
```



## Running Inference
In order to run inference of the dataset of your choice please run the following:

`python inference.py --test_dataset <test_dataset_path> --video_dir <video_folder_dir> --lang_model <lang_model_dir> --device <device> --batch_size <testing_batch_size>`


An example of running inference would be:

`python inference.py --test_dataset ./test.csv --video_dir ./Videos --lang_model ./Models/FLAN-T5-small-merged --device cuda --batch_size 32`