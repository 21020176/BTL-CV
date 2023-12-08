# Object Detection in Adverse weather conditions

### Members of our team

Nguyễn Duy Phúc - 21020228	

Vũ Hoàng Duy - 21021469	

Phạm Trung Dũng - 21020176	

Nguyễn Vinh Hiển - 21021488	

Trần Trọng Quân - 21020529

## Model pretrained
We trained the DETR model on both clean and unclean CCTV dataset. The saved model below have the best performance on the test set that we devided from the annotation file. We also trained the Transweather model with CCTV dataset. The ground truth is the clean data, and we also split the train test as the annotation file.

> Detr (train with clean data) : https://drive.google.com/file/d/1zINGayaMjtbK3iqI7HN2kSXtGBK7NLRL/view?usp=sharing

> Detr (train with unclean data) : https://drive.google.com/file/d/1sCa5fumwXGiqG962BRvKWKZ-K0PJysAG/view?usp=sharing

> Transweather: https://drive.google.com/file/d/1NdlKxcQHcD2pRjMFLXWbotXyYocPGUIk/view?usp=sharing

## Data Source (CCTV):
> Clean data : https://drive.google.com/file/d/10J92QNCNdDBMBamvtYVL2snmbeXAQMF6/view

> Unclean data : https://drive.google.com/file/d/1dmqd92DqVcl6v7wYo9d3_b8hY3b3FNH9/view

> Transweather Annotation : https://drive.google.com/file/d/1NvSuTG4mU39serrjOMGFynA6P3ZYRFw6/view?usp=sharing

## Demo Web:
    https://colab.research.google.com/drive/17eHkYy6l6arivQeBL8VHh3Hz3vuG3qis?usp=sharing


## Training and Inference:
### Detr
    ///Training
    GPUS_PER_NODE=1 DEVICE=cuda ./tools/run_dist_launch.sh 1 python -m train --batch_size 2 --print_feq 10 --lr 0.0001 --lr_backbone 0.00001 \
    --model_type detr-only --wandb 'name wandb' \
    --output_dir 'path to folder that save models'
    # --resume  'path to pretrained model if exist'


    ///Inference
    DEVICE=cuda python demo.py --output_dir ./output --num 25 \
    --resume 'path to saved models' \
    --model_type detr_only

### Transweather
    ///Training
    python train.py

    ///Inference
    python infer_image.py

### Transweather dataset format:

        TransWeather
    ├── data 
    |   ├── train # Training  
    |   |   ├── <dataset_name>   
    |   |   |   ├── input         # unclean images 
    |   |   |   └── gt            # clean images
    |   |   └── dataset_filename.txt
    |   └── test  # Testing         
    |   |   ├── <dataset_name>          
    |   |   |   ├── input         # unclean images 
    |   |   |   └── gt            # clean images
    |   |   └── dataset_filename.txt

    The txt file is generated from the image directory
