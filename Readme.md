# PerA

## Introduction
This repository implements PerA pre-training, fine-tuning, distillation, among others. The architecture of PerA is shown below:
![image](./Utils/PerA_arch.png)

## 1. Clone the repository
```
git clone git@github.com:SathShen/PerA_dev.git
```

## 2. Install the required packages
### 2.1. Python packages
Python 3.9 is required, only available for Linux. We test our code with Cuda 12.1.

Install dependencies:
```bash
pip install -r requirements.txt
```

conda install:
```bash
conda install gdal==3.6.2
```

### 2.2. Build the vit adapter
Build the vit adapter layers for linux by running the following command:
```bash
cd ./Networks/Finetune/vit_adapter_layers/ops
sh make.sh
```

## 3. Prepare the dataset
### 3.1. Pretraining dataset
The pretraining dataset should contain square images in same size in one folder. There's no other specific format required, all images under folders and subfolders will be loaded.

### 3.2. Image classification
The dataset of image classification should be in the following format:
```
dataset/
    class1/
       classA/
          img1.jpg
          img2.jpg
          ...
       classB/
          img1.jpg
          img2.jpg
          ...
    class2/
       classC/
          img1.jpg
          img2.jpg
          ...
       classD/
          img1.jpg
          img2.jpg
          ...
    ...
```
You can set --finetune_ic_num_depth 2 --finetune_ic_label_depth 1 to get exactly level of classes in the dataset.
"--finetune_ic_num_depth 2 --finetune_ic_label_depth 1" represents 2 levels of classes in the dataset, 1st level is (class1 class2..), 2nd level is (classA, classB..), etc. 
We want to finetune the model to classify images into class1, class2, according to the label_depth == 1.

### 3.3. Object detection
The dataset of object detection should be in the following format:
```
dataset/
    image/
        img1.jpg
        img2.jpg
        ...
    annotations/
        img1.xml
        img2.xml
        ...
```
The annotations should be in the VOC format.

### 3.4. Semantic segmentation
The dataset of semantic segmentation should be in the following format:
```
dataset/
    image/
        img1.jpg
        img2.jpg
        ...
    label/
        img1.png
        img2.png
        ...
```
Please remember to set the --finetune_seg_class_list to map label to index. For example, if the label is 0 for background and 255 for building, set --finetune_seg_class_list 0 255.

### 3.5. Change detection
The dataset of change detection should be in the following format:
```
dataset/
    A/
        img1.jpg
        img2.jpg
        ...
    B/
        img1.jpg
        img2.jpg
        ...
    label/
        img1.png
        img2.png
        ...
```
Please remember to set the --finetune_seg_class_list to map label to index, same as semantic segmentation.

## 4. Pretraining the model
The following is given as an example only, please adjust the command as needed.
Run in the terminal
if you want to run pretrain code with random initialization on one node with 8 gpus, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29999 pretrain.py -op output_path -e 200 -pdp your_dataset -lr 1e-3 -w 16 -b 24 -n pera -ned 1024 -nd 40 -nnh 16 -cn your_note

```
if you want to resume the pretrain code from the checkpoint, run the following command:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29999 pretrain.py -cn your_note -pp your_model.params -cfg your_config.yaml -r True
```
if you just want to load pretrained weights, set -r False and -pp to the path of the pretrained weights.

## 5. Fine-tuning 
### 5.1. Classification
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29901 finetune.py -op output_path -tp your_train_dataset -vp your_val_dataset -pp your_pretrained_model.params  -cfg your_pretrained_model_config.yaml -ibb True -ft ic -fne 500 -b 16 -is 512 -lr 1e-4 -fifb False -wd 1e-3 -wdsfv 1e-3 -gc 10.0 -cn your_note
```

### 5.2. Segmentation
if you want to run the segmentation code, run the following command
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29901 finetune.py -op output_path -tp your_train_dataset -vp your_val_dataset -pp your_pretrained_model.params -cfg your_pretrained_model_config.yaml -ibb True -ft seg -fne 500 -b 8 -is 512 -fscl 0 1 2 3 4 5 -lr 1e-4 -nadnh 16 -fifb False -gc 10.0 -cn your_note
```
if you want to resume the training, don't forget to change the -r to True, -ibb False and -pp to the path of the last checkpoint


### 5.3. Change Detection
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29901 finetune.py -op output_path -tp your_train_dataset -vp your_val_dataset -pp your_pretrained_model.params -cfg your_pretrained_model_config.yaml -ibb True -ft cd -fne 500 -b 4 -is 512 -fccl 0 255 -lr 1e-4 -nadnh 16 -fifb False -gc 10.0 -cn your_note

```

### 5.4. Object Detection
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29901 finetune.py -op output_path -tp your_train_dataset -vp your_val_dataset -pp your_pretrained_model.params -cfg your_pretrained_model_config.yaml -ibb True -ft det -fne 500 -b 8 -is 512 -lr 1e-4 -nadnh 16 -fifb False -gc 10.0 -cn your_note

```

## 6. Evaluation setup
evaluate python script only suppot single GPU now

### 6.1. Evaluate
run the code in the terminal to evaluate the fine-tuned model, if you only want to inference, set -inf True.
```bash
CUDA_VISIBLE_DEVICES=1 python evaluate.py -pp your_finetuned_model.params -cfg your_finetuned_model_config.yaml -tep your_test_dataset -op ouput_path -isp True -id True -cn your_note -inf False
```

### 6.2. Grad-CAM, Attention Map, t-SNE and Reconstruction
run the code in the terminal to visualize the attention map, the model should lload pretrained weights not finetuned weights.
```bash
CUDA_VISIBLE_DEVICES=1 python AttnMap.py -pp your_model.params -cfg your_config.yaml -tep your_test_dataset -op ouput_path -cn your_note 
```

run following command to visualize the grad-cam of the model, the model should load pretrained weights not finetuned weights.
```bash
CUDA_VISIBLE_DEVICES=1 python GradCAM.py -pp your_model.params -cfg your_config.yaml -tep your_test_dataset -op ouput_path -cn your_note
```

run following command to visualize the reconstructed image of the model, the model should load pretrained weights not finetuned weights.
```bash
CUDA_VISIBLE_DEVICES=1 python reconstruction.py -pp your_model.params -cfg your_config.yaml -tep your_test_dataset -op ouput_path -cn your_note
```

run following command to visualize the t-SNE of the model, the model should load FINETUNED weights not pretrained weights.
```bash
CUDA_VISIBLE_DEVICES=1 python tSNE.py -pp your_finetuned_model.params -cfg your_config.yaml -tep your_test_dataset -op ouput_path -cn your_note
```



## 7. Other tips
### Solve Bfloat SyncBN Bug
torch.nn.modules._functions.SyncBatchNorm
Line 96 
Replace count_all.view(-1) to count_all.view(-1).to(running_mean.dtype)
Line 146
Replace weight to weight.to(torch.float32)
### Solve MSDeformAttn Bug
/Networks/Finetune/vit_adapter_layers/ops/modules/ms_deform_attn.py Line 114:
```
output = MSDeformAttnFunction.apply(
    value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
```
to:
```
v_dtype = value.dtype
attn_dtype = attention_weights.dtype
if v_dtype != torch.float32:
    value = value.float()
if attn_dtype != torch.float32:
    attention_weights = attention_weights.float()
output = MSDeformAttnFunction.apply(
    value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
if v_dtype != torch.float32:
    output = output.to(v_dtype)
```
