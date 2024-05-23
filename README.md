# Physically Realizable Natural-looking Clothing Textures Evade Person Detectors via 3D Modeling

This is the official repository for the paper [Physically Realizable Natural-looking Clothing Textures Evade Person Detectors via 3D Modeling](https://openaccess.thecvf.com/content/CVPR2023/html/Hu_Physically_Realizable_Natural-Looking_Clothing_Textures_Evade_Person_Detectors_via_3D_CVPR_2023_paper.html).

<!-- toc -->
#### 1. Installation
### Requirements
All the codes are tested in the following environment:
* Linux (Ubuntu 16.04.6)
* Python 3.8.13
* PyTorch 1.10.1
* pytorch3d 0.6.2
* CUDA 11.0
* TensorboardX 2.5.1

#### 2. Preparation
The data and checkpoints are shared by [Google Drive](https://drive.google.com/file/d/1Uddyu5pjFymjX66AA4HnEKk3fA7r8UVT/view). You need to download it and place the *data* folder in the root directory of this project. If you want to evaluate the checkpoints, place the *results* folder also in the root directory and follow the instructions in the section of [Evaluation](#4-evaluation).

If you are going to use yolov3, you need to download its weights by running
```
./arch/weights/download_weights.sh
```
#### 3. Train
We provide the command to optimize AdvCaT for different target detectors.

##### Faster-RCNN
```
python train.py --nepoch 600 --save_path 'results/rcnn_sr07' --ctrl 50 --arch "rcnn" --seed_type variable --clamp_shift 0.01 --loss_type max_iou --seed_ratio 0.7
```
##### Deformable Detr
```
python train.py --nepoch 600 --save_path 'results/deformable_detr_07' --ctrl 50 --arch "deformable-detr" --seed_type variable --clamp_shift 0.01 --loss_type max_iou --seed_ratio 0.7
```
##### YOLOv3
```
python train.py --nepoch 600 --save_path 'results/yolov3_07' --ctrl 50 --arch "yolov3" --seed_type variable --clamp_shift 0.01 --loss_type max_iou --seed_ratio 0.7
```
#### 4. Evaluation
We provide the command to evaluate AdvCaT and visualize the result. For example, to evaluate the pattern saved in directory 'results/rcnn_sr07' targeting FasterRCNN, run
```
python train.py --device --checkpoint 600 --save_path 'results/rcnn_sr07' --ctrl 50 --arch "rcnn" --seed_type variable --clamp_shift 0.01 --seed_ratio 0.7 --test
```

To visualize the evaluation results, run
```
python visualize.py
```

python diffusion_model_patch_generator.py --prompt "three bear" --pattern_mode "repeat" --checkpoints 250 --lr 0.005 --device cuda:7 --do_classifier_free_guidance False --half_precision_weights True

python diffusion_model_patch_generator.py --prompt "two bears" --pattern_mode "repeat" --lr 0.01 --device cuda:2 --do_classifier_free_guidance True --half_precision_weights True --batch_size 4 --checkpoints 325

python diffusion_model_patch_generator.py --prompt "colorful repeated patterns" --pattern_mode "whole" --lr 0.01 --device cuda:2 --do_classifier_free_guidance False --half_precision_weights False --batch_size 4 --checkpoints 90

python diffusion_model_patch_generator.py --optimize_type image --diffusion_rate 0.3 --diffusion_steps 20 --prompt "three bears" --pattern_mode "repeat" --lr 0.001 --device cuda:3 --do_classifier_free_guidance True --half_precision_weights False --batch_size 2 --checkpoints 0


# failed 
conda create -n densepose python==3.7
conda activate densepose
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
export DENSEPOSE=~/AIGC/Adversarial_camou/detectron
git clone https://github.com/facebookresearch/densepose $DENSEPOSE
pip install -r $DENSEPOSE/requirements.txt
cd $DENSEPOSE && make
python tools/infer_simple.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    --output-dir ../demo_out/ \
    --image-ext jpg \
    --wts https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    ../demo


