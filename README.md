# BohoAttack: Physical-realizable, Free-pose and Transferable Evasion Attacks against Person Detectors
This is the repository for the paper BohoAttack: Physical-realizable, Free-pose and Transferable Evasion Attacks against Person Detectors

<!-- toc -->
## 1. Installation
### Requirements
conda env create -f environment.yml 

conda activate boho

## 2. Preparation: Use pretrained model

### 2.1 Use pretrained UV-Volume model
The data and checkpoints are shared by [Google Drive](). You need to download it and place the *data* folder in the root directory of this project. If you want to evaluate the checkpoints, place the *results* folder also in the root directory and follow the instructions in the section of [Evaluation](#5-evaluation). We also provided a pretrained texture stacks under data/texture_stacks

If you are going to use yolov3, you need to download its weights by running
```
./arch/weights/download_weights.sh
```
### 2.2 Use pretrained stable diffusion model
We use the pretrained [miniSD](https://huggingface.co/justinpinkney/miniSD) version, which is fintuned on the 256*256 images.


### 2.3 Use the SMPL model
In addition, if use GMM sampling (by set --use_GMM as True in the diffusion_model_patch_generator_uv_volumes.py), you should download the [SPIN code](https://github.com/nkolot/SPIN) and put the [SMPL](https://smpl.is.tue.mpg.de/) model to the SPIN/data folder. If set use_GMM as False, we will sample from exist poses.

## 3. Preparation: Train UV-Volume model from scratch
If you want to test the ASR on other person in ZJU-mocap datasets, you can train UV-Volume model from scratch.
### 3.1 Download the [zju_mocap dataset](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) and put the dataset under the rootfolder, like
- rootfolder/
  - zju_comap/
    - CoreView_377/
    - CoreView_387/
    - ...
### 3.2 Prepare UV-Volume model
Train the UV-Volumes through
```
python3 train_net.py --cfg_file configs/zju_mocap_exp/377.yaml
exp_name zju377 resume True output_depth True
```
Then the texture stacks will be put under the ./UV_Volumes/data/evaluate/UVvolume_ZJU/zju377 folder.


## 4. Train
We provide the command to optimize Boho for different target detectors.
### Prepare background data
We use the background data collected from the [AdvCaT](https://github.com/WhoTHU/Adversarial_camou). Some test images are put under the data folder.

##### Faster-RCNN with prompt "one horse"
python diffusion_model_patch_generator_uv_volumes.py --arc rcnn --prompt "one horse" --pattern_mode "repeat" --checkpoints 0 --lr 0.005 --device cuda:0 --do_classifier_free_guidance False --half_precision_weights True

##### YOLOv3 with prompt "two bears"
python diffusion_model_patch_generator_uv_volumes.py --prompt "two bears" --pattern_mode "repeat" --lr 0.01 --device cuda:2 --do_classifier_free_guidance False --half_precision_weights True --checkpoints 0


## 5. Evaluation
We provide the command to evaluate BohoAttack and visualize the result. For example, to evaluate the pattern saved in directory 'results/rcnn' targeting 
To visualize the evaluation results, run

python diffusion_model_patch_generator_uv_volumes.py --arc rcnn --prompt "one horse"  --pattern_mode "repeat" --checkpoints 0 --lr 0.005 --device cuda:0 --do_classifier_free_guidance False --half_precision_weights True --test True

