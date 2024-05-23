# BohoAttack: Physical-realizable, Free-pose and Transferable Evasion Attacks against Person Detectors
This is the repository for the paper BohoAttack: Physical-realizable, Free-pose and Transferable Evasion Attacks against Person Detectors

<!-- toc -->
## 1. Installation
### Requirements
conda env create -f environment.yml 

conda activate boho

## 2. Preparation: Use pretrained UV-Volume model
The data and checkpoints are shared by [Google Drive](). You need to download it and place the *data* folder in the root directory of this project. If you want to evaluate the checkpoints, place the *results* folder also in the root directory and follow the instructions in the section of [Evaluation](#4-evaluation).


If you are going to use yolov3, you need to download its weights by running
```
./arch/weights/download_weights.sh
```

## 3. Preparation: Train UV-Volume model from scratch
### Download the [zju_mocap dataset](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset) and put the dataset under the rootfolder, like
- rootfolder/
  - zju_comap/
    - CoreView_377/
    - CoreView_387/
    - ...

### Prepare UV-Volume model
Copy the [UV-Volume](https://github.com/fanegg/UV-Volumes) code to the rootfolder. Then train the UV-Volumes through
```
python3 train_net.py --cfg_file configs/zju_mocap_exp/377.yaml
exp_name zju377 resume True output_depth True
```

### Prepare background data
We use the background data collected from the [AdvCaT](https://github.com/WhoTHU/Adversarial_camou). Some test images are put under the data folder.

## 3. Train
We provide the command to optimize AdvCaT for different target detectors.

##### Faster-RCNN with prompt "one horse"
python diffusion_model_patch_generator_uv_volumes.py --arc rcnn --prompt "one horse" --pattern_mode "repeat" --checkpoints 0 --lr 0.005 --device cuda:0 --do_classifier_free_guidance False --half_precision_weights True

##### YOLOv3 with prompt "two bears"
python diffusion_model_patch_generator_uv_volumes.py --prompt "two bears" --pattern_mode "repeat" --lr 0.01 --device cuda:2 --do_classifier_free_guidance False --half_precision_weights True --checkpoints 0


## 4. Evaluation
We provide the command to evaluate BohoAttack and visualize the result. For example, to evaluate the pattern saved in directory 'results/rcnn' targeting 
To visualize the evaluation results, run

python diffusion_model_patch_generator_uv_volumes.py --arc rcnn --prompt "one horse"  --pattern_mode "repeat" --checkpoints 0 --lr 0.005 --device cuda:0 --do_classifier_free_guidance False --half_precision_weights True --test True

