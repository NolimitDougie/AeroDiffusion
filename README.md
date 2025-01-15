# AeroDiffusion






## Environment
```shell
conda create -n aerodiff python=3.11
conda activate aerodiff
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
git clone git@github.com:NolimitDougie/AeroDiffusion.git
cd AeroDiffusion
pip install -r requirements.txt
```

##  Training Object Detection Model
```shell
python train.py --img 640 --batch 16 --epochs 100 --data visdrone.yaml --weights yolov5s.pt --project visdrone-results --name yolov5-visdrone
```
## Configuration for Aero Diffusion 
Add the trained object detection model to [Link Text](yolov5/region.py)





## Data Preparation
* Download the VisDrone DET dataset [here](https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?pli=1).


## Training
Define your directory paths, device configuration, and training parameters in `config.yaml`, then execute the script.
```shell
python main.py
```
## Generating Aerial Images

Specify your directory paths, device configuration, inference steps, and guidance scale in `config.yaml`, then execute the script.
```shell
python main.py
```
