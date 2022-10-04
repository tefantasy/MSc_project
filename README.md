# Exploiting Motion and Visibility Cues for Tackling Occlusions in Multiple Object Tracking

This repository provides the code for the project: Exploiting Motion and Visibility Cues for Tackling Occlusions in Multiple Object Tracking.
This is a master's project of MSc DSML, done in Prism group, CS department, UCL. 

The report is [here](thesis.pdf).

## Installation

1. Clone and enter this repository. 

2. Install dependencies of this project for Python 3.7:
    1. `pip3 install -r requirements.txt`
    2. Install this project: `pip3 install -e .`

3. MOTChallenge data:
    1. Download [MOT17Det](https://motchallenge.net/data/MOT17Det.zip) and [MOT17Labels](https://motchallenge.net/data/MOT17Labels.zip) and place them in the `data` folder. 
    
    2. Unzip all the data by executing:
    ```
    unzip -d MOT17Det MOT17Det.zip
    unzip -d MOT17Labels MOT17Labels.zip
    ```

4. Download object detector and re-identifiaction Siamese network weights and MOTChallenge result files (provided by [Tracktor++](https://github.com/phil-bergmann/tracking_wo_bnw) project):
    1. Download zip file from [here](https://vision.in.tum.de/webshare/u/meinhard/tracking_wo_bnw-output_v2.zip).
    2. Extract in `output` directory.

5. Download the parameters of Neural Motion Model from [here](https://drive.google.com/file/d/1De0buwcdaNugu7I4OKliSrnmNsrfAle2/view?usp=sharing), and put it in `output/motion` directory. You may need to make a new `motion` directory. 

## Evaluation

0. (if you would like to evaluate on validation split of MOT17 training set) Run ``` experiments/scripts/create_val_dataset.py ```. 

1. The default configuration is evaluating Neural Motion Model on the validation split of MOT17 training set, by executing:

  ```
  python experiments/scripts/test_tracktor.py
  ```

2. The results are logged in the corresponding `output` directory.

3. If you would like to evaluate Neural Motion Model on the full MOT17 training set, you can modify the line 

  ```
  dataset = Datasets(tracktor['dataset'], {'use_val_split':True})
  ```
into
  ```
  dataset = Datasets(tracktor['dataset'])
  ```
in ``` test_tracktor.py ```. For other configurations of dataset please change ``` experiments/cfgs/tracktor.yaml ```.

## Training Neural Motion Model

0. (optional) Run ``` experiments/scripts/precompute_ecc_matrices.py ``` to avoid re-computing alignment matrices during training and evaluation. 

1. Pretrain visibility estimator using the script ``` experiments/scripts/train_vis_reid.py ```. 

2. Train the main branch of Neural Motion Model using the script ``` experiments/scripts/finetune_motion_model.py ```.

## Training and testing object detector, training the reidentifaction model

Please follow the corresponding instructions in the [Tracktor++](https://github.com/phil-bergmann/tracking_wo_bnw) project. 

## Acknowledgement
 Part of the codes in this repository are based on the following projects:

**[Tracktor++](https://github.com/phil-bergmann/tracking_wo_bnw) (Tracking Without Bells and Whistles)**
```
  @InProceedings{tracktor_2019_ICCV,
  author = {Bergmann, Philipp and Meinhardt, Tim and Leal{-}Taix{\'{e}}, Laura},
  title = {Tracking Without Bells and Whistles},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}}
```

**[py-motmetrics](https://github.com/cheind/py-motmetrics)**
