# ZeroShotOpt
[overall figure](https://github.com/user-attachments/files/20674560/overall_fig_may.pdf)

This repository contains the code for ZeroShotOpt, a transformer-based model for zero-shot global black-box optimization. It serves as a "plug-and-play" optimizer, addressing the common issue where the performance of state-of-the-art methods like Bayesian optimization (BO) depends on hand-tuned hyperparameters that fail to generalize. The model is trained on millions of synthetic functions generated using Gaussian processes  and demonstrates strong generalization to various synthetic and real-world benchmarks. ZeroShotOpt is trained using offline reinforcement learning on a large dataset of optimization trajectories collected from 12 BO variants. It is a 200 million parameter model trained on data ranging from 2D to 20D. The model has been tested on benchmarks including the Virtual Library of Simulated Experiments (VLSE), the Black-Box Optimization Benchmark (BBOB), and the Hyperparameter Optimization Benchmark (HPO-B). On these unseen tasks, ZEROSHOTOPT matches or surpasses the sample efficiency of leading global optimizers. The entire pipeline, including data generation, training, and testing of the model, as well as the dataset and pretrained model, are included in this repository.

## Installation
Uses **Python 3.11**. Install dependencies with:

```bash
pip install -r requirements.txt
```
## Data and Pretrained model

Our dataset and pretrained model can be found at the following [link](https://www.dropbox.com/scl/fo/t2r2212ebsstsako2fnig/ACRP_D286WIvowm-jRV9tQo?rlkey=izvljf3z9gk96k1p42ki02cit&st=w4es6wz0&dl=0). This contains training data from 2D to 20D that was used to train our full model, as well as our test results. This data is contained within pickle files for each dimension that contain all the information about each trajectory, including actions, states, and metadata. Each pickle file contains a NumPy array with a dictionary representing each trajectory as an entry in this array. Additionally, the folder contains our pretrained model that can be used to reproduce our results using the testing methodology described below.

## Data Generation

Data generation can be found in the baselines folder.

```bash
python generate.py \
  --cuda False \
  --result-dir sample/train_2d_40 \
  --num-envs 100 \
  --num-proc 48 \
  --seed 0 \
  --env-id GPEnv-2D-v0 \
  --num-steps 40
```

You can adjust the environment id, seed, and number of steps according to different selections for the generation.

## Compiling Data 

Code for compiling data can be found in model/compile.py. This performs preprocessing on the dataset to speed up training. You can adjust the parameters in the file for different dataset size and dimensions.

## Training

Training can be done with the following in the model folder:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 train.py --config simple_model.yaml
```

Adjustments to the parameters and data used for training are found within the config file. We provide a simple version for testing a small 2D-3D model and the config for our full model. 

## Testing

Testing the model can be done with the following in the model folder:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --model-path ZeroShotOpt/ckpt.pt \
  --num-envs 100 \
  --env-id bbob_2d \
  --num-steps 40 \
  --length-type adaptive \
  --norm-type traj_minmax_scaled_high \
  --sampling top_p \
  --input-dir '../baselines/test_100/bbob_2d_40' \
  --output-dir '../baselines/test_100/bbob_2d_40' 
```
You can adjust the parameters and model used for testing. Currently, testing is limited to environments following our specified structure. We plan to expand support for additional function formats in future updates.

Testing all baseline methods can be done with the following in the baselines folder:
```bash
python test.py \
  --env-id bbob_2d \
  --num-envs 100 \
  --num-proc 48 \
  --output-dir test_100/bbob_2d_40 \
  --num-steps 40
```

