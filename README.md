# Indoor Trajecotry Forecasting
## Supermarket Dataset
Download the dataset from the following [LINK](https://drive.google.com/file/d/10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe/view?usp=sharing)

  To preprocess the file to .npy file, run the following command
  ```
  cd supermarket\dataset\preprocess.py
  python preprocess.py --subset {subset_name}
  ```
  ### Run experiments
  To train, run (you can add optional arguments)
  ```
  CUDA_VISIBLE_DEVICES={GPU_ID} python main_supermarket.py --subset {subset_name}
  ```
  To evaluate, run
  ```
  CUDA_VISIBLE_DEVICES={GPU_ID} python main_supermarket.py --subset {subset_name} --test --model_name {saved_model_name}
  ```

## Particle Dynamic
### Data preparation
We already provide the dataset file in "n_body_system/dataset". If you want to generate data by yourself, for the prediction task please run:
```
cd n_body_system/dataset
python generate_dataset.py --n_balls 5 --simulation charged --num-train 50000
```
For the reasoning task please run:
```
cd n_body_system/dataset
python generate_dataset.py --n_balls 5 --simulation springs --num-train 50000
```
### Run experiments
For the prediction task:
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_nbody.py 
```

For the reasoning task:
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_nbody_reasoning.py 
```

## Molecule Dynamic
### Data preparation
The MD17 dataset can be downloaded from [MD17](http://www.sgdml.org/#datasets). Put the downloaded file in "md17/dataset" and run
```
cd md17/
python preprocess.py 
```
### Run experiments
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_md17.py --mol {molecule_name} 
```

## 3D Human Skeleton Motion
### Data preparation
Download Human3.6M dataset from its [website](http://vision.imar.ro/human3.6m/description.php) and put the files into "h36m/dataset".
### Run experiments
#### Training
To train a model of short-term prediction task, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_h36m.py --past_length 10 --future_length 10 --channel 72  
```
To train a model of long-term prediction task, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_h36m.py --past_length 25 --future_length 25 --channel 96 --apply_decay  
```
#### Evaluation
To evaluate a model of short-term prediction task, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_h36m.py --past_length 10 --future_length 10 --channel 72 --model_name {your_model_name} --test
```
To evaluate a model of long-term prediction task, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_h36m.py --past_length 25 --future_length 25 --channel 96 --model_name {your_model_name} --test
```
## Pedestrian Trajectory
### Data preparation
To preprocess the raw data to .npy file, run

```
cd eth_ucy/
python process_eth_data_diverse.py --subset {subset_name} 
```
### Run experiments
To train, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_eth_diverse.py --subset {subset_name}
```
To evaluate, run
```
CUDA_VISIBLE_DEVICES={GPU_ID} python main_eth_diverse.py --subset {subset_name} --test --model_name {saved_model_name}
```

