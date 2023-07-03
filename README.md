# Indoor Trajecotry Forecasting


<html>
<head>
<style>
table {
  border-collapse: collapse;
  width: 100%;
}


</style>
</head>
<body>

<table>
  <caption style="font-weight: bold; font-size: 1.5em;">ADE/FDE score Comparison</caption>
  <thead>
    <tr>
      <th rowspan="2">Model Name</th>
      <th colspan="5">Outdoor Dataset</th>
      <th colspan="2">Indoor Dataset</th>
    </tr>
    <tr>
      <th>ETH</th>
      <th>Hotel</th>
      <th>Univ</th>
      <th>ZARA01</th>
      <th>ZARA02</th>
      <th>Supermarket</th>
      <th>THÖR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PECnet</td>
      <td>0.88/1.56</td>
      <td>0.40/0.63</td>
      <td>0.66/0.95</td>
      <td>0.29/0.42</td>
      <td>0.20/0.17</td>
      <td>1.59/1.42</td>
      <td>Value</td>
    </tr>
    <tr>
    </tr>
    <tr>
      <td>Eqmotion (Deterministic)</td>
      <td>0.94/1.90</td>
      <td>0.30/0.56</td>
      <td>0.50/1.08</td>
      <td>0.40/0.88</td>
      <td>0.32/0.70</td>
      <td>3.84/6.33</td>
      <td>Value</td>
    </tr>
    <tr>
      <td>Eqmotion (Multi Prediction)</td>
      <td>0.42/0.62</td>
      <td>0.14/1.88</td>
      <td>0.23/0.42</td>
      <td>0.21/0.37</td>
      <td>0.13/0.24</td>
      <td>2.65/2.20</td>
      <td>Value</td>
    </tr>
    <tr>
      <td>Our model</td>
      <td>Value</td>
      <td>Value</td>
      <td>Value</td>
      <td>Value</td>
      <td>Value</td>
      <td>Value</td>
      <td>Value</td>
    </tr>
  </tbody>
</table>

</body>
</html>



## Thor Dataset
To download the dataset, run
```
cd thor/raw_dataset
./download.sh
```
You can also download the dataset from the following [LINK](https://zenodo.org/record/3382145)

**Note**: Download tsv files for 3D data

To preprocess the file to .npy file, run the following command, this will preprocess all .tsv files in the raw_dataset folder
  ```
  python thor/preprocess.py
  ```
To visualize the processed data, run the following command
```
python visualize.py
```

  ### Run experiments
  To train, run (you can add optional arguments)
  ```
  CUDA_VISIBLE_DEVICES={GPU_ID} python main_thor.py --subset {subset_name}
  ```
  To evaluate, run
  ```
  CUDA_VISIBLE_DEVICES={GPU_ID} python main_thor.py --subset {subset_name} --test --model_name {saved_model_name}
  ```
## Supermarket Dataset
To download the dataset, run
```
python -u supermarket/dataset/download_dataset.py
```
You can also download the dataset from the following [LINK](https://drive.google.com/file/d/10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe/view?usp=sharing)

  To preprocess the file to .npy file, run the following command
  ```
  cd supermarket/dataset/
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

