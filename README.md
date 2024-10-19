# RLOMM

Online map matching with reinforcement learning

## Preparation
```bash
cd data_preprocess

city='beijing'

python build_road_graph.py $city
python build_road_dis.py $city
downsample_rate=0.5
python data_process.py $downsample_rate $city
python build_trace_graph.py $downsample_rate $city
python maproad2grid.py $downsample_rate $city
python build_grid_road_matrix.py $downsample_rate $city
```

## Usage
```bash
python main.py --config config/beijing_config.json --gpus 0
python main.py --config config/porto_config.json --gpus 0
```