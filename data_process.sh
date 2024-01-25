cd data_preprocess

python build_road_graph.py
echo 'finish build_road_graph'

downsample_rate=0.5

python build_road_dis.py $downsample_rate
echo 'finish build_road_dis'

python data_process.py $downsample_rate
echo 'finish data_process'

python build_trace_graph.py $downsample_rate
echo 'finish build_trace_graph'

python maproad2grid.py $downsample_rate
echo 'finish maproad2grid'

python build_grid_road_matrix.py $downsample_rate
echo 'finish build_grid_road_matrix'