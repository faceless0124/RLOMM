cd data_preprocess

city='beijing'

python build_road_graph.py $city
echo 'finish build_road_graph'

python build_road_dis.py $city
echo 'finish build_road_dis'

downsample_rate=0.5

python data_process.py $downsample_rate $city
echo 'finish data_process'

python build_trace_graph.py $downsample_rate $city
echo 'finish build_trace_graph'

python maproad2grid.py $downsample_rate $city
echo 'finish maproad2grid'

python build_grid_road_matrix.py $downsample_rate $city
echo 'finish build_grid_road_matrix'