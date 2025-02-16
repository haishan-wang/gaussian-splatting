
output_base=results/3dgs
dataset=mipnerf360
scene=bicycle
# scene=bonsai
# * rendering
python scripts/get_fps/get_fps_results.py   -s ../../../data/GS/$dataset/$scene -m ${output_base}/$dataset/$scene --skip_train

# # * metrics calculation
# python metrics.py -m ${output_base}/$dataset/$scene

# 16k, 133.60 bicycle