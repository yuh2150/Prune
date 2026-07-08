python prune.py --weights yolov5s.pt --pruning-params "[(0, 0.25), (42, 0.5), (45, 0.5), (48, 0.25), (25, 0.25), (51, 0.5), (55, 0.25), (56, 0.25)]" --criterion 0 --name yolov5s-pruned.pt


python layer_selection.py --output output/yolov5s --params 7225885 --flops 16.436 --params-layers 6 --flops-layers 5 --save
