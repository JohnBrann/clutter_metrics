#!/bin/bash
 
BASE=~/clutter_quantification
 
for i in $(seq 1 3); do
    echo ""
    echo "=== Scene $i ==="
 
    cd $BASE/data_collection
 
    # Update scene_name in config
    sed -i "s/scene_name: .*/scene_name: \"$i\"/" config/scene_config.yaml
 
    python3 create_scene.py
    echo ""
    python3 visualize_npy.py --scene $i
    echo ""
    cd $BASE
    python3 scripts/calculate_occlusion.py --dataset-name $i
    echo ""
    python3 scripts/calculate_distance.py --dataset-name $i
    echo ""
 
done
