#!/bin/bash
 
BASE=~/clutter_quantification

# Defaults
num_scenes_default=1
scene_prefix_default="batch1"

# Arguments
num_scenes="${2:-$num_scenes_default}"
scene_prefix="${1:-$scene_prefix_default}"

scene_prefix="${scene_prefix}_"

# Find the most recent (highest) scene number
scenes_dir="$BASE/scenes"
echo "Checking for existing scenes in: $scenes_dir"
last_scene=$(ls -d $scenes_dir/${scene_prefix}*/ 2>/dev/null | xargs -I{} basename {} | sort | tail -n 1)

# Check if no folders currently exist
if [ -z "$last_scene" ]; then
    start_num=1
else
    # Strip prefix and leading zeros, then increment
    last_num="${last_scene#${scene_prefix}}"
    last_num=$((10#$last_num))
    start_num=$((last_num + 1))
fi

end_num=$((start_num + num_scenes - 1))
echo "Last scene found: ${last_scene:-none}"
echo "Generating $num_scenes scene(s): $(printf "${scene_prefix}%03d" $start_num) to $(printf "${scene_prefix}%03d" $end_num)"

for i_num in $(seq $start_num $end_num); do
    i=$(printf "${scene_prefix}%03d" $i_num)
    echo ""
    echo "=== Scene $i ==="

    cd $BASE/data_collection

    # Update scene_name and scene_prefix in config
    sed -i "s/scene_name: .*/scene_name: \"$(printf "%03d" $i_num)\"/" config/scene_config.yaml
    sed -i "s/scene_prefix: .*/scene_prefix: \"$scene_prefix\"/" config/scene_config.yaml

    echo "Creating and collecting data..."
    python3 create_scene.py

    echo ""
    echo "Creating segmentation images..."
    python3 visualize_npy.py --scene $i

    echo ""
    cd $BASE

    echo "Calculating Occlusion Metric..."
    python3 scripts/calculate_occlusion.py --dataset-name $i

    echo ""
    echo "Calculating Proximity Metric..."
    python3 scripts/calculate_distance.py --dataset-name $i

    echo ""
    cd scenes/$i
    rm -rf object_groundtruths
    rm -rf scene_groundtruths
done