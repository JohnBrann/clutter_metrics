#!/bin/bash
set -e
xhost +local:docker

docker run -it --rm --gpus all \
  --net=host \
  -v "$HOME/clutter_metrics:/clutter_metrics:rw" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -e DISPLAY="$DISPLAY" \
  --workdir /clutter_metrics \
  clutter_metrics \
  bash -lc '
    source /opt/conda/etc/profile.d/conda.sh && conda activate clutter_metrics
    export PYTHONPATH="/clutter_metrics/src:$PYTHONPATH";
    cd data_collection && python convonet_setup.py build_ext --inplace;
    exec bash
  '