#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torch-model-archiver --model-name face_model --version=1.0 --model-file /home/app/archive/model.py \
                         --serialized-file /home/app/model-store/face_model.pt --export-path /home/app/model-store/ \
                         --extra-files /home/app/archive/emp_emb_feat.npy \
                         --handler archive/model_handler.py -r /home/app/archive/requirements.txt -f
    torchserve --start --ncs --model-store /home/app/model-store --models face_model.mar --ts-config /home/app/archive/config.properties 
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
