#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    torch-model-archiver --model-name model --version=1.0 --model-file /home/app/archive/model.py --serialized-file /home/app/model-store/dot_model.ckpt --export-path /home/app/model-store/ --handler archive/model_handler.py -r /home/app/archive/requirements.txt -f
    torchserve --start --ncs --model-store /home/app/model-store --models model.mar --ts-config /home/app/archive/config.properties 
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
