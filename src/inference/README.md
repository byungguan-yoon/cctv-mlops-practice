### Docker 사용방법 ###

# 1. ./build_image.sh -g -cv cu102 -t torchserve:cu102
# 2. docker run -itd --gpus '"device=0"' --name torchserve -v ${PWD}/archive:/home/app/archive -v ${PWD}/model-store:/home/app/model-store \
                -p 8090:8090 -p 8091:8091 -p 8092:8092 -p 7070:7070 -p 7071:7071 torchserve:cu102
