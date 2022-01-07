# Docker 사용방법

### 1. Dockerfile build하여 docker image 생성 Deploy하려는 server의 그래픽 드라이버 확인후 cuda version을 cu102 cu111 etc...맞춰준다.
```
  ./build_image.sh -g -cv cu102 -t torchserve:cu102
```

### 2. 1번에서 build된 docker image를 run 하여준다. http/grpc port정보가 바뀌면 -p 옵션을 바꾸어 준다.
```
  docker run -itd --gpus '"device=0"' --name torchserve 
             -v ${PWD}/archive:/home/app/archive \
             -v ${PWD}/model-store:/home/app/model-store \
             -p 8090:8090 -p 8091:8091 -p 8092:8092 -p 7070:7070 -p 7071:7071 torchserve:cu102
```
