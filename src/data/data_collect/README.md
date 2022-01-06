## Data Collect Service 실행방법

### 도커 이미지 빌드 및 실행
```
docker build -t {tag_name} .

# cv2 imread를 위한 설정
xhost local:root
# /home/bgyoon/blah-blah 본인 디렉토리에 맞게 수정
docker run -p 8888:8888 --rm -it -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v /dev/snd:/dev/snd -e="QT_X11_NO_MITSHM=1" -v ~/docker:/data -v /home/bgyoon/Documents/cctv-mlops-practice/:/home/cctv-mlops-practice --device=/dev/video0 {tag_name}
```

### 이미지 저장 디렉토리
- cctv-mlops-practice/data/raw/{연도-월-일}/{시-분-초}
- ex) cctv-mlops-practice/data/raw/2022-01-05/13-51-50
