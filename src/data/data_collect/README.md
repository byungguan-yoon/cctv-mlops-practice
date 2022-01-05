xhost local:root

docker run -p 8888:8888 --rm -it -v /dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -v /dev/snd:/dev/snd -e="QT_X11_NO_MITSHM=1" -v ~/docker:/data -v /home/bgyoon/Documents/cctv-mlops-practice/:/home/cctv-mlops-practice --device=/dev/video0 test2