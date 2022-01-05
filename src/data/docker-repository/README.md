## Docker Image Repository(Nexus) 연결 방법

### 목적
- 로컬에서 만든 도커 이미지를 저장소에 저장해 엣지 단에 배포할 때 사용 

### 환경 설정
```
# local terminal
vim /etc/docker/daemon.json
```

```
# daemon.json
{
        "insecure-registries" : ["183.105.120.175:2224"]
} 
```

```
service docker restart
```

### 로그인
```
docker login 183.105.120.175:2224
```
### Docker Image Pull
```
docker pull busybox
```
### Docker Image Push
```
docker push 183.105.120.175:2224/busybox:v20200205
```
