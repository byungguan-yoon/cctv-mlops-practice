## Docker Image Repository 연결 방법

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