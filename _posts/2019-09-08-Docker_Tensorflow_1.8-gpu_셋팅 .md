
---
layout: post

title:  "[linux] Docker Tensorflow 1.8-gpu 셋팅  "

subtitle:   "[linux] Docker Tensorflow 1.8-gpu 셋팅 "

categories: linux

tags: docker ubuntu18.04 tensorflow1.8 gpu 

comments: true

img: 


---


## 1. Docker Pull
```shell
docker pull tensorflow/tensorflow:1.8.0-gpu-py3       // 태그명이 1.8.0-gpu-py3인 텐서플로우 이미지 다운로드
docker images  

```

## 2. Docker Run 
```shell
nvidia-docker run -it \
-p 8888:8888 -p 6006:6006 \
--name ailab-yurim \
-v /home/zayden/Desktop/test_garbage:/notebooks \
-e PASSWORD="0000" \
--restart always \
tensorflow/tensorflow:1.8.0-gpu-py3
```


## 3. Docker Commit 
```shell
docker commit ailab-yurim riverzayden/ailab-yurim:0.1
```


## 4. Docker stop, remove 
```shell
docker ps -a  

docker stop eade3e95f777

docker rm eade3e95f777
```


## 5. Commit된것으로 다시 실행 해보기 
```shell
nvidia-docker run -it \
-p 8888:8888 -p 6006:6006 \
--name ailab-yurim \
-v /home/zayden/Desktop/test_garbage:/notebooks \
-e PASSWORD="0000" \
--restart always \
riverzayden/ailab-yurim:0.1
```


## 6. Docker Login and Push
```shell 
docker login
#docker tag local-image:tagname new-repo:tagname
docker tag riverzayden/ailab-yurim:0.1
docker push riverzayden/ailab-yurim:0.1
docker pull riverzayden/ailab-yurim:0.1

```


#### 출처
```html
https://devyurim.github.io/python/tensorflow/development%20enviroment/docker/2018/05/25/tensorflow-3.html
```