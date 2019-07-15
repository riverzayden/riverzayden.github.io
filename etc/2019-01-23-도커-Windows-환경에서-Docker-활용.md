---
layout: post
category: tech
title: "도커: Windows 환경에서 Docker 활용"
description: OPEN
tags: tech
img: docker.png
comments: true
---



## 0. 도커 설치 on Windows 10



`Windows 10 pro` , `Windows 10 home`

- CPU- 가상환경 사용 설정  (작업관리자 > 성능 > 가상화 : 사용이 되어야함)

아래 링크타고 UEFI 펌웨어 설정 들어가면 자동 재부팅됨. 그때 나는 cpu configuration에서 virtualizatino Technology가 있었음. 거기 Enabled 로 체크하고 F10 눌러서 저장하고 나오기

[관련링크](https://support.bluestacks.com/hc/ko/articles/115003910391-%EB%82%B4-PC%EC%97%90%EC%84%9C-%EA%B0%80%EC%83%81%ED%99%94-VT-%EB%A5%BC-%ED%99%9C%EC%84%B1%ED%99%94%ED%95%98%EB%A0%A4%EB%A9%B4-%EC%96%B4%EB%96%BB%EA%B2%8C%ED%95%A9%EB%8B%88%EA%B9%8C-)

- [참고 링크](https://steemit.com/kr/@mystarlight/docker)
- 윈도우 10 프로 이상은 다음링크로 다운로드(541MB) : [링크](https://hub.docker.com/editions/community/docker-ce-desktop-windows) (로그인 해야함)
- 실행파일 클릭 전 Hyper-V 설치 : 제어판 > 프로그램 > Windows 기능 켜기/끄기 > Hyper-V 클릭하고 확인 > 재부팅
- Docker for Windows Installer.exe 실행해서 클릭 (완료되면 logout 버튼 클릭, 로그아웃되고 다시 로그인)
- GUI- Kitematic 부분은 [다음 링크 참고](https://blog.hanumoka.net/2018/04/28/docker-20180428-windows10pro-install-docker/)





## 1. 도커 Tutorial

- [도커 관련 Overview](https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html?fbclid=IwAR0ZELbxakvfey3bO1yDiLBSvpsT1QavGCpPBd5v50i3BeHVm_l67wgyx2I) 

- [Docker Docs](https://docs.docker.com/get-started/) 의 Part1 ~ Part3 까지 따라해봄
- 영어가 익숙치 않다면 [백기선님의 유튜브 영상](https://www.youtube.com/watch?v=9tW0QSsrhwc)을 보며 따라해보기 (추천)



## 2. 도커명령어 정리

Windows power shell 또는 cmd 에서 docker 를 붙여서 입력하면된다.

docker 치면 명령어들 쭉 나옴.

아래는 inflearn, [Ralf Yang 님의 Docker강의](https://www.inflearn.com/course/devops-docker-hands-on/)를 듣고 정리한 내용

`명령어`

- pull : 도커 이미지 다운로드
- images: 도커 이미지 list 보기
- run: 도커 적재 (stop id 하면 ps stop)
- ps: 도커 프로세스 상태 확인
- attach: 정지된 도커 프로세스 탑제
- logs: 도커 프로세스의 로그 확인
- tag: 도커 이미지 변경
- commit: 도커 프로세스를 스냅샷 형태로 이미지로 저장

`실습`

- docker login을 먼저 한다 username( not ID), PW 입력하면 Login Succeeded
- docker pull busybox
- docker run -it busybox sh (busy box 이미지 안에 들어온거)
- exit 하면 detach되서 docker ps. process 없음 ( docker ps -a 모든 프로세스 돌아갔던것까지 보여줌)
- 다시 실행
- docker start 0fd(container ID 앞부분) 하면 다시 pr 실행됨 
- docker attach 0fd (그 프로세스 다시 attach하면서 들어감)
- history (busybox에서 내가 실행했던 명령어 lis)
- ctl + p, ctl+q will be  suspend mode
- img 그대로 있음
- docker commit 0fd(컨테이너아이디 앞부분) test(이미지 이름):0.1(tag부분)
- docker images 보면 test 가 만들어진걸 볼 수 있다.  
- docker tag test:0.1 test:latest
- commit 부분은 기존 이미지에 조금 작업해서 새로운 이미지 만들어 테스트할때.

`명령어`

- login : 도커 이미지 Push를 위해 저장소에 로그인 [관련 이슈](https://stackoverflow.com/questions/41984399/denied-requested-access-to-the-resource-is-denied-docker)

- push: 도커 저장소(registry)에 image upload
- tag 로 push할 이미지를 <Docker Hub 사용자 계정>/<이미지 이름>:<태그> 로 만들고
- docker push <Docker Hub 사용자 계정>/<이미지 이름>:<태그>
- rmi: 시스템에 저장된 도커 이미지 삭제 , images id로 (rm) 

> docker stop $(docker ps -a -q)
>
> docker rm $(docker ps -a -q)
>
> 구동중인 모든 컨테이너들을 중지시키고 삭제
>
> 그리고 도커 이미지 삭제
>
> docker rmi $(docker images -q) -> 원하는 이미지

- save: 도커 이미지를 파일로 저장
- load: 파일로 저장된 도커 이미지 시스템에 저장
- prune: 도커 데몬이 사용하는 시스템 purge
- cp: Container 내부의 파일을 Host로 복사

`실습`

- docker save specia1tw/test > ./test.tgz 

- docker save, load 할 때 https://forums.docker.com/t/invalid-tar-header-on-docker-load/22256/2>

- > docker save image -o tmp.tar
  > docker load -i tmp.tar

- save는 망분리 환경. 다 검증이 된 이미지를 save 해서 파일단위로 delivery하고 docker를 관리해서 배포할 수 있는경우. 혹은 contest

- docker cp imgId:/필요한부분 저장할경로

- docker network prune ( network )

- docker volume prune



## 3. 카브환경에 맞게 Docker Image 만들기

- 도커 파일 만들기 : 다음 [Dockerfile](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/docker-inception-3-4/Dockerfile) 참고
  - (ubuntu + tensorflow-gpu + openslide + 필요 packages)

> - Dockerfile
> - inference.sh
> - src
>   - inference.py
>   - train.py
> - train.sh

- windows 쉘파일 작업할때 엔터가 `'\r'`이 추가로 들어감  `train.sh`, `inference.sh` 에  `'\r'`이 들어간 걸 하나씩 빼줬었음 -> 성공 

- 프린트 시 한글 못 찍어 에러 남. (변환 코드를 작성하던가 아니면 한글 빼야함)

- 기본적으로 image 빌드하는데 시간이 너무 오래 걸림. 기존의 빌드했던 부분에서 python 파일만 바꾸는 방법은 있을텐데 찾아봐야겠다. -> `commit 이용`

  >- 새로운 이미지 run 으로 들어가서 > 
  >- apt-get install vim : vim 설치하고 
  >- vim train.py 등으로 수정할거 수정하고 (sh 이나 py 같은 파일 수정시 반드시 :wq 하고 나오기)
  >- ctrl + p , ctrl + q로 나오고 > 
  >- docker ps 보고 
  >- container에서 최신작업한거 commit 하면 끝.

- 위의 부분도 vim 사용해 python file 편집하는데 불편한 점이 많아 다음과 같은 방법 찾음 **(by 대영님)**

  > 일단 도커 실행시킨 다음 `ctrl+p,q`로 빠져 나오고
  > `docker cp <file_name> <container_name>:/` 
  > `docker cp train.py my_container:/`으로 했더니 자동으로 덮어씌움
  >
  > `docker commit <컨테이너아이디앞부분> <이미지명>`
  >
  > `docker commit 0fd inception:1`



대회 때 만들었떤 이미지들.. 사실 이보다 훨씬 많다.. 올린 이미지만 290개니까.. ㅎㅎ

![di](https://user-images.githubusercontent.com/24144491/54084749-fe696500-4377-11e9-9beb-8b3227607b5c.png)



## 4. 그 밖에..

- windows 10 pro : docker 저장경로 바꾸기

중간에 도커 용량 문제로 외장하드로 옮겨야겠다는 생각이 듦

https://kiros33.blog.me/220351298695 mac이라서 패스 (pass)

도커 데스크탑 icon 우클릭해서 settings > advanced 에 있네 적용이 늦으니까 조금 (failed) 

https://github.com/docker/for-win/issues/1589 : **nategraf** comments 참고 (failed)

https://stackoverflow.com/questions/40465979/change-docker-native-images-location-on-windows-10-pro : 이대로 해보자 (succeeded!)

> - Hyper-V 매니저 들어가서
> - 오른쪽 바에 Hyper-V 설정 , D드라이브로 설정하고
> - 실행시키고나서도 오른쪽 MobyLinuxVM 설정에 스마트 페이징 경로도 D 드라이브로 바꿔줌
> - 다시 시작 시켜줘야함

**근데 가능한 경로는 바꾸지 않는편이 나음**. SSD가 아니라 이미지 save 할 때 더 느림.



- 가능한 Docker Hub에 있는 이미지를 활용하자!

> 그걸 몰라서 Ubuntu 깔고, Python 깔고 Tensorflow-gpu 깔고, cuda 깔고 ...  또 Openslide깔고.. 해야할 게 너무 많아서 하나씩 해보면서 너무 깔아야할게 많은데 하면서 좌절했었는데, 생각해보니 tensorflow gpu openslide 관련된 이미지가 있었네 ㅎㅎ 정리하면서 생각났다.. 