
---
layout: post

title:  "[linux] Ubuntu18.04 git setting "

subtitle:   "[linux] Ubuntu18.04 git setting "

categories: linux

tags: docker ubuntu18.04 git commit

comments: true

img: 


---




### git 세팅 사용법

```sh
sudo apt-get install git-core
sudo git config --global user.name "본인 계정 입력"

sudo git config --global user.email "본인 메일 주소 입력"

sudo git config --global color.ui "auto"

```


### git 사용해보기
```
sudo git clone 자신의git레퍼지토리 

git init

sudo git remote add origin 자신의git레퍼지토리 

sudo git fetch origin

cd 자신의git레퍼지토리 

sudo git add -A

sudo git commit -m "system change"

git push -u origin master 
```