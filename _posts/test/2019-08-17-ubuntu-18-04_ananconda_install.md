#### Anaconda  Install ( Ubuntu18.04)

1. Download 

https://www.anaconda.com/distribution/#download-section



2. 권한 설정 및 설치

```sh
chmod +x Anaconda3-2019.07-Linux-x86_64.sh
./Anaconda3-2019.07-Linux-x86_64.sh
```



3. zsh쉘이라면   ( vim ~/.zshrc  추가 )

```sh
export PATH=$HOME/anaconda3/bin:$PATH
```

```sh
source ~/.zshrc
```





4. 가상환경 생성 

```sh
conda create -n python36_serving python=3.6 --y
```



5. 가상환경 접속

```sh
source activate python36_serving
```



6. 가상환경 정보 확인

```sh
conda info --envs
conda env list
```



7. 가상환경 삭제

```sh
conda remove -n python36_serving --all
```



8. 가상환경 패키지 설치 리스트

```sh
conda list
```

