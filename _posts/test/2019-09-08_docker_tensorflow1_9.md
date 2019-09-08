```
https://devyurim.github.io/python/tensorflow/development%20enviroment/docker/2018/05/25/tensorflow-3.html
```

```shell
nvidia-docker run -it \
-p 8888:8888 -p 6006:6006 \
--name ailab-yurim \
-v /home/zayden/Desktop/test_garbage:/notebooks \
-e PASSWORD="0000" \
--restart always \
tensorflow/tensorflow:1.8.0-gpu-py3
```

```
docker commit tf2 eungbean/tf2:latest
```

docker ps -a  

docker stop eade3e95f777



```
docker commit ailab-yurim riverzayden/ailab-yurim:0.2
```



nvidia-docker run -it \
-p 8888:8888 -p 6006:6006 \
--name ailab-yurim \
-v /home/zayden/Desktop/test_garbage:/notebooks \
-e PASSWORD="0000" \
--restart always \
riverzayden/ailab-yurim:0.1



```
docker login
docker tag local-image:tagname new-repo:tagname
docker tag riverzayden/ailab-yurim:0.1
docker push riverzayden/ailab-yurim:0.1

docker pull riverzayden/ailab-yurim:0.1


```



## pgadmin 

https://judo0179.tistory.com/48

```
docker pull postgres
docker run -d -p 5432:5432 --name pgsql -e POSTGRES_PASSWORD=mypassword postgres
docker volume create pgdata

docker run -d -p 5432:5432 --name pgsql -it --rm -v pgdata:/var/lib/postgresql/data -e POSTGRES_PASSWORD=mypassword postgres



docker exec -it pgsql bash

root@cb9222b1f718:/# psql -U postgres
psql (10.3 (Debian 10.3-1.pgdg90+1))
Type "help" for help.
postgres=# CREATE DATABASE mytestdb;
CREATE DATABASE
postgres=#\q


```





## git 사용법

```sh
sudo apt-get install git-core
sudo git config --global user.name "본인 계정 입력"

$ sudo git config --global user.email "본인 메일 주소 입력"

$ sudo git config --global color.ui "auto"



출처: https://emong.tistory.com/228 [에몽이]
```

