---
layout: post

title:  "[linux] Docker Postgres 셋팅  "

subtitle:   "[linux] Docker Postgres 셋팅 "

categories: linux

tags: docker ubuntu18.04 Postgres

comments: true

img: 
---




## Docker Postgresql 

#### 출처
```html
https://judo0179.tistory.com/48

```

### Postgresql 받아오기 
```sh
docker pull postgres
```

### Postgresql 실행해보기
```sh
docker run -d -p 5432:5432 --name pgsql -e POSTGRES_PASSWORD=mypassword postgres
```

### Postgresql 마운트시키고 실행해보기 ( 추후에 껏다켜도 테이블들이 남아있게 하기 위함.)
```sh
docker volume create pgdata

docker run -d -p 5432:5432 --name pgsql -it --rm -v pgdata:/var/lib/postgresql/data -e POSTGRES_PASSWORD=mypassword postgres
```



### Postgresql shell 접속해보기 
```sh
docker exec -it pgsql bash

root@cb9222b1f718:/# psql -U postgres
psql (10.3 (Debian 10.3-1.pgdg90+1))
Type "help" for help.
postgres=# CREATE DATABASE mytestdb;
CREATE DATABASE
postgres=#\q

```


