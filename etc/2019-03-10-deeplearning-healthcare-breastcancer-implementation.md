---
layout: post
title: "[HC] 유방암 진단 딥러닝 모델 구현"
category: healthcare
tags: dl vision breastCancer healthcare medical
comments: true
img: medical2.jpg 
---



# 유방암 진단 딥러닝 모델 구현

본 글은 서울 아산병원과 카카오브레인에서 주최하는 HeLP Challenge 의 2-2. Breast cancer Metastasis Detection 의 모델을 구현하기 위해 고려했던 사항들이다.

![ni](https://user-images.githubusercontent.com/24144491/54084761-26f15f00-4378-11e9-9c43-151b88dd1cde.png)





# 1. Whole Process Sketch

Breast Cancer Metastasis Detection Model의 전반적인 스케치는 다음과 같다.

1.  Preprocessing - tissue patches sampling
2.  Train Data Generate
3.  Network Define
4.  Train
5.  Predict



다음은 각 과정에서 고려했던 혹은 고려해야할 세부사항들이다.



## [1] Preprocessing - tissue patches sampling

**[1-1] Sampling 전 슬라이드 데이터 분석**

> [train.py](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/docker-inception-3-4/train.py)에서 170번째 줄 ~ 203번째 줄까지

- data analysis python 을 작성해 docker 이미지 올린 결과 아래와 같음
  - num of slide = 157
   - total num of patches(patch size = 256, all tumor 기준) = 4,000,000
   - total num of tumor patches =  600,000 
   - non tumor : tumor = 5.8 : 1 

- sampling
  -  1 : 1 = positive slide patches : negative slide patches
  -  1 : 1 : 2 = positive tumor patches : positive non-tumor patches : negative patches



**[1-2] Training patches** 

> [model.py](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/docker-inception-3-4/model.py) 에서  ```find_patches_from_slide()``` 

- tissue check 
  - grey 변환 검정색 걸러내고
  - ostu method 통해 남은 흰색 배경 걸러냄
- tumor check
  - positive 경우에만 고려
  - mask가 이미 1/16 이 되었기 때문에 mask는 patch_size/16 만큼만, 원 slide는 lv 0에서 patch_size 만큼 level down. (256 기준 slide는 level 8, truth_mask 는 level 4)
  - all tumor 만 고려. mask된 비율이 차이가나므로 원본 데이터와 비교했을 때 정확하게 masking 되지 않는 문제 발생. 따라서 all tumor(모든 영역이 tumor 인 patch만 고려해줘야함)
  - 256 기준으로 약 약 25개의 slide의 tumor patch 개수가 100개 미만, 그 중 10개정도는 0개. 이 부분에 대해서는 128 size로 patch를 조정해 tumor patch를 더 확보하는 방안도 있음 

- 각 기록된 것의 loc 가져 오기
- return all_tissue_samples



## [2] Train Data Generate

> [train.py](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/docker-inception-3-4/train.py)에서 ```gen_imgs()```

- X: (batch_size, patch_size, patch_size, 3)
- Y: (batch_size, patch_size, patch_size, 2)
- Data augmentation 
  - rotation
  - brightness
  - horizontal, vertical flip
- yield 로 메모리 overhead 방지



## [3] Network

> [model.py](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/docker-inception-3-4/model.py)에 구현된 ```inceptionV3```, ```simple```, ```unet``` 참고

- simple U-Net (no skip-connection)

- Inception-v3 (Google paper 17)

  

## [4] Train

- num_samples = len(all_tissue_samples)
- batch_size = 64~128 (OOM check)
- train : validation = 8:2 or 9:1
- model save path : /data/model
- hyper parameter tuning : ```keras  callback```
- parameter size reduction : InceptionV3 에서 ```down_para``` 부분





## [5] Predict

> [inference.py](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/docker-inception-3-4/inference.py) 부분 참고

- 5000 patch samples from slide (정확도는 조금 낮추더라도 빨리 inference하기위해)
- in each patch, find max value in 128 x 128, center area of patch (google 17 paper)
- obtain max value of them (5000 samples)
- ensemble simple model and InceptionV3 model





# 2. Things We Tried Didn’t Work



- data augmentation

  - color
  - hue
  - contrast 

  : using PIL Image.enhace 였던가

  - elastic deformation

  : 몇 번 hyperparameter 바꿔봤지만 잘 수렴이 안되길래 시간 상 pass 

- patch_size 

  - 512 x 512 (너무 느리게 학습돼서 포기) 
  - [model_inception_1-512.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/model_inception_1-512.ipynb), [model_simple_512.ipynb](https://github.com/Taeu/HeLP-Challenge-Goldenpass/blob/master/model_simple_512.ipynb).

- flow_from_directory

  > [docker-simple-final](https://github.com/Taeu/HeLP-Challenge-Goldenpass/tree/master/docker-simple-final)

  - 학습 속도를 높이기 위해, sampling한 이미지들을 저장하고 directory에서 학습하는 방식을 적용
  - 다른 조건은 동일한데 메모리상에 올려놓고 할 때와 달리 loss가 그 만큼 빨리 안 떨어짐..
  - 구현했을 당시 시간이 4일정도밖에 안 남아서 더 개선 안하고 원래 잘 됐던 모델로 training

- 잘못 sampling 한 model

  > [docker-simple-3](https://github.com/Taeu/HeLP-Challenge-Goldenpass/tree/master/docker-simple-3)

  - 학습 속도 개선위해 4개 슬라이드씩 묶어서 미리 열어두고 학습을 진행했는데, 마지막에 슬라이드 4개에 overfitting 될 가능성 있어서 변경함













​    	
