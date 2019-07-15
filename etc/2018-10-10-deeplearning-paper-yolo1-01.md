---
layout: post
title: "[논문] YOLO v1 분석 - 01" 
subtitle: "논문, paper, yolo, yolo-v1"
category: paper
tags: dl paper objectDetection
comments: true
img: yolo-v1.jpg
---

# YOLO v1 : You Only Look Once
---

![1](https://user-images.githubusercontent.com/24144491/46718830-6c194400-cca7-11e8-8c5f-5fd3f61c123e.png)

**Image Detection**의 hot한 여정의 시작인 **YOLO v1** 논문을 리뷰하고자 한다. 이미 YOLO에 대한 많은 리뷰 자료들이 있지만 일부 오류들과 전체적인 과정을 순차적으로 자세히 설명한 자료는 없었다. 딥러닝의 전체적인 프로세스를 순차적으로 따라가는 것은 논문 이해와 더불어 코드를 이해하는 가장 정확하고 쉬운 길이다. 따라서 YOLO 논문을 딥러닝 모델 구성부터 학습 및 테스트까지 전체적인 프로세스를 순차적으로 정리하는 것이 의미가 있을 것이라 판단하여 이 글을 작성하게 되었다. 구체적인 YOLO v1 리뷰 글의 목차는 다음과 같다. 

![2](https://user-images.githubusercontent.com/24144491/46718831-6cb1da80-cca7-11e8-866b-56e8fb3fac13.png)

---
- **[0. Background 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-01/)**
- **[1. Introduction 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-01/)**
- **[2. Unified Detection 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-02/)**
- **[3. Conclusion 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-03/)**
---


가독성을 위해 각 파트별 글을 따로 포스팅했다. 이번 글에서 다룰 파트는 **0. Background**와 **1. Introduction**이다.


## 0. Background
---

YOLO 리뷰에 앞서 Image Detection이 왜 새로운 연구주제로 떠올랐는지부터 살펴볼 필요가 있다.


![4](https://user-images.githubusercontent.com/24144491/46718832-6cb1da80-cca7-11e8-8529-4509955b89ec.png)

Image Classification 은 2012년도 AlexNet의 등장으로 많은 사람들이 주목하기 시작했다. 2014년도 GoogleNet과 VGGNet등장으로 이 부분에 대한 연구는 가속화 되었다. 2015년도 ResNet의 등장으로 딥러닝이 드디어 인간의 판단능력보다 더 우위에 설 수 있게 되었다. 이제 사람들의 관심사는 Image Classification에서 특정 class가 Image에 어디에 있는지 예측하는 Localization 문제를 풀고자 노력한다. 그리고 한 단계 더 나아가 이미지 안에 여러가지의 물체가 어디에 있는지와 그 물체는 어떤 종류의 물체인지를 판단하는 **Image Detection**에 대한 연구가 활발히 진행되고 있다.

![5](https://user-images.githubusercontent.com/24144491/46718833-6cb1da80-cca7-11e8-872c-dc05304e2b61.png)

2014년도부터 2018년도(현재)까지 Image Detection과 관련된 논문을 시간순으로 [정리한 그림](https://github.com/hoya012/deep_learning_object_detection)이다. 빨간색으로 표시된 논문은 그 중에서도 핵심적인 것들이라 할 수 있다. 이 논문들을 크게 2가지로 분류할 수 있는데,

![6](https://user-images.githubusercontent.com/24144491/46718835-6d4a7100-cca7-11e8-996c-be10edaab670.png)

one stage method란 한 번만에 image detection을 할 수 있는 알고리즘들이고 two stage method는 2번 만에 image detection을 완료하는 알고리즘이다. 우리가 Review할 **YOLO**와 SSD 의 알고리즘은 one stage method에 해당하고, R-CNN 과 관련된 알고리즘들은 two stage method이다.

![7](https://user-images.githubusercontent.com/24144491/46718836-6d4a7100-cca7-11e8-80da-83fa27896b04.png)

위는 **YOLO v3**의 논문에서 가져온 표와 그래프인데, **YOLO v1**도 그랬듯 최신 버전의 **YOLO v3** 역시 성능은 비슷하게 나오면서 제일 빠른 속도를 자랑한다.

이제 YOLO가 어떻게 나오게 되었는지 알아보자!

![8](https://user-images.githubusercontent.com/24144491/46718837-6d4a7100-cca7-11e8-93bc-f42a8a3b0183.png)


## 1. Introduction
---

![9](https://user-images.githubusercontent.com/24144491/46718838-6de30780-cca7-11e8-8532-270ccc801421.png)

**Single Neural Network.** 우리가 어떻게 사물을 판단하는지 잠깐 생각해보자. 각자 볼 수 있는 시야 범위 안에서 어떤 종류의 물체가 어디에 있는지 바로 판단한다. 이처럼 YOLO는 인간의 시각체계와 비슷하게 작동하게끔 모델을 **single neural network**로 구성했다. 물체를 싸고 있는 bounding box와 그 박스 안에 물체의 종류를 동시에 예측하는 **Regression**문제로 Image Detection을 풀고자 한다. 이런 일련의 과정을 하나의 그림으로 잘 표현한 것이 위의 그림에서 아래 부분에 있는 그림이다. 어떤 input image가 있으면, 하나의 신경망을 통과하여 물체의 bounding box와 class를 동시에 예측하게 된다.

![10](https://user-images.githubusercontent.com/24144491/46718829-6c194400-cca7-11e8-928f-a61bbd383837.png)

**Simple is fast.** YOLO의 장점을 정리하자면 다음과 같다.

- (1) 빠르다
- (2) 다른 알고리즘과 비슷한 정확도를 가진다.
- (3) 다른 도메인에서 좋은 성능을 보인다.


# Next.
---

YOLO가 구체적으로 어떤 네트워크를 가지고 어떤 loss function을 설정하여 Loss를 최적화하는 학습과정을 거치는지에 대해서는 [다음 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-02/)에서 구체적으로 살펴볼 것이다.


## 참고자료
---

- [object detection paper list](https://github.com/hoya012/deep_learning_object_detection)
- [yolo v3 paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- [yolo site](https://pjreddie.com/darknet/yolo/)










