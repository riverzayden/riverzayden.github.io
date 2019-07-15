---
layout: post
title: "[논문] YOLO9000: Better, Faster, Stronger 분석"
category: paper
tags: dl paper objectDetection
comments: true
img: ssd1.jpg
---

# YOLO9000: Better, Faster, Stronger
---

![01](https://user-images.githubusercontent.com/24144491/48917028-2071d100-eec8-11e8-9824-ccc3a72c2168.png)

Image Detection의 Hot한 YOLO 알고리즘의 2번째 버전 **YOLO9000: Better, Faster, Stronger** 논문에 대한 분석을 해보았다. 앞서 [YOLO v1: You Only Look Once](https://taeu.github.io/paper/deeplearning-paper-yolo1-01/)과 [SSD: Single Shot Multibox Detector](https://taeu.github.io/paper/deeplearning-paper-ssd/)을 살펴봤으므로 이번 논문에서는 많은 부분들에 대한 설명들은 생략하고 깔끔하게 정리하되, 설명이 추가적으로 필요한 부분을 집중적으로 다루고자 한다. 따라서 읽다가 혹시 이해되지 않는 부분이 있다면 앞선 논문의 내용과 참고자료에 대한 블로그 글을 참고하길 바란다.(참고자료에서 잘못된 부분도 있음..)

# 0. Abstract
---

 이번 논문은 크게 3 부분으로 나뉜다. 

- Better : Accruacy , mAP 측면의 개선사항
- Faster : 속도 개선
- Stronger : 더 많은, 다양한 클래스 예측

그럼 이제 하나씩 살펴보자.


# 1. Better
---

![table2](https://user-images.githubusercontent.com/24144491/48917025-1fd93a80-eec8-11e8-8c73-8df1649d79d9.JPG)


**1-1. Batch normalization.** 모든 컨볼루션 네트워크에 배치 정규화 추가로 2%p mAP 향상.

**1-2. High resolution classifier.** 이미지넷 데이터로 앞단의 네트워크를 먼저 학습시키는데, 이때 448x448의 해상도로 10 epoch 동안 fine tuning함. 그리고 나머지 부분들을 디텍션하면서 tuning 시킨다. 고해상도로 학습된 앞단의 classification 네트워크 덕분에 4%p mAP 향상.

**1-3. Convolutional with Anchor boxes.** YOLO v1에서 사실 의문이기도 했던 detection을 위한 output을 연결시키는 마지막 레이어에서 Fully Connect 되었었는데, 이 fully connect를 제거하고 convolution으로 바꾸었다. 동시에 Anchor box도 적용했다고 한다. mAP는 0.3%p 만큼 줄어들었지만 recall이 81%에서 88%로 증가했다. FC를 convolutional connect로 바꿨으니 당연히 computation이 33%p 떨어졌다고한다. 

>사실 여기서 두개를 동시에 적용시켰기 때문에 어느 부분이 얼만큼의 효과를 내었는지는 모른다. 대략 FC -> convolutional는 computation을 많이 감소시켰을 거고, anchor box를 써서 recall 이 증가했지만 computation은 늘어났을거라 추측되지만 anchor box를 얼마나 잘 적용시켰는지는 나와있지 않으므로 물음표. 뭐 크게 중요하지 않을 수도 있는게 뒤에서 이 anchor box들을 잘 적용시키기위한 2가지 전략이 나오므로 뒤에서 더 살펴보도록 하자.


**1-4. Dimension clusters.** 

![f2](https://user-images.githubusercontent.com/24144491/48917029-2071d100-eec8-11e8-9965-7c2702e3b31c.JPG)

SSD나 Fast R-CNN의 경우 사람이 임의로 선정한 anchor box의 도움을 받아 predicted box를 구한다. 하지만 임의라는 말 역시 데이터의 특성에 따라 성능이 달라질 수 있고, 실제 실험할 때도 어느 정도의 anchor box의 비율을 두어야할지, 몇 개를 두어야 할지 애매해질 수 있다. YOLO v2에서는 이런 고민을 해결하기 위해 detection dataset에서 k-means 알고리즘을 활용해 이 하이퍼파라미터들을 설정했다. 이때 기존의 k-means에서 유클리디안 거리를 쓰지 않고 `d(box, centroid) = 1 - IOU(box,centroid)` 사용했다.

![t1](https://user-images.githubusercontent.com/24144491/48917022-1fd93a80-eec8-11e8-8eba-b813d3f55d6e.JPG)

9개의 앵커박스를 두었을때 그만큼 다양한 anchor box로 잘 예측할 수 있지만, 5개를 두었을 때랑 크게 차이가 나지 않는다고 판단하고(computation은 2배가 되는데에 비해) YOLO v2에서는 5개의 anchor box를 사용하기로 결정.


**1-5. Direct location prediction.**

![f3](https://user-images.githubusercontent.com/24144491/48917030-2071d100-eec8-11e8-8f53-a078c4974e7c.JPG)

이 부분은 SSD나 Fast R-CNN의 Loss fucntion 부분을 떠올리면 이해가 더 쉬울 것이다. anchor box를 예측하되 기존의 Predicted box의 중심좌표들이 초기 단계에 잘못 설정된다면 학습이 잘 안될 수 있다는 점을(예측해야되는 곳보다 멀어질 수 있음) 설명하고 있다. (SSD의 경우는 l, predict된 값이 default box로 normalize(?)된 값이라 상대적으로 잘 될 것) YOLO v2는 이런 값들이 범위를 크게 벗어나지 않고 적절히 학습시키게 하기 위해서 다음과 같은 설정을 한다.

![fomula](https://user-images.githubusercontent.com/24144491/48917020-1f40a400-eec8-11e8-9498-35d942de9989.JPG)

- cx, cy는 그리드 셀의 좌상든 끝 offset
- pw, ph는 prior(우선순위 앵커박스)의 width, height
- tx, ty, tw, th가 우리가 예측해야할 값들
- bx, by, bw, bh는 각 값들을 조정하여 실제 GT와 IOU를 계산할 최종 bounding box의 offset 값들

중심좌표는 시그모이드를 통해서 0으로 initialize되면 중심에 가게, 너비과 높이 역시 0으로 초기화 했을때 prior, anchor box의 값들에서 시작될 수 있게 한 조치이다. 이렇게 하면 안정적으로 학습이 잘 될 것.

최종적으로 1-4, 1-5번을 통해서 약 5%p mAP 향상.


**1-6. Fine-grained features.** 이 부분 역시 [SSD](https://taeu.github.io/paper/deeplearning-paper-ssd/)의 multi-scale feature map과 엮어 생각하면 좋은데, 13x13은 큰 이미지를 검출하기엔 충분한 feature map이지만, 작은 물체를 detect하기에는 약간 충분하지 않을 수 있다. 따라서 26x26 layer 에서 그다음 conv하지 않고 26x26x512의 특징맵을 13x13x(512x4)로 변환한다음(26x26에서 중심을 기준으로 2x2로 나눈 네 조각을 concatenate) detection을 위한 output으로 이어준다. 여기서 1%의 성능향상이 있음.


**1-7. Multi-scale training.** 기본 사이즈는 416x416을 이용한다.( 마지막 부분이 홀수 grid를 갖게 하기 위해서) 거기에 추가로 한 네트워크로 다양한 해상도의 이미지도 잘 처리하게 만들기 위해 다양한 해상도를 골고루 학습시킨다. 10번의 배치마다 {320,352, ... , 608}로 resize된다. 


# 2. Faster
---

**2-1. Darknet.** 많은 Image Detection Model에서 classifier netowrk(앞단의 네트워크)로 VGG Net을 많이 쓴다. 하지만 VGG-16 은 30.69십억의 floating point 계산을 필요로 한다.(224x224 해상도 imgae의 경우) YOLO v2에서는 GoogleNet을 기반으로한 독자적인 **Darkent**을 만들어 30십억의 계산량을 8십억으로 줄였다. (accuracy는 88%로 VGG-16의 90% 성능과 크게 차이나지 않는다.)  

**2-2. Training for classification.** ImageNet 1000개 클래스 구별 데이터셋을 160 epoch동안 학습하면서 learning rate = 0.1, polynomial rate decay = a power of 4(4로 나눈다는 뜻일듯), weight decay = 0.0005, momnetum = 0.9, 처음 튜닝은 224로 하다가 중간에 448로 fine tune. 

**2-3. Traininig for detection.** 5 bounding box로 5개의 좌표(Confidence score + coordinate)와 20개의 class 점수를 예측하므로 한 그리드 셀에서는 총 5x(5+20) = 125개의 예측값을 가지게 된다. 또 중간에 passthrough layer로부터(26x26) concatenate된 예측값도 포함. 160 epoch동안 10^(-3)에서 시작하여 10, 60, 90 에폭마다 decay하고, weight decay = 0.0005, momentum = 0.9를 사용했다. data augmentation 역시 random crops, color shifting등을 이용했다.

# 3. Stronger
---
이 부분은 imgae classification은 클래스가 몇천~몇만개정도로 많지만 detection의 라벨은 20~몇 백개 정도가 전부이다. 이런 갭을 완화하기위한 전략을 소개하는 부분이라고 보면된다.

- trining때 classification과 detection data를 섞어서씀
- data set에서 detection data가 들어오면 원래 loss function을 활용해 계산
- data set에서 classification data가 들어오면 loss function에서 classification loss만 활용해 계산

여기서 드는 의문점은, detection에는 '개'라고 되어있지만 classification에서는 '시츄', '비글', '푸들' 등과 같이 개 종류만 수백 종류가 있다. 따라서 라벨들을 일관성 있게 통합해야하고, 시츄를 개라고 했다고 해서 완전히 틀린 것은 아니므로 상호 배타적이지 않은 예시에 대해서 multi-label 모델을 사용한다.

**Hierarchical classification.** 이미지넷(ImageNet)의 라벨들은 워드넷(WordNet)의 구조에 따라 정리되어 있다. 워드넷의 구조를 보면, '노퍽 테리어나, 요크셔 테리어는' 가축 - 개과 - 개 - 사냥개 - 테리어의 hyponym 하위어 이다.([워드넷에 대한 설명 위키](https://ko.wikipedia.org/wiki/%EC%9B%8C%EB%93%9C%EB%84%B7)) 여기서 YOLO 역시 이미지넷의 컨셉인 계층적 트리 구조를 이용해 label을 공통적으로 묶는 작업을 한다. 계층 트리를 만들기 위해서 워드넷 그래프로 어느 경로로 나타내는지 찾고, 많은 관련어들이 하나의 경로를 가지는 경우가 많으므로 그런 것들부터 먼저 처리하고 트리를 늘려갔다. (개 아래 요크셔 테리어, 시츄, 등등 이런 단어들을 먼저 개로 묶고 그런 카테고리들을 여러개 붙인다는 의미) 루트에서 특정경로가 제일 가까운거부터 선택해 붙여나간다. 워드트리를 구축하기 위해 1000개의 클래스를 갖는 이미지넷 데이터를 이용해 Darknet-19 모델을 학습시켰다. 워드트리1k 를 만들기 위해서는 중간 노드들을 추가해야 했기 때문에 라벨의 갯수가 1000개에서 1369개로 들어났다. (예를 들어 이미지의 라벨이 "노르포크 테리어" 인 경우 워드넷에서 관련어(sysnets)인 "개", "포유동물" 등의 라벨 까지도 얻게 된다. 트리 중간 중간에 중복되는 라벨이 생기고 "포유동물" 등의 포괄 개념을 갖는 라벨이 생기기 때문에 369개만큼 노드가 늘어난다.)

![wordtree](https://user-images.githubusercontent.com/24144491/48917026-1fd93a80-eec8-11e8-826d-b895a5070821.jpg)


이렇게 word tree가 만들어졌다고 한다면, 우리가 어떤 특정 노드를 예측할때는 조건부 확률을 쓰게된다. 예를들어 bald eagle을 예측해야하는 경우

> Pr(bald eagle) = Pr(bald eagleㅣeagle) x Pr(eagleㅣbird) x Pr(birdㅣanimal)
> (Pr(animal) = 1이라 가정)

이제 이 식이 이해가 될 것이다.

![fomula2](https://user-images.githubusercontent.com/24144491/48917021-1f40a400-eec8-11e8-8d1b-67589f1d0d35.JPG)

학습과정에서는 실제(ground truth) 라벨부터 트리 루트까지 모든 상위의 값들을 업데이트한다(다 loss를 계산). 예를 들어 이미지의 라벨이 "노르포크 테리어" 인 경우 워드넷에서 관련어(sysnets)인 "개", "포유동물" 등의 라벨 까지도 얻게 된다. 조건부 확률을 계산하기 위해서 모델은 1369개의 값을 갖는 벡터를 예측.

![f5](https://user-images.githubusercontent.com/24144491/48917018-1f40a400-eec8-11e8-9ada-94878531cbea.JPG)

같은 계층(word tree에서 같은 level에 있는)을 기준으로 softmax.


**Dataset combination with WordTree.**

![f6](https://user-images.githubusercontent.com/24144491/48917019-1f40a400-eec8-11e8-8e2e-2428d4a482ff.JPG)


이미지넷과 코코데이터라벨들을 합쳐 WordTree를 만들었다.

**Joint classification and Detection.**

위에서 Top 9000개의 클래스를 학습시키고자 했고, 이미지넷 데이터셋 : 코코 데이터셋 = 4 : 1로 조정했다. 아웃풋 사이즈 문제로 5개의 bounding box에서 3개의 bounding box를 예측하게 조정한다.

**back prop.** 제일 처음 언급했던 것처럼

- trining때 classification과 detection data를 섞어서씀
- data set에서 detection data가 들어오면 원래 loss function을 활용해 계산
- data set에서 classification data가 들어오면 loss function에서 classification loss만 활용해 계산
- classification loss 계산시 ground truth label의 그 계층이나 상위계층만 backprop하고 아래 계층에 내려가지 못하도록 아래 계층에 대한 예측할 시 error 부과.

검출 성능이 19.7% mAP, 학습과정에서 전혀 본적이 없는 156개의 클래스를 포함하면 16.0% mAP. 그래도 다른 DPM(object detection with discriminatively trained part based models)보다는 좋은 성능.

# 끝으로

YOLO v2는 YOLO v1, Fast R-CNN, SSD의 문제점들을 해결하면서 Real-time에 구현하기위해 다양한 전략들을 짰다. 끝부분의 Stronger 부분은 이제 다양한 detection을 하기위한 새로운 시도는 창의적이다. 아직 많은 클래스에 대해 성능이 잘 나오진 않지만 detection label 데이터가 그만큼 많이 받쳐준다면 학습역시 잘 될 것일테고, 이 부분을 좀 더 빨리 해결하기위해 segmentation model을 함께 적용하면 어떨까라는 생각이든다. 어찌됐건 YOLO v2의 한계점은.. 부수적이고 작은거 제외하곤(사실 Real time, 속도까지 고려하면 당연 trade-off라..) 발견하지 못했다. (하루만에 리뷰하는거라 급해서 안보였을 수도 있지만) 요약본은 YOLO v3까지 요약한다음에 다른 Detection Algorithm을 같이 요약한 글로 대체하는게 좋을 것 같다는 생각이 든다.


# 참고자료
[논문](https://arxiv.org/abs/1612.08242)
[참고 블로그](https://m.blog.naver.com/sogangori/221011203855)
[recall 및 mAP 관련 이해 자료](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)
