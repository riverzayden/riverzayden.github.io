---

layout: post
title: "[논문] YOLOv3: An Incremental Improvement 분석"
category: paper
tags: dl paper objectDetection
comments: true
img: ssd1.jpg
---


# YOLOv3: An Incremental Improvement


<iframe width="1280" height="720" src="https://www.youtube.com/embed/MPU2HistivI" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


저자는 이번 해에 연구할 시간이 없었다고 한다. 그래서 이번 논문은 **`TECH REPORT`** 느낌으로 기술, 다른 Detection Algorithm들의 방법들을 가져와 도움을 얻은 것들 혹은 그러지 못했던 것들을 나열하고 있다. 따라서 나도 간략하게 내용을 정리할 생각이다. YOLOv3을 이해하기 위해서는 [YOLOv1](https://taeu.github.io/paper/deeplearning-paper-yolo1-01/),   [SSD: Single Shot Multibox Detector](https://taeu.github.io/paper/deeplearning-paper-ssd/),   [YOLO9000: Better, Faster, Stronger](https://taeu.github.io/paper/deeplearning-paper-yolov2/) 글들을 보고 오는 것이 좋을 것 같다.

![1](https://user-images.githubusercontent.com/24144491/48928079-f4356f00-ef1e-11e8-890e-f9177c9d0f37.png)


# 1. The Deal
---

## 1-1. Bounding Box Prediction

![fomula](https://user-images.githubusercontent.com/24144491/48928083-f4ce0580-ef1e-11e8-83d8-ca7879e75beb.JPG)

![f3](https://user-images.githubusercontent.com/24144491/48928082-f4ce0580-ef1e-11e8-8044-a8f925f88a65.JPG)

YOLO9000 - YOLO v2에서도 소개되었듯이, YOLO는 anchor box(yolo 에서는 prior box)를 도입한다. 위와 같은 식을 통해서 bx, by는 [0,1] 값들로, bw, bh는 초기화된 값이 prior box부터 시작할 수 있게해 학습을 안정적으로 만들었다. loss를 계산할때도 ground truth의 cx,cy, w,h와 바로 비교할 수 있게 했으므로 loss를 계산하는 과정도 간단하다.

거기에 추가로 YOLOv3에서는 다른 Detection Algorithms의 Matching Strategy를 가져왔다. YOLO v1과 다르게 각각의 바운딩박스마다 objectness score(그 바운딩박스에 물체가 있는지 없는지)를 예측하고, 이때 prior box(anchor box)와 ground truth box의 IOU가 가장 높은 박스를 1로 두어 매칭시켰다.(loss를 prior box와 같은 index, 위치를 가지는 predicted box의 offset만 계산해주겠다는 의미이고 SSD와 다른 알고리즘과는 다르게 Best IOU에 대해서만 1값을 가지게했다. 나머지는 무시한다.)

## 1-2. Class Prediction

multi-label이 있을 수 있으므로 class prediction으로 softmax를 쓰지 않고 independent logistic classifiers를 썼다. 따라서 loss term도 binary cross-entropy로 바꾸었다. 이는 좀 더 복잡한 데이터셋(Open Image Dataset)을 학습하는데 도움이 되었다.(multi-label을 예측할 때 좋았다고 한다)

## 1-3. Predictions Accross Scales

- 3개의 bounding box
- 3개의 feature map 활용 (다른 scale, 각각 2배씩 차이)
- 한 feature map에서의 output 형태는 Grid x Grid x (#bb *(offset + objectiveness + class)) = NxNx(3x(4+1+80))
- 총 9개(3개 바운딩박스 x 3개 피쳐맵)의 anchor box는 k-means clustering을 통해 결정
- (10x13), (16x30), (33x23), (30x61), (62x45), (59x119), (116x90), (156x198), (373x326)


## 1-4. Feature Extractor

Darknet-19 -> Darknet-53으로 변경, 모델과 성능은 아래와 같음

![2](https://user-images.githubusercontent.com/24144491/48928080-f4356f00-ef1e-11e8-9467-40dee0e31bf6.JPG)

![3](https://user-images.githubusercontent.com/24144491/48928081-f4356f00-ef1e-11e8-9eb2-9cf92f9be147.JPG)

- 성능 이정도면 SotA

## 1-5. Training

- full image 사용
- no hard negative mining(IOU 낮은값들: Best IOU 값 비율 조정 안해주고 그냥 학습)
- multi-scale training
- data augmentation
- batch normalization
- all the standard stuff...

그래서 성능은 다음과 같이 나왔다.

![result](https://user-images.githubusercontent.com/24144491/48928089-f5ff3280-ef1e-11e8-8d37-1d8846353f74.JPG)


# 2. Things We Tried Didn't Work
---

- Anchor box x,y offset predictions. 다른 알고리즘들이 하는 일반적인 anchor box 와 linear activation을 이용했더니 잘 안됨.
- Linear x,y predictions instead of logistic. x,y offset을 예측하는데 linear activation을 이용했는데 잘 안 나옴.
- Focal loss. mAP 2%p 감소시킴. 이미 우리는 objectness score과 conditional class prediction을 잘 구분해서 견고한 모델을 만들었기 때문.
- Dual IOU thresholds and truth assignmnet. Faster R-CNN의 경우 학습중 2가지 IOU threshold를 지정했지만 적용하니 좋은 결과는 안 나옴.


# 3. What this All Means
---

위의 결과 테이블을 다시 보자.

![result](https://user-images.githubusercontent.com/24144491/48928089-f5ff3280-ef1e-11e8-8d37-1d8846353f74.JPG)


누군가 이것을 본다면 '음?? YOLO v3 별로 안 좋은데??' 라고 생각할 수 있다. 그런데 Russakovsky et al 에 따르면 사람도 IOU .3과 IOU.5를 구분하는데 어려움을 느낀다고. 하지만 mAP를 측정하는데 작은 IOU metric도 들어가 성능이 안좋게 보이게 된 것. 저자는 old metric에서 지금 metric으로 바꾼 이유에 대해 지적하며, 굳이 바뀐 metric에 맞춰 다시 개선할 필요가 있을까라는 의문점을 던진다. ('How much does it matter?') 다음 부분으로 저자는 이 글을 마무리하는데 굳이 번역하는 것보다 원글을 살리는게 낫다는 생각이 들어 그대로 가져왔다.

![last1](https://user-images.githubusercontent.com/24144491/48928085-f5669c00-ef1e-11e8-8b64-cd3157586ef5.JPG)

![last2](https://user-images.githubusercontent.com/24144491/48928088-f5669c00-ef1e-11e8-9d62-d8cbe92c1d30.JPG)


# 4. 끝으로

Image Detection에서 세심한 부분까지 다 건드리는 건 역시 YOLO 뿐이라는 생각이 든다. 연구도, 결과를 낼때도 되는지 안되는지 하나씩 다 체크해가면서 실험해본 것 같고. 이제 Image Detection의 성능을 끌어올리기 위한 데이터 작업, 그리고 그 작업을 수월하게 해줄 알고리즘을 찾던가 혹은 어떤 다른 알고리즘과 결합하여 참신한 방법의 Image Detection 알고리즘이 나오겠지. 그래도 2018년도까지의 Image Detection의 큰 흐름은 어느정도 다 살펴본 느낌이다. 정리하는 글을 오늘 밤에 올려야겠다. (Retina Net과 R-CNN 계열은 훑어만 봐서 그걸로 정리해도 되나 싶지만)


# 참고자료

- [논문](https://pjreddie.com/media/files/papers/YOLOv3.pdf)


