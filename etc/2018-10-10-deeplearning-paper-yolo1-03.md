---
layout: post
title: "[논문] YOLO v1 분석 - 03" 
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

___
- **[0. Background 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-01/)**
- **[1. Introduction 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-01/)**
- **[2. Unified Detection 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-02/)**
- **[3. Conclusion 글](https://taeu.github.io/paper/deeplearning-paper-yolo1-03/)**


___

가독성을 위해 각 파트별 글을 따로 포스팅했다. 이번 글에서 다룰 파트는 마지막 파트인 **3. Conclusion**이다.


# 3. Conclusion
---

![10](https://user-images.githubusercontent.com/24144491/46719964-3590f880-ccaa-11e8-848e-1ee3d8b518c0.png)

YOLO의 장점은 빠르다! 정확하다! 다른 도메인에서도 작동 잘 됨! 이다.


## 3-1. 장점
---

![59](https://user-images.githubusercontent.com/24144491/46719982-3e81ca00-ccaa-11e8-87db-e7d4b3b86d3d.png)
![60](https://user-images.githubusercontent.com/24144491/46719774-bdc2ce00-cca9-11e8-92dc-9e802956da93.png)

**fast.**다른 알고리즘과 정확도(mAP)가 비슷하면서 훨씬 빠르다. 또 background error가 fast R-CNN에 비해 현저히 낮다.
[mAP 관한 설명 참고](https://datascience.stackexchange.com/questions/25119/how-to-calculate-map-for-detection-task-for-the-pascal-voc-challenge)

![61](https://user-images.githubusercontent.com/24144491/46719775-bdc2ce00-cca9-11e8-9431-df8ea07acd18.png)

**accurate.**또 fast R-CNN과 결합하면 더 좋은 성능을 보인다. (YOLO는 background에러를 줄이는데 사용하고 거기에서 비슷한 바운딩 박스의 R-CNN 예측값과 비교해 결과 산출)

![62](https://user-images.githubusercontent.com/24144491/46719776-be5b6480-cca9-11e8-8fc3-86bcec51e46c.png)

**Generalizability.** 아트 작품에서 detection 해 본 결과 다른 알고리즘보다 훨씬 좋은 성능을 보인다.

## 3-2. 한계
---

![63](https://user-images.githubusercontent.com/24144491/46719777-be5b6480-cca9-11e8-9568-dd3b124fe13c.png)

- 각 grid cell은 하나의 클래스만을 예측 / object가 겹쳐서 있으면 제대로 예측 x.

- Bounding box의 형태가 training data를 통해서만 학습되므로, 새로운/독특한 형태의 bouding box의 경우 정확히 예측하지 못함.

- 작은 bounding box의 loss term이 IOU에 더 민감하게 영향을 줌. localization이 다소 부정확함

![65](https://user-images.githubusercontent.com/24144491/46719773-bd2a3780-cca9-11e8-95d2-e17e4f333e43.png)

- Data dependency. 거의 모든 Supervised learning이 그러하듯, 이미지 데이터의 학습정도에 따라 모델의 성능이 크게 달라짐.

- Grid cell dependent(hyperparameter). 인풋 이미지의 크기에 따라 성능이 달라질 수 있음. ( 각 그리드 셀에 해당되는 부분만 classification을 하기 때문. 이는 다시 data dependency로 이어짐. 이 문제는 size 를 적당히 조절한 data augmentation을 해서 결과를 ensemble 하면 해결될 수 있지만 그만큼 속도가 느려짐 )

---
하지만 2018년도에 나온 YOLO v3는 이런 한계점들을 개선하여 여전히 좋은 성능과 빠름을 보장한다! [Yolo v3 youtube 영상](https://www.youtube.com/watch?v=MPU2HistivI)을 보면서 직접 확인해보라!


# Next
---

YOLO v1에 관한 논문 정리가 끝났다. 나의 삶의 모토(YOLO, you only live once)와 같은 이름을 가진 YOLO 논문의 작명센스가 논문을 읽게 만들었지만 버전 1을 이해하면서 어떤 점들을 어떻게 개선해 성능을 끌어 올렸는지 궁금해졌다.

이미지 처리, 데이터 분석, 자연어 처리, 강화학습 등 딥러닝의 가장 중요하면서 필요한 부분은 환경설정과 모델 학습을 위한 데이터 처리라는 것을 깨닫고 난 다음 이 부분들을 어떻게 빨리 처리할지에 대한 통찰과 숙련도를 쌓기위한 일환으로도 Imagee detection은 공부할 가치가 있는 영역이다.

따라서 다음에 Yolo v2, v3논문을 deep하게 리뷰하고 시간이 된다면 다른 detection 알고리즘들도 살펴볼 예정이다.



## 참고자료
---

- [https://curt-park.github.io/2017-03-26/yolo/](https://curt-park.github.io/2017-03-26/yolo/)
- [https://github.com/gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- [What’s new in YOLO v3](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
