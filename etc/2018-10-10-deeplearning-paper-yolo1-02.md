---
layout: post
title: "[논문] YOLO v1 분석 - 02" 
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




가독성을 위해 각 파트별 글을 따로 포스팅했다. 이번 글에서 다룰 파트는 **2. Unified Detection**이다.


# 2. Unified Detection
---


![11](https://user-images.githubusercontent.com/24144491/46719101-0e392c00-cca8-11e8-9d7f-13879a7be3dd.png)

YOLO의 모델 구성부터 학습 및 테스트까지의 일련의 과정을 위와 같은 순서로 살펴보고자 한다.

![12](https://user-images.githubusercontent.com/24144491/46719102-0e392c00-cca8-11e8-943b-af80d7c935dd.png)

**single Regression problem.** 1부에서 언급했듯이 YOLO는 image detection을 하나의 신경망을 구성하여 회귀문제로 정의한다. 

![13](https://user-images.githubusercontent.com/24144491/46719103-0ed1c280-cca8-11e8-8b8e-5946d4feeebc.png)

**To optimize the Loss(regression term), repeat the process.** 회귀 문제를 어떻게 최적화 했는지 그 과정을 한 번 생각해보자. 입력(input)이 있으면 적절한 모델을 구성해 output을 내고 그 output과 label간의 error를 담고 있는 loss를 계산한다. 그 loss를 최소화 시키기 위해 최적화 알고리즘을 통해 loss가 최소가 되도록 model 안에 있는 parameter(weight)들을 update시키는 과정을 반복한다.

![14](https://user-images.githubusercontent.com/24144491/46719104-0ed1c280-cca8-11e8-83e8-a8ab37510c65.png)

**We should define proper model, output and loss function.** 따라서 학습하기에 앞서 문제를 풀기위한 적절한 모델과 출력값 그리고 loss function을 설정해야한다.

![15](https://user-images.githubusercontent.com/24144491/46719105-0ed1c280-cca8-11e8-8fea-f5b221b2fa6f.png)

**Model, Output, Loss.** Model은 조금 변형한 GoogleNet, Output은 bounding box에 대한 정보와 클래스 확률 그리고 그에 맞는 적절한 loss fucntion을 구성했다. 

지금부터 하나씩 그 속을 들여다 보자.


## 2-1. Model design
---

![16](https://user-images.githubusercontent.com/24144491/46719106-0f6a5900-cca8-11e8-92a9-b9735f0d9e73.png)

**2-1에 관한 논문 내용.**모델 디자인에 관한 논문의 내용은 위와 같다. 뇌피셜과 배경지식을 최대한 배제하고 혹시 논문에 대한 내용을 내가 잘못 이해할 수도 있기에 논문에 대한 내용도 같이 참고하면서 글을 봤으면 한다. 각 파트를 설명하기 전에 위에처럼 각 파트를 이해하는데 필요한 논문에 나와 있는 부분들을 모아놨으니 참고하기 바란다.

![17](https://user-images.githubusercontent.com/24144491/46719107-0f6a5900-cca8-11e8-8414-40a74c195f3d.png)

**24 Conv layers.** 앞에 20개의 Conv 레이어는 구글넷의 구성과 동일하고 3x3만 쌓는대신 1x1 reduction conv 레이어를 몇개 추가했다. 그리고 4개의 Conv 레이어를 더 쌓았다. Fast YOLO는 이보다 더 compact한 9개의 conv 레이어를 가진다.

![18](https://user-images.githubusercontent.com/24144491/46719109-0f6a5900-cca8-11e8-88e2-3ff5f5ff7b42.png)

**2 fully conncected layers.** 우리가 예측해야할 output을 만들기 위해 2개의 fully connected레이어를 쌓았다. (yolo v2, v3 논문을 잠깐 봤을 때 fully connected 레이어를 더이상 쓰지 않았던 것 같은데 이 부분은 추후에 yolo v2, v3 논문을 리뷰할 때 다시 언급할 예정이다)


## 2-2. Model Output
---

![19](https://user-images.githubusercontent.com/24144491/46719111-1002ef80-cca8-11e8-88ab-476cf2ab5f86.png)

**2-2에 관한 논문 내용.**

![20](https://user-images.githubusercontent.com/24144491/46719099-0da09580-cca8-11e8-9251-7b946ebf9bea.png)

**Output tensor = SxSx(B*5+C).** 출력 텐서 형태는 S(그리드)와 B(바운딩 박스) 그리고 C(클래스)에 의해 결정된다. 우리가 예측해야할 출력 값은 클래스 확률과 바운딩 박스에 대한 좌표와 바운딩박스의 확률이다. 아래에서 더 자세히 살펴보자.

![21](https://user-images.githubusercontent.com/24144491/46719256-696b1e80-cca8-11e8-81c7-e02b74ec9a4a.png)

좀 더 직관적인 이해를 위해 [deepsystem의 피피티 내용](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_969)을 가져왔다.

![22](https://user-images.githubusercontent.com/24144491/46719257-696b1e80-cca8-11e8-9ece-dcfe99335118.png)
**바운딩 박스 예측.** 전체 이미지를 SxS로 나눈다고 생각했을때 총 SxS의 그리드 셀이 생기게 된다. 각 그리드 셀 안의 x,y 좌표와 w,h 바운딩 박스의 폭과 높이, c 바운딩 박스의 confidence score를 예측하게 되는데 이런 5개의 예측값을 B개의 바운딩 박스 갯수만큼 만들어주어야 한다.

- 전체 이미지를 SxS 로 나눔.
- 나눈 각 그리드 셀 영역에는 (x,y,w,h,c) * B(바운딩 박스의 갯수) 만큼의 예측값이 필요함.
- x는 바운딩 박스 중심의 x좌표 ([0,1]로 normalize, 그리드 셀의 좌상단 x값을 더해주면 원래 좌표가 나오므로)
- y는 바운딩 박스 중심의 y좌표 ([0,1]로 normalize)
- w는 바운딩 박스의 폭	([0,1]로 normalize)
- h는 바운딩 박스의 높이	([0,1]로 normalize)
- c는 박스의 confidence score로 Pr(Object) x IOU, 물체가 있을 확률과 실제 물체의 바운딩 박스와 얼만큼 겹치는지에 대한 값과의 곱. (여기서 IOU는 intersection of Union으로 예측된 바운딩 박스가 실제 바운딩 박스(truth ground value)와 얼마나 겹치는지를 계산한 값)

![23](https://user-images.githubusercontent.com/24144491/46719258-6a03b500-cca8-11e8-8619-90e473ec6254.png)

**B개의 바운딩 박스.** 여기서 B는 2이므로 총 2개의 서로 다른 바운딩 박스를 예측하게 된다.

![24](https://user-images.githubusercontent.com/24144491/46719259-6a03b500-cca8-11e8-8e1b-b35f3f7d5540.png)

**예측 값들의 결과.** 우하단에 있는 그림처럼 S=7, B=2 일때는, S x S x B = 7 x 7 x 2 = 98개의 서로 다른 바운딩 박스의 값들을 예측하게 된다. 박스 영역이 두꺼울수록 더 높은 Confidence score를 가지게 된다.

![25](https://user-images.githubusercontent.com/24144491/46719260-6a03b500-cca8-11e8-9857-9db8e5c1a68c.png)

**클래스 스코어.** deepsystem.io에서 약간 용어의 선택을 잘못한 부분이있는데, 해당 영역에 대한 클래스를 판단할 때는 바운딩 박스 안에서가 아닌, 바운딩 박스의 중심좌표가 들어있는 그리드 셀에서만 판단하게 된다. 여기서 C=20, 총 20개의 클래스를 예측하는 문제이고, 각 그리드 셀 하나하나는 20개의 클래스에 대한 예측 값들을 가지게 된다. 오른쪽 아래의 그리드셀을 여러가지의 색으로 구분한 그림을 보자. 서로 다른 색은 서로 다른 클래스이고, 각 그리드 셀에서 가장 높게 예측된 클래스의 색칠을 칠하면 오른쪽 아래와 같은 그림이 나올 것이다.

![26](https://user-images.githubusercontent.com/24144491/46719261-6a9c4b80-cca8-11e8-9ce8-386bbd658137.png)

**Output을 그림으로 표현하면 위와 같다.** 즉 우리가 예측한 출력값은 S x S x (B x 5 + 20) 으로 구성된 텐서이며, 이 예에서는 S = 7, B = 2로 총 7 x 7 x 30의 텐서가 나오게 된다. 아웃풋 그림의 위쪽은 각 그리드 셀에서 예측한 2개의 바운딩박스에 관한 값들을 시각화한 것이고 아래는 그리드셀에서 예측한 가장 높은 클래스를 색칠한 것을 시각화한 그림이다.	

## 2-3. Loss function
---

![27](https://user-images.githubusercontent.com/24144491/46719262-6a9c4b80-cca8-11e8-9ae8-7e72efccba6e.png)

**2-3에 관한 논문 내용.**

![28](https://user-images.githubusercontent.com/24144491/46719263-6b34e200-cca8-11e8-9669-7646c2b2e39a.png)

**Loss term에 대한 정리.** loss term들을 분류하자면 바운딩박스의 위치와 크기에 대한 텀, 해당 그리드셀에 오브젝트가 있는지 없는지에 대한 loss term 그리고 클래스에 대한 loss term으로 나눠 생각할 수 있다. (output 형태를 생각해보면 이해가 더 쉽다. 각 부분에 대한 loss term들을 만들었기 때문)

![29](https://user-images.githubusercontent.com/24144491/46719264-6b34e200-cca8-11e8-8314-19c30354ec38.png)

- 검은색 영역 : 최적화하기 쉽게 sum-squred error로 구성
- 노란색 영역 : 오브젝트가 있는것과 없는 것 간의 차이를 둠, 람다_coord = 5, 람다_noobj = 0.5

![30](https://user-images.githubusercontent.com/24144491/46719255-696b1e80-cca8-11e8-84ce-1b9e4f54f399.png)

- 파란색 영역과 노란색 영역 : 2-2.Output에서 이미 살펴봄.
- 빨간색 영역 : 1_obj_ij는 오브젝트가 있는 i번째의 그리드 셀에 j번째 바운딩 박스 / 1_obj_i는 오브젝트가 있는 i번째 그리드 셀

![31](https://user-images.githubusercontent.com/24144491/46719382-c7980180-cca8-11e8-8d08-ed3f6eed0417.png)

- 빨간색 영역 : 큰 박스와 작은 박스의 부분을 제곱해주면 작은 박스는 큰 박스에 비해 더 작아지므로 이를 상쇄시키기 위해 루트를 씌어준 다음 제곱해줌
- 파란색 영역 : 물체가 없는 곳에서는 클래스 확률과 바운딩 박스의 크기를 더이상 고려할 필요가 없으므로 물체가 있는 영역에서만 바운딩 박스에 대한 loss term과 그리드 셀의 클래스 확률에 대한 loss term울 고려한다.

## 2-4. Training
---

![34](https://user-images.githubusercontent.com/24144491/46719385-c8309800-cca8-11e8-8005-22af2e0e57b5.png)

**2-4에 해당하는 논문 내용.**

![35](https://user-images.githubusercontent.com/24144491/46719386-c8309800-cca8-11e8-9df5-49604594d72e.png)

**pre training.** 앞의 20개의 conv 레이어는 pre-training 시킨다.

![36](https://user-images.githubusercontent.com/24144491/46719387-c8309800-cca8-11e8-88dc-0e5051df4bbb.png)

- googleNet 1000-class dataset으로 20개의 convolutioanl layer를 pre-training
- Batch size: 64
- Momentum: 0.9 and a decay of 0.0005
- Learning Rate: [0.001, 0.01, 0.001, 0.0001]  ( 처음에 0.001에서 서서히 감소시키다 75 - epoch동안 0.01, 30 epoch동안 0.001, 마지막 30 epoch동안 0.0001)
- Dropout Rate: 0.5
- Data augmentation: random scailing and translations of up to 20% of the original image size
- Activation function: leaky ReLU ( 알파 = 0.1 )

## 2-5. Testing
---

![37](https://user-images.githubusercontent.com/24144491/46719388-c8c92e80-cca8-11e8-8efd-18553264fd9d.png)

**2-5에 해당하는 논문 내용.**

![38](https://user-images.githubusercontent.com/24144491/46719389-c8c92e80-cca8-11e8-978e-84f78ce97938.png)

테스트할 때, 성능을 확인하기 위해서 최종적인 bounding box를 예측해야한다. 2-2에서 다룬 output 형태에서 어떻게 최종적인 detection output을 뽑아내는지 살펴보자. 첫 번째로는 output 에서 예측된 바운딩 박스의 confidence score과 그리드 셀의 클래스 score를 곱한다. 마지막으로 non-maximal subpression 알고리즘을 통해 각 오브젝트당 하나의 바운딩 박스를 예측하면 끝.

![39](https://user-images.githubusercontent.com/24144491/46719390-c8c92e80-cca8-11e8-8878-a2cab76c4fd5.png)

첫 번째 과정을 조금 더 자세히 살펴보자. 이에 대한 설명 역시 [deepsystem의 피피티 내용](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.g137784ab86_4_969)을 가져왔다. 

![40](https://user-images.githubusercontent.com/24144491/46719391-c961c500-cca8-11e8-9623-5e4449613afb.png)

![41](https://user-images.githubusercontent.com/24144491/46719392-c961c500-cca8-11e8-973c-3a431e8880e3.png)

각 바운딩 박스와 그 바운딩 박스의 중심좌표가 있는 그리드 셀의 클래스 스코어 값들을 각각 곱한다.

![42](https://user-images.githubusercontent.com/24144491/46719393-c961c500-cca8-11e8-8877-64313c00c0a9.png)

그러면 한 그리드 셀에는 2개의 바운딩 박스의 클래스 값들이 있는 20 x 1의 벡터들이

![43](https://user-images.githubusercontent.com/24144491/46719394-c9fa5b80-cca8-11e8-8925-9669cbce4ef6.png)

다음과 같이 쌓이고

![44](https://user-images.githubusercontent.com/24144491/46719396-c9fa5b80-cca8-11e8-943f-5b78e6e20860.png)

최종적으로 B x (S x S) x C = 2 x 7 x 7 x 20 = 1440개의 값이 나오게 된다.

![44](https://user-images.githubusercontent.com/24144491/46719396-c9fa5b80-cca8-11e8-943f-5b78e6e20860.png)

다음으로 위에서 나온 1440개의 값들에서 어떻게 해당 물체에 하나의 바운딩 박스가 선택되는지 알아보자.


![46](https://user-images.githubusercontent.com/24144491/46719398-c9fa5b80-cca8-11e8-8205-66cb9f970f32.png)

일단 threshold 값으로 0.2를 지정하고 해당 값이 0.2보다 작은 결과에 대해서는 다 0으로 만든다. 그리고 내림차순으로 정렬을 하고, NMS의 알고리즘을 통해서 최종적인 detection output을 만들 수 있다.

![47](https://user-images.githubusercontent.com/24144491/46719399-ca92f200-cca8-11e8-9e06-076409735c21.png)

그럼 NMS(non-maximal subpression)은 어떻게 작동할까?

![48](https://user-images.githubusercontent.com/24144491/46719400-ca92f200-cca8-11e8-8061-211d20f077a5.png)

첫 반복에서 클래스 별로 가장 높은 값을 가지는 바운딩 박스의 값을 bbox_max라 하면, bbox_max랑 현재 비교하고 있는 bbox_cur의 IOU (Intersection of Union, 얼마나 겹치는지)를 계산하고 이게 0.5보다 크면 같은 물체를 다른 바운딩 박스로 예측하고 있다고 판단하고 bbox_cur의 값을 0으로 만들어준다.

![49](https://user-images.githubusercontent.com/24144491/46719402-ca92f200-cca8-11e8-9989-a86a01c97e22.png)

IOU(겹치는부분)이 0.5보다 크지 않으면 놔둔다.

![50](https://user-images.githubusercontent.com/24144491/46719403-cb2b8880-cca8-11e8-9628-8f8be355d2df.png)

분홍색 값도 마찬가지로 놔둔다.

![51](https://user-images.githubusercontent.com/24144491/46719406-cb2b8880-cca8-11e8-9276-d38dae63d259.png)

그리고 다음 반복에서 제일 큰 값부터 위와 같은 과정을 똑같이 반복한다.

![52](https://user-images.githubusercontent.com/24144491/46719407-cb2b8880-cca8-11e8-9b96-24877645f00a.png)
겹치는 부분이 0.5보다 크므로 0으로 만들어주고.

![55](https://user-images.githubusercontent.com/24144491/46719410-cbc41f00-cca8-11e8-861e-aa3ec55ddfcc.png)

위의 과정이 끝나면 0 값이 엄청 많이 만들어 질 것이다.

![57](https://user-images.githubusercontent.com/24144491/46719412-cc5cb580-cca8-11e8-8721-3183626b0a0d.png)

마지막으로 각 바운딩 박스에 대해서 가장 크게 예측되고 0보다 큰 클래스만 찍어주면

![58](https://user-images.githubusercontent.com/24144491/46719413-cc5cb580-cca8-11e8-98f9-55c30bda1094.png)

완성!

![59](https://user-images.githubusercontent.com/24144491/46719381-c6ff6b00-cca8-11e8-8ff5-3accfe181f9c.png)

**YOLO에 대한 모든 과정을 한 번에 살펴 보았다!**


# Next
---

다음은 yolo의 성능과 한계에 대한 [3. Conclusion](https://taeu.github.io/paper/deeplearning-paper-yolo1-03/) 내용을 살펴볼 것이다.


## 참고자료
---

- [https://curt-park.github.io/2017-03-26/yolo/](https://curt-park.github.io/2017-03-26/yolo/)
- [https://github.com/gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- [What’s new in YOLO v3](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
- [https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/edit#slide=id.g15092aa245_0_15](https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/edit#slide=id.g15092aa245_0_15)
- [https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/present?ueb=true#slide=id.g151008b386_0_51](https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/present?ueb=true#slide=id.g151008b386_0_51)
