---
title: "(AlexNet) ImageNet Classification with Deep Convolutional Neural Networks"
date: 2025-08-28 13:00:00 +0900
categories: [AI-ML-DL, etc.]
tags: [paper, Computer-Vision]
---

# (AlexNet) ImageNet Classification with Deep Convolutional Neural Networks

---

### 🔗 출처

> https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
> 

---

## 🧩 방법론

> **by GPT5**
> 

<aside>

## 1. 모델 아키텍처 (Architecture)

- **구조**: 5 convolutional layer + 3 fully-connected layer + 1000-way softmax
- **입력**: 224×224×3 RGB 이미지
- **Conv1**: 96개, 11×11×3, stride=4 → ReLU → LRN → max-pooling
- **Conv2**: 256개, 5×5×48 (GPU 분할) → ReLU → LRN → max-pooling
- **Conv3**: 384개, 3×3×256 (GPU 전체 연결)
- **Conv4**: 384개, 3×3×192 (GPU 내부 연결)
- **Conv5**: 256개, 3×3×192 (GPU 내부 연결) → max-pooling
- **FC1, FC2**: 각각 4096 뉴런 (Dropout 적용)
- **FC3**: 1000 뉴런 → softmax

---

## 2. 핵심 기법 (Key Techniques)

- **ReLU 활성화 함수**
    - f(x) = max(0,x)
    - sigmoid, tanh 대비 학습 속도 수배 향상 (경사 소실 문제 완화).
- **Multi-GPU 학습**
    - 2개 GPU에 커널을 분산.
    - Conv3은 양쪽 GPU 연결, Conv4·5는 같은 GPU 연결.
    - “columnar CNN”과 유사하지만 독립이 아닌 협력 구조.
- **Local Response Normalization (LRN)**
    - 뉴런 간 경쟁(lateral inhibition) 유도.
    - ReLU 출력값을 인접 채널 값으로 정규화.
    - Top-5 오류율 ~1% 개선.
- **Overlapping Pooling**
    - Pool size z=3, stride s=2 (겹치는 영역).
    - Non-overlapping 대비 오류율 소폭 개선, 과적합 완화.

---

## 3. 과적합 방지 (Regularization)

- **Data Augmentation**
    1. Random crop & horizontal flip → 2048배 데이터 확장 효과.
    2. PCA 기반 색상 변형(color jitter) → 조명/색상 변화 불변성 반영.
- **Dropout (0.5 확률)**
    - FC1, FC2 층에 적용.
    - 뉴런 co-adaptation 방지, 일반화 성능 향상.

---

## 4. 학습 (Optimization & Training)

- **Optimizer**: SGD with momentum
    - Batch size = 128
    - Momentum = 0.9
    - Weight decay = 0.0005 (학습 안정화에도 기여)
- **초기화**
    - 가중치: N(0, 0.01²) Gaussian
    - Bias: Conv2/4/5, FC층 = 1 (ReLU 활성화 유도), 나머지 = 0
- **학습률 스케줄**
    - 초기 lr = 0.01
    - Validation error 개선 멈추면 /10
    - 총 3회 감소
- **훈련 세팅**
    - 데이터셋: ILSVRC (120만 장)
    - Epoch: ~90
    - 하드웨어: NVIDIA GTX 580 (3GB) × 2
    - 소요 시간: 약 5–6일

---

## 📖 요약

- **구조**: 8층 (5 conv + 3 fc) CNN, GPU 2장 분산 구조
- **핵심**: ReLU, LRN, Overlapping Pooling, Dropout
- **과적합 방지**: 데이터 증강 + Dropout
- **학습 전략**: SGD + momentum + weight decay, 수동 lr 스케줄
- **훈련 환경**: 당시 한계(3GB GPU)를 극복하기 위해 multi-GPU 활용

---

👉 이렇게 정리하면 “AlexNet이 어떻게 설계·학습되었는가”를 한눈에 볼 수 있습니다.

원하시면 제가 이걸 **그림/플로우차트 형태의 시각화 자료**로도 만들어드릴까요?

</aside>

---

# 📌 논문

## 💡 요약

> **by GPT5**
> 

<aside>

## 1. 배경

- 2010년대 초반까지 객체 인식은 **SIFT, HOG, Fisher Vector** 등 전통적 특징 기반 접근이 주류.
- 그러나 실제 환경의 복잡한 사물 인식을 위해서는 **더 큰 데이터셋 + 더 강력한 모델**이 필요.
- **ImageNet (15M 이미지, 22k 클래스)** + **GPU 발전** → 대규모 CNN 학습이 가능해짐.

---

## 2. 데이터셋

- **ILSVRC-2010/2012**: 1000 클래스, 120만 train, 50k val, 150k test.
- **전처리**: 이미지 리사이즈(짧은 변 256) → 중앙 256×256 crop → 평균값 빼기.
- **평가 지표**:
    - Top-1 오류율: 가장 큰 확률이 정답이 아님.
    - Top-5 오류율: 상위 5개 확률 안에 정답이 없음.

---

## 3. 아키텍처

- **구조**: 5 convolutional + 3 fully-connected → 1000-way softmax.
- **Conv1**: 96×(11×11), stride=4
- **Conv2**: 256×(5×5), GPU 분할
- **Conv3**: 384×(3×3), 두 GPU 모두 연결
- **Conv4**: 384×(3×3), 같은 GPU 연결
- **Conv5**: 256×(3×3), 같은 GPU 연결 → max-pooling
- **FC1–2**: 4096 뉴런 (dropout 적용)
- **FC3**: 1000 뉴런 → softmax

---

## 4. 핵심 기법

- **ReLU (non-saturating neuron)**
    - f(x) = max(0,x). sigmoid/tanh 대비 학습 속도 수배 향상.
- **Multi-GPU 분산 학습**
    - 2 GPU에 네트워크 분할, 일부 계층만 cross-connection → 단일 거대 CNN처럼 동작.
- **Local Response Normalization (LRN)**
    - 뉴런 간 경쟁 유도(lateral inhibition). top-5 오류율 ~1% 개선.
- **Overlapping Pooling**
    - 3×3, stride=2 사용. 과적합 완화, 오류율 소폭 개선.
- **Data Augmentation**
    - Random crop + horizontal flip (2048배 데이터 확장 효과).
    - PCA 기반 RGB 채널 색상 변화 (조명·색 변화 불변성 반영).
- **Dropout**
    - FC1, FC2에 적용. 뉴런 co-adaptation 방지. 과적합 크게 감소.

---

## 5. 학습 세부사항

- Optimizer: SGD (batch=128, momentum=0.9, weight decay=0.0005).
- 초기화: 가우시안(0, 0.01), 일부 bias=1 (ReLU 활성화 유도).
- 학습률 스케줄: 초기 0.01 → 개선 정체 시 /10 (총 3회).
- 학습: 90 epoch, 2×GTX 580 GPU (5~6일 소요).

---

## 6. 결과

- **ILSVRC-2010**:
    - CNN: top-1 37.5%, top-5 17.0%
    - 기존 최고: top-1 47.1%, top-5 28.2%
- **ILSVRC-2012**:
    - 단일 CNN: top-5 18.2%
    - 5개 앙상블: 16.4%
    - 사전학습+앙상블: 15.3% (2등은 26.2%)
- **ImageNet 2009 (10k 클래스)**:
    - CNN: top-1 67.4%, top-5 40.9%
    - 기존 최고: 78.1%, 60.9%

---

## 7. 고찰

- **깊이의 중요성**: conv layer를 하나만 제거해도 성능 2%↓.
- **비지도 사전학습**: 본 연구에서는 미사용, 향후 더 큰 네트워크에서 유용할 것.
- **스케일링 효과**: 네트워크를 키우고 학습을 오래 할수록 성능 개선.
- **인간 시각과의 격차**: IT cortex 수준까지는 아직 멀었음.
- **미래 방향**: 정적 이미지 → **비디오 학습**으로 확장 (temporal structure 활용).

---

## 🌟 의의

- AlexNet은 **전통적 feature-engineering을 완전히 대체**하며, 컴퓨터 비전 패러다임을 **딥러닝 기반**으로 전환시킨 기념비적 연구.
- ReLU, Dropout, Data augmentation, GPU 분산 학습 등 **오늘날 딥러닝 기본 요소**들을 정립.
- “더 큰 데이터 + 더 큰 모델 + 더 빠른 하드웨어”가 성능을 향상시킨다는 딥러닝 연구의 기본 방향을 제시.

---

👉 이걸 발표 슬라이드용으로 **더 요약한 한 페이지 버전**도 만들어드릴 수 있습니다. 원하시나요?

</aside>

---

## 📚 정리

### 📌 제목

<aside>

## ImageNet Classification with Deep Convolutional Neural Networks

</aside>

---

### 🌟 초록

### 번역

We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of ﬁve convolutional layers, some of which are followed by max-pooling layers, and three fully-connected layers with a ﬁnal 1000-way softmax. To make training faster, we used non-saturating neurons and a very efﬁcient GPU implementation of the convolution operation. To reduce overﬁtting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry.

우리는 ImageNet LSVRC-2010 대회의 120만 장의 고해상도 이미지를 1000개의 서로 다른 클래스에 분류하기 위해, 크고 깊은 컨볼루션 신경망(convolutional neural network)을 학습시켰다. 테스트 데이터에서 우리는 top-1과 top-5 오류율을 각각 37.5%와 17.0% 달성했으며, 이는 이전 최첨단 성능보다 상당히 우수하다. 이 신경망은 6천만 개의 파라미터와 65만 개의 뉴런으로 구성되어 있으며, 다섯 개의 컨볼루션 계층(일부는 맥스 풀링 계층과 결합)과 세 개의 완전 연결 계층(fully-connected layer)으로 구성되며, 마지막은 1000-way softmax를 갖는다. 학습 속도를 높이기 위해, 우리는 non-saturating neuron(비포화형 뉴런)을 사용하고, 컨볼루션 연산의 매우 효율적인 GPU 구현을 활용했다. 또한 완전 연결 계층에서 과적합을 줄이기 위해 dropout이라는 최근 개발된 정규화 기법을 적용했는데, 이는 매우 효과적임이 입증되었다. 우리는 또한 이 모델의 변형을 ILSVRC-2012 대회에 출전시켰고, top-5 테스트 오류율 15.3%를 달성하여, 26.2%를 기록한 2등 결과를 크게 앞섰다.

<aside>

- 1000-way softmax : 1000개의 cls.문제이기 때문에, output을 1000개의 vec.으로 만들었다.
    - Top-1 error: 가장 큰 확률이 정답이 아닐 때.
    - Top-5 error: 가장 큰 확률 5개 안에 정답이 없을 때.
- saturating neuron : 뉴런의 활성화 함수중에서 입력값이 커지거나 작아지면 포화되는 현상으로, sigmoid, tanh에 해당 → 경사소실 문제 따라서 AlexNet에서는 ReLU 채택

| 항목      | 내용                                                                                                |
| --------- | --------------------------------------------------------------------------------------------------- |
| 데이터셋  | ImageNet LSVRC-2010 (120만 장, 1000 클래스), ILSVRC-2012 대회 참가                                  |
| 모델 규모 | 60M 파라미터, 650k 뉴런                                                                             |
| 아키텍처  | 5개 convolutional layer + 3개 fully-connected layer + softmax(1000-way)                             |
| 핵심 기법 | ReLU(non-saturating neuron), GPU 기반 효율적 학습, Dropout 정규화                                   |
| 성능      | ILSVRC-2010: top-1 37.5%, top-5 17.0% / ILSVRC-2012: top-5 15.3% (1등)                              |
| 기여      | CNN을 대규모 데이터·GPU와 결합 → 기존 feature-engineering 기반 방법보다 획기적으로 우수한 성능 달성 |
- **아키텍쳐**

![image.png]((AlexNet)%20ImageNet%20Classification%20with%20Deep%20Convol%2025ca9b246de1800b878cd59e8addb4dd/image.png)

</aside>

---

### 💡 결론 & 고찰

### 번역

**결론**

우리의 ILSVRC-2010 결과는 Table 1에 요약되어 있다. 우리가 제안한 네트워크는 top-1과 top-5 테스트 오류율을 각각 37.5%와 17.0% 달성했다. ILSVRC-2010 대회 당시의 최고 성능은 여섯 개의 sparse-coding 모델을 다른 특징(feature)에 대해 학습시킨 뒤 그 예측을 평균 내는 접근법으로, top-1 47.1%, top-5 28.2%였다. 이후 발표된 최고 성능도 SIFT와 Fisher Vectors(FVs)를 활용한 방법으로, top-1 45.7%, top-5 25.7%였다.

우리는 또한 ILSVRC-2012 대회에도 참가했고, 그 결과는 Table 2에 정리되어 있다. ILSVRC-2012의 테스트 라벨은 공개되지 않았기 때문에 모든 모델의 테스트 오류율을 보고할 수는 없었다. 하지만 검증(validation)과 테스트 오류율은 거의 차이가 없었으므로, 여기서는 동일하게 사용한다. 우리가 제안한 CNN은 top-5 오류율 18.2%를 기록했다. 다섯 개의 유사한 CNN의 예측을 평균한 경우 오류율은 16.4%로 낮아졌다. 마지막 풀링 계층 위에 여섯 번째 convolutional layer를 추가하고, ImageNet Fall 2011 전체 데이터셋(1500만 장, 22,000 클래스)에서 사전학습(pre-train)한 뒤 ILSVRC-2012에 파인튜닝한 경우 top-5 오류율은 16.6%였다. 또, 위에서 언급한 다섯 개 CNN에 두 개의 사전학습된 CNN을 추가로 앙상블하면 top-5 오류율이 15.3%까지 떨어졌다. 2위 팀은 Fisher Vectors 기반 여러 분류기의 앙상블로 26.2%를 기록했다.

마지막으로, 우리는 ImageNet 2009 버전(10,184 클래스, 890만 장 이미지)에 대해서도 결과를 보고한다. 이 경우 데이터셋을 절반은 학습, 절반은 테스트로 나누었다. 이 데이터셋에서는 우리의 네트워크가 top-1 오류율 67.4%, top-5 오류율 40.9%를 기록했으며, 이는 기존 최고 성능(78.1%, 60.9%)보다 훨씬 우수하다.

추가적으로, Figure 3은 네트워크가 학습한 첫 번째 convolutional layer의 커널을 보여준다. 다양한 주파수 및 방향 선택적 필터뿐 아니라 색상 blob들도 학습되었음을 확인할 수 있다. 두 개의 GPU에 분산된 구조 덕분에 GPU1에서는 주로 색상에 무관한 필터, GPU2에서는 색상 특화 필터가 학습되었다.

Figure 4는 네트워크의 top-5 예측 결과와, 마지막 은닉층(4096차원 feature space)에서 가장 가까운 훈련 이미지들을 보여준다. 여기서 네트워크가 객체의 위치가 중심에 있지 않아도 인식 가능하며, 유사한 semantic feature를 가진 이미지를 embedding 공간에서 가깝게 매핑함을 확인할 수 있다. 다만 원시 픽셀 기준(L2 거리)에서는 비슷하지 않은 경우도 많다. 따라서 feature space 기반 유사도는 더 의미 있는 시각적 유사성을 제공한다.

마지막으로, 이러한 4096차원 feature를 효율적으로 검색하기 위해 auto-encoder 기반 압축 기법을 적용하면, 단순 raw pixel 기반 retrieval보다 훨씬 더 나은 결과를 기대할 수 있다고 제안한다.

**논의**

우리의 결과는, 대규모의 깊은 컨볼루션 신경망이 순수한 지도학습만으로도 매우 도전적인 데이터셋에서 기록적인 성능을 달성할 수 있음을 보여준다. 흥미로운 점은, 네트워크에서 단일 convolutional layer를 제거하면 성능이 저하된다는 것이다. 예를 들어, 중간 계층 중 하나를 제거하면 top-1 정확도가 약 2% 정도 하락한다. 따라서 깊이가 좋은 성능 달성에 실제로 중요한 역할을 한다는 사실을 확인했다.

실험을 단순화하기 위해, 우리는 이번 연구에서 어떤 비지도 사전학습(unsupervised pre-training)도 사용하지 않았다. 하지만 이는 도움이 될 수 있을 것으로 예상된다. 특히, 훨씬 더 큰 네트워크를 사용할 수 있는 계산 자원을 확보할 수 있다면, 라벨 데이터가 충분하지 않은 상황에서도 성능 향상에 기여할 수 있다. 지금까지 우리의 결과는 네트워크를 크게 하고 학습 시간을 늘릴수록 개선되어 왔으나, 여전히 인간 시각 시스템의 infero-temporal pathway와 비교하면 갈 길이 멀다.

궁극적으로 우리는 매우 크고 깊은 컨볼루션 네트워크를 비디오 시퀀스에 적용하고 싶다. 영상의 시계열적 구조는 정적인 이미지에서는 결여되거나 덜 명확한 중요한 정보를 제공하기 때문이다.

<aside>

| 항목                       | 내용                                                                                                     |
| -------------------------- | -------------------------------------------------------------------------------------------------------- |
| ILSVRC-2010                | CNN: top-1 37.5%, top-5 17.0% (기존 최고: 47.1%, 28.2%)                                                  |
| ILSVRC-2012                | 단일 CNN: top-5 18.2% → 5개 앙상블: 16.4% → 사전학습+앙상블: 15.3% (2위는 26.2%)                         |
| ImageNet 2009 (10k 클래스) | CNN: top-1 67.4%, top-5 40.9% (기존 78.1%, 60.9%)                                                        |
| Qualitative 분석           | CNN은 색상·방향·주파수 selective kernel 학습 / GPU별 specialization 발생                                 |
| Feature space              | 4096차원 feature vector로 이미지 간 semantic similarity 반영, raw pixel distance보다 의미 있는 검색 가능 |
| 추가 제안                  | auto-encoder로 feature vector 압축 → 효율적 이미지 검색(Image retrieval                                  |
- 네트워크를 깊게 만든다면 효과가 더 클것으로 예상되어 진다. 또한 unsupervised pretraining도 효과적으로 작동할것으로 판단된다.
- CNN의 깊이가 성능향상에 기여한다.

| 항목            | 내용                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------ |
| 깊이의 중요성   | convolution layer 하나만 제거해도 top-1 성능 약 2% 저하 → 깊은 구조가 핵심                 |
| 비지도 사전학습 | 본 논문에서는 사용하지 않았음. 하지만 더 큰 네트워크·라벨 부족 환경에서 유용할 것으로 예상 |
| 성능 스케일링   | 네트워크를 크게 하고 학습 시간을 늘리면 성능이 계속 향상됨 (scale-up 효과)                 |
| 한계            | 인간 뇌의 시각 피질(infero-temporal pathway)에 비하면 아직 멀었다고 언급                   |
| 미래 방향       | 정적 이미지가 아닌 **비디오 학습**으로 확장 → 시계열적 단서(temporal structure) 활용       |
</aside>

---

### 🗃️ 데이터

### 번역

ImageNet은 약 2,2000개 카테고리에 걸쳐 1,500만 장 이상의 라벨링된 고해상도 이미지를 포함하는 데이터셋이다. 이 이미지는 웹에서 수집되었고, Amazon Mechanical Turk라는 크라우드소싱 도구를 통해 사람들에 의해 라벨링되었다. 2010년부터 Pascal Visual Object Challenge의 일부로서 매년 ImageNet Large-Scale Visual Recognition Challenge (ILSVRC)라는 대회가 열리고 있다. ILSVRC에서는 ImageNet의 일부 하위셋을 사용하며, 각 1,000개 카테고리마다 약 1,000개의 이미지가 주어진다. 전체적으로는 약 120만 장의 학습 이미지, 50,000장의 검증 이미지, 150,000장의 테스트 이미지로 구성된다.

ILSVRC-2010은 테스트셋 라벨이 공개된 유일한 버전이므로, 대부분의 실험은 이 버전을 기반으로 수행했다. 또한 우리는 모델을 ILSVRC-2012 대회에도 출전시켰으며, 해당 데이터셋에 대한 결과는 Section 6에서 보고한다. (ILSVRC-2012의 테스트 라벨은 공개되지 않는다.) ImageNet에서는 일반적으로 두 가지 오류율을 보고한다:

- **Top-1 오류율**: 모델이 예측한 확률이 가장 높은 클래스가 정답이 아닌 경우의 비율.
- **Top-5 오류율**: 모델이 예측한 상위 5개 클래스 안에 정답이 포함되지 않은 경우의 비율.

ImageNet은 해상도가 다양한 이미지를 포함하지만, 우리의 시스템은 고정된 입력 크기를 필요로 한다. 따라서 이미지를 **256 × 256**의 고정 해상도로 다운샘플링했다. 직사각형 이미지는 짧은 변이 256이 되도록 먼저 리사이즈한 뒤, 그 결과물에서 중앙 256 × 256 패치를 잘라냈다. 다른 전처리는 하지 않았으며, 훈련셋 전체에서 픽셀 단위 평균값을 빼는 작업만 수행했다. 따라서 네트워크는 중심화된(centered) 원시 RGB 픽셀 값으로 학습되었다.

<aside>

<aside>

- Amazon Mechanical Turk : **작업을 잘게 쪼개어(HITs, Human Intelligence Tasks) 온라인으로 다수의 사람(worker)에게 할당**, 소액 보상으로 처리하게 하는 시스템. 적은 데이터셋으로 대규모데이터셋 구성가능
</aside>

- CNN모델에 input으로 넣기 위해서, 256x256으로 다운샘플링 진행
    - 직사각형의 경우에는 이미지가 짧은변을 256으로 resizing한 후, 그 결과물에서 중앙 256x256패치를 잘라넴
- 전처리는 하지 않고, 훈련셋 전체에서 픽셀 단위 평균값을 빼는 작업만 수행해서, 네트워크는 중심화된 원시 RGB값으로 학습되어짐

| 항목          | 내용                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------- |
| 원본 데이터셋 | ImageNet (15M+ 이미지, 22k 클래스)                                                          |
| ILSVRC 버전   | 2010/2012 대회용 하위셋 (1000 클래스, 약 120만 train, 50k val, 150k test)                   |
| 특징          | ILSVRC-2010만 test 라벨 공개 (실험 검증용)                                                  |
| 오류 지표     | Top-1 (최고 확률 예측이 틀린 경우), Top-5 (상위 5개 예측 안에 정답 없음)                    |
| 전처리        | 모든 이미지를 256×256으로 리사이즈 후 중앙 crop / train set 픽셀 평균값을 빼고 raw RGB 사용 |
| 의의          | 당시로서는 unprecedented(전례 없는) 대규모 labeled dataset. CNN 학습 가능케 한 기반.        |
</aside>

---

### 📌 서론

### 번역

현재의 객체 인식(object recognition) 접근법은 본질적으로 머신러닝 방법에 의존한다. 성능을 향상시키기 위해서는 더 큰 데이터셋을 수집하고, 더 강력한 모델을 학습시키며, 과적합을 방지할 더 나은 기법을 사용하는 것이 필요하다. 최근까지 라벨링된 이미지 데이터셋은 비교적 작았으며, 수만 장 수준(NORB, Caltech-101/256, CIFAR-10/100 등)이었다. 이러한 규모의 데이터셋은 단순한 인식 과제에는 충분하며, 라벨을 보존하는 변환(label-preserving transformation)으로 증강하면 더 잘 동작하기도 한다. 예를 들어 MNIST 숫자 인식에서는 오류율이 0.3% 미만으로 인간 성능에 근접했다.

그러나 실제 환경 속 사물은 큰 변화를 보이므로, 이를 인식하려면 훨씬 더 큰 학습 데이터셋이 필요하다. 작은 이미지 데이터셋의 한계는 오래전부터 잘 알려져 있었지만, 최근에 와서야 수백만 장 규모의 라벨링된 데이터셋 수집이 가능해졌다. 예를 들어, LabelMe는 수십만 장의 완전 분할된 이미지로 이루어져 있으며, ImageNet은 2,2000개 이상의 카테고리에 걸쳐 1,500만 장 이상의 라벨링된 고해상도 이미지를 포함한다.

수백만 장의 이미지로부터 수천 개 객체를 학습하려면 큰 학습 용량(large capacity)을 가진 모델이 필요하다. 하지만 객체 인식의 복잡성은 ImageNet처럼 큰 데이터셋으로도 충분히 기술되지 못한다. 따라서 모델은 부족한 데이터를 보완할 사전 지식을 포함해야 한다. 컨볼루션 신경망(CNNs)은 이러한 모델의 대표적인 예이다. CNN의 용량은 깊이와 폭을 조절해 관리할 수 있으며, 이미지의 특성을 반영하는 올바른 가정을 포함한다(통계적 특성의 stationarity, 픽셀 간 의존성의 locality). 따라서 동일한 크기의 층을 가진 일반 feedforward 신경망과 비교하면, CNN은 연결 수와 파라미터가 훨씬 적어 학습이 용이하다. 이론적으로 최적의 성능에서도 크게 손해 보지 않는다.

이러한 장점에도 불구하고, 고해상도 이미지를 대규모로 처리하기에는 CNN조차도 계산 비용이 매우 컸다. 다행히 최근 GPU 발전과 2D convolution의 고도로 최적화된 구현 덕분에 대규모 CNN 학습이 가능해졌으며, ImageNet 같은 대규모 라벨 데이터셋 덕분에 과적합 없이 모델을 훈련시킬 수 있게 되었다.

이 논문의 구체적 기여는 다음과 같다.

- 우리는 ILSVRC-2010, ILSVRC-2012 서브셋에서 가장 큰 CNN 중 하나를 학습시켜, 해당 데이터셋에서 지금까지 보고된 최고 성능을 달성했다.
- CNN 학습에 필요한 2D convolution 및 연산들을 GPU로 고도로 최적화하여 공개했다.
- 네트워크 성능을 향상시키고 학습 시간을 줄이는 새로운/비정형적 구조적 요소들을 도입했다(Section 3에서 설명).
- 120만 장 라벨 이미지에도 불구하고 네트워크 규모 때문에 과적합이 발생했는데, 이를 방지하기 위한 효과적 기법들을 사용했다(Section 4).
- 최종 네트워크는 5개의 convolutional layer와 3개의 fully-connected layer로 이루어졌으며, 각 convolutional layer는 전체 파라미터의 1% 미만을 차지했음에도 어느 한 층을 제거하면 성능 저하가 발생했다 → 깊이가 성능에 매우 중요.

네트워크의 크기는 주로 GPU 메모리와 학습 시간에 의해 제한된다. 우리의 네트워크를 학습시키는 데 두 개의 GTX 580 (3GB) GPU로 5~6일이 소요되었다. 우리의 실험은, 더 빠른 GPU와 더 큰 데이터셋만 있으면 성능이 더 향상될 수 있음을 시사한다.

<aside>

<aside>

### 1) **Stationarity of statistics (통계적 특성의 불변성)**

- 의미: **이미지의 통계적 패턴(경계, 질감, 색 분포 등)은 공간 전체에서 비슷하다**는 가정.
- 예: 고양이의 귀(edge 패턴)든 자동차의 바퀴(circle-like edge)든, **이미지의 어느 위치**에 있더라도 같은 특징 검출기가 적용 가능해야 함.
- CNN에서 이를 구현하는 방식:
    - **가중치 공유(weight sharing)** → 같은 convolution filter(커널)를 이미지 전역에 적용.
    - 덕분에 파라미터 수가 급격히 줄고, 학습된 특징은 위치에 무관하게 사용 가능 (translation invariance).

---

### 2) **Locality of pixel dependencies (픽셀 간 의존성의 지역성)**

- 의미: **픽셀은 멀리 떨어진 것보다 가까운 픽셀끼리 더 강한 상관관계**를 가진다는 가정.
- 예: 이웃한 픽셀은 같은 물체(고양이의 털, 자동차 표면)에 속할 확률이 높음. 반면, 왼쪽 위 픽셀과 오른쪽 아래 픽셀은 독립적일 가능성이 큼.
- CNN에서 이를 구현하는 방식:
    - **지역적 연결(local receptive field)** → 각 뉴런은 이미지 전체가 아닌 **작은 영역(예: 3×3, 5×5)**만 본다.
    - 이 지역적 특징들이 계층적으로 조합되면서, 저수준(edge) → 중간수준(texture, parts) → 고수준(object) 표현으로 발전.

---

### 3) **CNN의 장점으로 연결**

- 위 두 가지 가정 덕분에:
    - **적은 파라미터**로도 큰 모델 표현력 확보 (weight sharing으로 절약).
    - **학습 효율**이 개선 → 전체 이미지를 다 보지 않아도, 작은 필터로 국소 특징을 잡아내고, 이걸 전체 이미지에 반복 적용.
    - **일반화 성능 향상** → 모델이 데이터셋에만 특화되지 않고, 위치·배경 변화에도 강인.
</aside>

- 최근까지 라벨링된 데이터는 규모가 작다. 또한 이러한 규모의 데이터셋은 단순한 문제로 충분하며 인간의 성능에 근접했다.
- 그러나 실제환경은 큰 변화를 보이므로 큰 데이터셋이 필요하고 ImageNet같은 대규모 데이터셍으로 인해 가능해졌다.
- 큰 데이터셋을 학습하려면 큰 용량이 가진 모델이 필요한데, CNN은 이미지의 특성을 반영하여 올바른 가정을 가지며, 깊이와 폭을 조절할 수 있고, 인반적인 레이어와 비교하면 파라미터수가 적어 훨씬 학습이 용이하다.
- 이러한 장점에도 불구하고, 아직까지도 CNN은 계산비용이 크나 GPU의 발전과 cnn2d의 고도로 최적화된 구현으로 학습이 가능해졌고, 대규모 데이터셋 덕분에 모델을 과적합없이 훈련시킬 수 있게 되었다.

| 항목             | 내용                                                                                |
| ---------------- | ----------------------------------------------------------------------------------- |
| 기존 한계        | MNIST, CIFAR 등 소규모 데이터셋 → 단순 과제에는 충분하지만, 실제 사물 인식은 불가능 |
| 대규모 데이터    | ImageNet (15M+ 이미지, 22k 클래스) → 본격적인 CNN 학습 가능                         |
| 필요한 모델 특성 | 큰 용량(capacity), 사전 지식 내장(translation invariance, locality 반영)            |
| CNN 장점         | 적은 파라미터, 지역적 연결, 이미지 통계 가정 활용 → 일반 신경망 대비 효율적         |
| 기술적 돌파구    | GPU + 고속 2D convolution 구현 → 대규모 CNN 훈련 현실화                             |
| 논문 기여        | - 대규모 CNN 학습 성공                                                              |
</aside>

---

---

## 🔬 실험과정

### 📚 3. The Architecture

### 번역

우리 네트워크의 구조는 Figure 2에 요약되어 있다. 총 여덟 개의 학습 가능한 계층이 있으며 — 다섯 개는 convolutional layer, 세 개는 fully-connected layer이다. 아래에서는 네트워크 구조의 새롭거나 특이한 특징들을 설명한다. Section 3.1–3.4는 중요도 순으로 나열했으며, 가장 중요한 것부터 시작한다.

**3.1 ReLU Nonlinearity**

전통적으로 뉴런의 출력 f(x)는 입력 x의 함수로 tanh(x)나 sigmoid f(x) = (1 + e^(-x))^(-1) 등을 사용한다. 그러나 이러한 포화(saturating) 비선형 함수는 경사하강법을 통한 학습에서 비포화(non-saturating) 함수 f(x) = max(0, x)에 비해 훨씬 느리다. Nair와 Hinton(2010)을 따라, 우리는 이 함수를 사용하는 뉴런을 Rectified Linear Units(ReLUs)라 부른다.

ReLU를 사용하는 깊은 CNN은 동일한 구조의 tanh 뉴런 기반 네트워크보다 학습이 몇 배나 빠르다. 이는 Figure 1에서 CIFAR-10 데이터셋에 대해 4계층 CNN을 학습한 결과로 보여준다. 25%의 훈련 오류율에 도달하는 데 필요한 반복 횟수를 보면, ReLU 기반 네트워크가 tanh 기반 네트워크보다 6배 빠르다. 만약 전통적인 포화형 뉴런을 사용했다면, 우리는 이 연구에서 다룬 것처럼 큰 네트워크를 실험할 수 없었을 것이다.

우리가 처음 ReLU 이외의 비전통적 뉴런을 고려한 것은 아니다. 예를 들어 Jarrett et al.은 f(x) = |tanh(x)| 함수가 특정 normalization + pooling 구조와 함께 잘 작동한다고 보고했다. 그러나 그 경우 초점은 과적합 방지였으며, 우리가 ReLU에서 보고한 **훈련 데이터 적합 속도의 가속** 효과와는 다르다. 큰 모델을 대규모 데이터셋에서 학습할 때는, 이러한 빠른 학습 속도가 매우 중요한 영향을 미친다.

**3.2 Training on Multiple GPUs**

단일 GTX 580 GPU는 메모리가 3GB에 불과하여 학습 가능한 네트워크 크기에 제한이 있다. 그런데 ImageNet의 120만 장 훈련 이미지는, 한 GPU 메모리에 들어가지 못할 정도로 큰 네트워크를 학습시키기에 충분하지않다. 따라서 우리는 네트워크를 두 GPU에 분산시켰다. GPU는 서로 간 메모리에 직접 읽기/쓰기가 가능하기 때문에, 호스트 메모리를 거치지 않고도 교차-GPU 병렬화를 잘 지원한다.

우리가 사용하는 병렬화 기법은 각 GPU에 절반의 커널(또는 뉴런)을 배치하는 것이다. 다만, GPU 간 통신은 특정 계층에서만 일어난다. 예를 들어, 3번째 convolutional layer의 커널은 2번째 layer의 모든 커널 맵에서 입력을 받지만, 4번째 layer의 커널은 같은 GPU에 있는 3번째 layer의 커널 맵에서만 입력을 받는다. 이러한 연결 패턴은 cross-validation으로 조정하며, 통신량이 계산량에 비해 적절한 수준이 되도록 한다.

이 아키텍처는 Cireşan et al.(2011)이 제안한 “columnar” CNN과 유사하나, 우리의 경우 column이 완전히 독립적이지 않다. 이 기법을 쓰면, 단일 GPU에서 반 크기의 커널을 가진 네트워크에 비해 top-1 오류율이 1.7%, top-5 오류율이 1.2% 낮아진다. 또한 두 GPU를 쓴 네트워크가 학습 시간도 약간 더 짧다.

**3.3 Local Response Normalization**

ReLU는 포화 문제를 피할 수 있어서 입력 정규화가 꼭 필요하지 않다. 입력이 양수인 경우 학습이 일어나기 때문이다. 그러나 우리는 다음과 같은 지역 정규화(local normalization) 기법이 일반화에 도움이 된다는 것을 발견했다.

뉴런 i의 위치 (x,y)에서의 활성값을 aᵢ₍ₓ,ᵧ₎라 하고, 이를 ReLU에 통과시킨 뒤 response-normalized 값 bᵢ₍ₓ,ᵧ₎는 다음과 같이 정의된다.

bᵢ₍ₓ,ᵧ₎ = aᵢ₍ₓ,ᵧ₎ / ( k + α Σⱼ (aⱼ₍ₓ,ᵧ₎)² )^β

(합은 같은 위치 (x,y)에서 인접한 n개의 커널 맵에 대해 계산됨)

여기서 N은 해당 layer의 전체 커널 수이며, 하이퍼파라미터 k=2, n=5, α=10⁻⁴, β=0.75를 사용했다. 이 정규화는 생물학적 뉴런의 측면억제(lateral inhibition)에서 영감을 얻은 것으로, 서로 다른 커널 출력들 간에 경쟁을 유도한다. 우리는 ReLU 이후 특정 계층에 이 정규화를 적용했다.

이 기법은 Jarrett et al.(2009)이 제안한 local contrast normalization과 유사하지만, 평균값을 빼지 않으므로 “밝기 정규화(brightness normalization)”라고 부르는 것이 더 적절하다. 이 정규화는 top-1 오류율을 1.4%, top-5 오류율을 1.2% 낮췄다. 또한 CIFAR-10 데이터셋에서도 효과를 검증했는데, 정규화를 쓰지 않은 4층 CNN은 13% 오류율, 정규화를 쓰면 11%로 개선되었다.

**3.4 Overlapping Pooling**

CNN에서 pooling layer는 동일한 커널 맵(kernel map) 내 인접한 뉴런 그룹들의 출력을 요약한다. 전통적으로는 인접한 pooling unit들이 요약하는 영역이 서로 겹치지 않는다(non-overlapping). 좀 더 구체적으로 말해, pooling layer는 s 픽셀 간격으로 배치된 pooling unit들의 격자로 구성되며, 각 unit은 크기 z × z의 영역을 요약한다.

만약 s = z라면, 전통적인 non-overlapping pooling이 된다. 반대로 s < z라면, pooling unit들이 겹치는(overlapping) pooling을 얻게 된다. 우리는 네트워크 전반에 걸쳐 후자의 방식을 사용했으며, s = 2, z = 3으로 설정했다.

이 방식은 non-overlapping (s = 2, z = 2) pooling과 출력 차원은 동일하지만, top-1 오류율은 0.4%, top-5 오류율은 0.3% 낮아졌다. 또한 학습 과정에서 overlapping pooling을 사용하면 모델이 과적합되기가 조금 더 어려워진다는 것을 관찰했다.

**3.5 Overall Architecture**

이제 CNN 전체 구조를 설명한다. Figure 2에 나타난 것처럼, 네트워크는 가중치를 가진 8개의 계층으로 이루어진다. 처음 다섯 개는 convolutional layer이고, 나머지 세 개는 fully-connected layer이다. 마지막 fully-connected layer의 출력은 1000-way softmax로 입력되며, 이는 1000개의 클래스 라벨에 대한 확률 분포를 생성한다. 네트워크는 다항 로지스틱 회귀(multi-class logistic regression) 목적 함수를 최대화하는데, 이는 곧 학습 데이터 각 사례에서 정답 라벨의 로그 확률을 최대화하는 것과 같다.

두 번째, 네 번째, 다섯 번째 convolutional layer의 커널들은 같은 GPU에 존재하는 이전 layer의 커널 맵에만 연결된다. 세 번째 convolutional layer의 커널은 두 번째 layer의 모든 커널 맵과 연결된다. Fully-connected layer의 뉴런은 이전 계층의 모든 뉴런과 연결된다. Response-normalization layer는 첫 번째와 두 번째 convolutional layer 뒤에 배치된다. Max-pooling layer는 이들 response-normalization layer 뒤와, 다섯 번째 convolutional layer 뒤에 위치한다. 모든 convolutional layer와 fully-connected layer의 출력에는 ReLU 비선형 함수가 적용된다.

첫 번째 convolutional layer는 224×224×3 크기의 입력 이미지를 받아, 크기 11×11×3의 커널 96개로, 스트라이드 4를 사용해 필터링한다(여기서 스트라이드는 인접 뉴런의 수용영역 중심 간 거리이다). 두 번째 convolutional layer는 첫 번째 convolutional layer의 출력(정규화 및 pooling 처리된 결과)을 입력받아, 크기 5×5×48의 커널 256개로 필터링한다.

세 번째, 네 번째, 다섯 번째 convolutional layer는 pooling이나 normalization layer 없이 연속적으로 연결된다. 세 번째 convolutional layer는 두 번째 convolutional layer의 출력(정규화 및 pooling된 결과)에 연결되며, 3×3×256 크기의 커널 384개를 갖는다. 네 번째 convolutional layer는 3×3×192 크기의 커널 384개를 갖는다. 다섯 번째 convolutional layer는 3×3×192 크기의 커널 256개를 갖는다.

Fully-connected layer는 각각 4096개의 뉴런을 포함한다.

<aside>

- ReLU : 포화 비선형 함수는 경사하강법에서 비포화함수에 비해 훨씬 느리다. ReLU는 같은 깊이의 Tanh에 비해 몇배나 학습속도가 빠르다.

![image.png]((AlexNet)%20ImageNet%20Classification%20with%20Deep%20Convol%2025ca9b246de1800b878cd59e8addb4dd/image%201.png)

| 항목          | 내용                                                                                   |
| ------------- | -------------------------------------------------------------------------------------- |
| 네트워크 계층 | 총 8개 학습 계층 → 5 convolution + 3 fully-connected                                   |
| 출력          | 마지막 fully-connected layer → 1000-way softmax (ImageNet 클래스 확률)                 |
| ReLU 정의     | f(x) = max(0, x) (비포화형 비선형 함수)                                                |
| 기존 방식     | sigmoid, tanh → 포화 영역에서 gradient ≈ 0 → 학습 느림 (경사 소실 문제)                |
| 장점          | ReLU는 양수 영역에서 gradient=1 → 학습 속도 수배 향상                                  |
| 실험 근거     | CIFAR-10 실험에서 tanh 대비 6배 빠른 수렴 속도 확인                                    |
| 의의          | GPU + 대규모 데이터셋과 결합하여, “실제로 훈련 가능한” 대규모 CNN을 가능하게 만든 핵심 |

<aside>

- columnar
    - “Column” = 독립적으로 학습된 하나의 CNN.
    - 여러 개 column을 병렬로 두고, 각각의 CNN이 같은 입력 이미지를 처리.
    - 마지막에 각 column의 softmax 출력(probability distribution)을 평균하거나 투표하여 최종 예측.
    - 장점: 단일 CNN보다 일반화 성능이 뛰어나고, 과적합을 줄일 수 있음.
    - 단점: 각 column이 완전히 독립적이므로, 계산량이 크고 column 수에 비례해 자원이 필요.
</aside>

- GPU는 서로간의 메모리에서 직접 읽고 쓰기가 가능하기 때문에, 호스트 메모리를 거치지 않고 교차 GPU를 잘 지원한다.
- GPU 2개에 대해 절반의 커널을 위치시켜 사용하며 특정 층에서는 gpu간 공유가 가능하다. 2-3번 layer가 해당부분으로 3번 layer는 2번의 모든 gpu에 대해서 입력을 받지만, 4번 layer는 같은 gpu내에서만 입력을 받는다.
- 이러한 패턴은 cross-validation으로 조절한다. 성능이 향상되었으며, 시간또한 약간 짧아졌다.
- columnar == 독립구조, AlexNet == 협력구조(featuremap을 나눔)
- Local Response Normalization을 사용하였음
    
    ![스크린샷 2025-08-27 오후 3.54.26.jpg]((AlexNet)%20ImageNet%20Classification%20with%20Deep%20Convol%2025ca9b246de1800b878cd59e8addb4dd/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-08-27_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.54.26.jpg)
    
    - 뉴런간 경쟁을 유도(생물학적 뉴런의 측면억제에서 영감을 받음, 서로 다른 커널 출력간의 경쟁을 유도)
    - CNN → ReLU → LRN (몇몇 층에 사용)

| 항목            | 내용                                                                       |
| --------------- | -------------------------------------------------------------------------- |
| GPU 병렬 학습   | 2 GPU에 절반씩 커널 분산, 특정 계층만 교차 연결 → 성능 향상, 학습시간 단축 |
| 비교            | 단일 GPU 대비 top-1 오류율 1.7%↓, top-5 1.2%↓                              |
| 구조적 특성     | Cireşan et al.의 columnar CNN 유사, 하지만 column 간 완전히 독립X          |
| 정규화 아이디어 | Local Response Normalization (LRN), lateral inhibition에서 영감            |
| 정규화 식       | bᵢ = aᵢ / (k + α Σⱼ (aⱼ²))^β, 하이퍼파라미터 (k=2, n=5, α=1e-4, β=0.75)    |
| 효과            | ILSVRC top-1 1.4%↓, top-5 1.2%↓ / CIFAR-10 오류율 13%→11%                  |
- pooling을 overlap해서 진행하였다. 이러한 기법으로 일반화 성능을 향상시켰다.

| 항목         | 내용                                                                |
| ------------ | ------------------------------------------------------------------- |
| Pooling 정의 | 일정한 영역의 뉴런 출력을 요약 (대표적으로 max-pooling)             |
| 전통 방식    | non-overlapping (s = z) → 격자 영역이 겹치지 않음                   |
| AlexNet 방식 | overlapping pooling (s < z) 사용 → s=2, z=3                         |
| 효과         | top-1 오류율 0.4%↓, top-5 오류율 0.3%↓                              |
| 추가 관찰    | 겹치는 pooling은 네트워크가 학습 데이터에 과적합되기 조금 더 어려움 |

![image.png]((AlexNet)%20ImageNet%20Classification%20with%20Deep%20Convol%2025ca9b246de1800b878cd59e8addb4dd/image%202.png)

- 큰 필터와 큰 stride로 시작 → 점차 작은 필터로 깊이 있게 연결 → fully-connected layer에서 high-level representation 형성 → softmax 출력
- Conv3는 두 GPU 모두 연결, Conv4·5는 같은 GPU 내부 연결” 정도를 표에 넣어주면 더 완성도가 높습니다.

| 항목                  | 내용                                                             |
| --------------------- | ---------------------------------------------------------------- |
| 전체 구조             | 5 convolutional + 3 fully-connected → 마지막 softmax(1000-way)   |
| 입력                  | 224×224×3 RGB 이미지                                             |
| Conv1                 | 96개의 11×11×3 필터, stride=4                                    |
| Conv2                 | 256개의 5×5×48 필터 (이전 layer 출력 일부만 연결, GPU 분산 구조) |
| Conv3                 | 384개의 3×3×256 필터 (양쪽 GPU 모두 연결)                        |
| Conv4                 | 384개의 3×3×192 필터 (GPU 내부 연결)                             |
| Conv5                 | 256개의 3×3×192 필터 (GPU 내부 연결, 이후 pooling)               |
| FC1–2                 | 각각 4096 뉴런 (dropout 적용)                                    |
| FC3                   | 1000 뉴런 → softmax                                              |
| ReLU                  | 모든 conv, FC layer에 적용                                       |
| Normalization/Pooling | Conv1, Conv2 뒤에 LRN + pooling, Conv5 뒤에 pooling              |
</aside>

### 📚 4. Reducing Overfitting

### 번역

우리 신경망 아키텍처는 총 6천만 개의 파라미터를 갖고 있다. ILSVRC의 1000개 클래스는 각 학습 예제마다 이미지에서 라벨로 가는 매핑에 약 10비트의 제약을 제공하지만, 이는 이렇게 많은 파라미터를 학습하기에는 충분하지 않아 과적합(overfitting)이 심각하게 발생한다. 따라서 우리는 과적합을 줄이기 위해 두 가지 주요 방법을 사용했다.

**4.1 Data Augmentation**

이미지 데이터에서 과적합을 줄이는 가장 간단하고 일반적인 방법은 라벨을 보존하는 변환(label-preserving transformation)을 이용해 인위적으로 데이터셋을 확장하는 것이다. 우리는 두 가지 형태의 데이터 증강(data augmentation)을 적용했으며, 이들은 원본 이미지로부터 계산량 거의 없이 변형된 이미지를 생성할 수 있다. 따라서 이 변형 이미지는 디스크에 저장할 필요가 없으며, CPU가 변형 이미지를 생성하는 동안 GPU는 이전 배치를 학습한다. 결과적으로 이 데이터 증강은 사실상 추가 계산 비용이 들지 않는다.

첫 번째 증강 방식은 **이미지 평행 이동 및 좌우 반전**이다. 구체적으로, 256×256 이미지에서 임의의 224×224 패치(및 그 좌우 반전본)를 잘라내어 학습에 사용했다. 이로써 학습 데이터셋 크기는 약 2048배로 증가한다. 만약 이 과정을 쓰지 않으면, 네트워크는 심각한 과적합을 보였을 것이며, 우리는 훨씬 작은 네트워크를 쓸 수밖에 없었을 것이다. 테스트 시에는 224×224 패치를 다섯 개(네 구석 + 중앙) 잘라내고, 각각의 좌우 반전을 포함해 총 10개 패치를 네트워크에 통과시킨 뒤 softmax 출력 평균으로 최종 예측을 했다.

두 번째 증강 방식은 **RGB 채널 강도의 변형**이다. ImageNet 훈련셋 전체 픽셀 값에 대해 PCA를 수행한 뒤, 각 학습 이미지의 픽셀 값에 PCA의 주성분 벡터를 일정 비율로 더해주는 방식이다. 수식으로는, 각 RGB 픽셀 Iₓᵧ = [Iᴿₓᵧ, Iᴳₓᵧ, Iᴮₓᵧ]^T 에 대해,

[p1,p2,p3][α1λ1, α2λ2, α3λ3]^T

를 더해주었다. 여기서 pi, λi는 각각 RGB 픽셀 공분산 행렬의 고유벡터와 고유값이고, αi는 N(0,0.1²)에서 뽑은 난수이다. 이 방식은 조명 세기나 색깔이 달라져도 물체 정체성이 유지되는 자연 이미지의 성질을 근사한다. 이 증강을 적용하면 top-1 오류율이 1% 이상 감소했다.

**4.2 Dropout**

여러 모델의 예측을 결합하면 성능이 향상되는 것은 잘 알려져 있지만, 대규모 신경망에서는 계산 비용이 너무 크다. “Dropout”(Hinton et al., 2012)은 매우 효율적인 모델 앙상블 방법으로, 학습 시 각 hidden neuron의 출력을 확률 0.5로 0으로 만들어버린다. 이렇게 “dropout”된 뉴런은 forward pass에도 기여하지 않고, backpropagation에도 참여하지 않는다. 결과적으로 매번 입력을 처리할 때마다 서로 다른 아키텍처의 신경망이 샘플링되는 셈이지만, 이들 모두 가중치를 공유한다.

이 방식은 특정 뉴런이 다른 뉴런의 존재에 지나치게 의존(co-adaptation)하지 못하게 하여, 더 강인한 특징을 학습하게 만든다. 테스트 시에는 모든 뉴런을 사용하지만, 출력은 0.5를 곱해준다(이는 dropout을 적용한 많은 모델의 예측 분포를 기하평균하는 것과 유사하다).

우리는 Figure 2의 첫 두 fully-connected layer에서 dropout을 사용했다. Dropout이 없을 때는 네트워크가 심각하게 과적합되었으며, Dropout을 적용하면 학습 수렴 속도가 약 두 배 느려지지만 과적합은 크게 줄어들었다.

<aside>

## 4.1. 증강

- 이미지 crop 및 좌우 반전 : 224x224패치를 임의로 만들고, 또한 그걸 반전 시켜 사용
    - 2048증가 효과 테스트시에는 4구성 + 중앙 패치를 잘라내고 각각의 좌우 반전을 포함해 10개의 패치를 통과하여 softmax출력 평균을 최종 예측
- RGB채널 강도 변환 : 훈련셋 전체 픽셀 값에 대해 PCA를 수행하여, 각 이미지의 픽셀 값에 PCA의 주성분 벡터를 일정 비율로 더해줌(난수 활용)
- 이 효과를 통해 조명 세기나 색이 달라져도 물체 정체성이 유지되는 자연 이미지의 성질을 모방

![image.png]((AlexNet)%20ImageNet%20Classification%20with%20Deep%20Convol%2025ca9b246de1800b878cd59e8addb4dd/image%203.png)

## 4.2. Dropout

- 전파 및 역전파에 일정 뉴런들이 참여하지 않음. 따라서 매번 입력시 다른 아키텍쳐를 가지는 효과를 가짐 but 가중치는 공유된다.
- 즉 특정 뉴런이 다른 뉴런을 지나치게 의존하지 못하게 하여 일반화 성능을 높인다.

## 정리

| 항목                                                | 내용                                                    |
| --------------------------------------------------- | ------------------------------------------------------- |
| 문제                                                | 파라미터 60M → 데이터(120만 장)로는 과적합 심각         |
| 해결책1                                             | **Data Augmentation**                                   |
| – 무작위 crop & 좌우 반전 (훈련 데이터 2048배 확장) |                                                         |
| – PCA 기반 색상 증강 (조명·색 변화 불변성 반영)     |                                                         |
| 효과1                                               | crop 없으면 과적합 심각 / 색상 증강 시 top-1 오류율 1%↓ |
| 해결책2                                             | **Dropout**                                             |
| – 학습 시 뉴런을 확률 0.5로 제거                    |                                                         |
| – 뉴런 간 공적응(co-adaptation) 방지                |                                                         |
| – 일반화 성능 ↑, 하지만 수렴 속도 약 2배 느려짐     |                                                         |
| 적용 위치                                           | Fully-connected 1, 2층                                  |

</aside>

### 📚 5. Details of learning

### 번역

우리는 모델을 **확률적 경사하강법(Stochastic Gradient Descent, SGD)** 으로 학습했다. 배치 크기는 128, 모멘텀(momentum)은 0.9, 가중치 감쇠(weight decay)는 0.0005로 설정했다. 이 작은 weight decay는 단순한 정규화 이상의 역할을 했다. 즉, 모델이 실제로 학습을 하도록 도와주었으며, 단순히 일반화 향상을 위한 정규화가 아니었다.

가중치 w에 대한 업데이트 규칙은 다음과 같다:

v_{i+1} := 0.9·v_i − 0.0005·ε·w_i − ε·<∂L/∂w | w_i>*D_i
w*{i+1} := w_i + v_{i+1}

여기서 i는 반복 인덱스, v는 모멘텀 변수, ε은 학습률, <∂L/∂w | w_i>_D_i 는 i번째 배치 D_i에서의 손실 L에 대한 w의 기울기 평균을 의미한다.

**가중치 초기화:**

각 계층의 가중치는 평균 0, 표준편차 0.01의 Gaussian 분포에서 샘플링해 초기화했다. 두 번째, 네 번째, 다섯 번째 convolutional layer와 fully-connected hidden layer의 뉴런 bias는 상수 1로 초기화했다. 이렇게 하면 ReLU 뉴런이 초기 학습 단계에서 양수 입력을 더 잘 받게 되어 학습이 빨라진다. 나머지 layer의 bias는 0으로 초기화했다.

**학습률 스케줄:**

모든 layer에 동일한 학습률을 사용했고, 학습 중 수동으로 조정했다. 구체적으로, 검증 오류(validation error)가 개선되지 않으면 학습률을 10배 낮췄다. 초기 학습률은 0.01이었고, 종료 전까지 총 세 번 낮췄다.

전체 학습은 약 120만 장의 이미지 데이터셋을 90 epoch 순환하는 동안 진행되었으며, 3GB 메모리를 가진 NVIDIA GTX 580 GPU 두 장에서 5~6일이 소요되었다.

<aside>

## 5. 학습 세부사항

- SGD로 학습, 특히 작은 weight decay는 정규화 이상의 역할을 하였음 → 실제로 학습을 하도록 도와주었음

![image.png]((AlexNet)%20ImageNet%20Classification%20with%20Deep%20Convol%2025ca9b246de1800b878cd59e8addb4dd/image%204.png)

- 파라미터들은 가우시안 분포에서 샘플링해 초기화, 2, 4, 5와 fc에서는 bias를 1로 나머지넌 0으로 초기화(bias = 1 : ReLU뉴런이 초기 학습 단계에서 양수 입력을 더 잘 받게 되어 학습이 빨라지는 효과)
- lr은 수동으로 조정. val err가 개선되지 않으면 학습률을 10배로 낮췄음
</aside>

---