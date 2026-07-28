---
title: "(AlexNet) ImageNet Classification with Deep Convolutional Neural Networks"
date: 2025-08-29 18:00:00 +0900
categories: [AI-ML-DL, etc.]
tags: [Computer-Vision, paper]
---

## 📚 정리

### 📌 제목

> **ImageNet Classification with Deep Convolutional Neural Networks**

-----

### 🌟 초록

>   - 1000-way softmax : 1000개의 cls.문제이기 때문에, output을 1000개의 vec.으로 만들었다.
>       - **Top-1 error**: 가장 큰 확률이 정답이 아닐 때.
>       - **Top-5 error**: 가장 큰 확률 5개 안에 정답이 없을 때.
>   - **saturating neuron** : 뉴런의 활성화 함수중에서 입력값이 커지거나 작아지면 포화되는 현상으로, sigmoid, tanh에 해당 → 경사소실 문제 따라서 AlexNet에서는 ReLU 채택
>
> | 항목 | 내용 |
> | --- | --- |
> | 데이터셋 | ImageNet LSVRC-2010 (120만 장, 1000 클래스), ILSVRC-2012 대회 참가 |
> | 모델 규모 | 60M 파라미터, 650k 뉴런 |
> | 아키텍처 | 5개 convolutional layer + 3개 fully-connected layer + softmax(1000-way) |
> | 핵심 기법 | ReLU(non-saturating neuron), GPU 기반 효율적 학습, Dropout 정규화 |
> | 성능 | ILSVRC-2010: top-1 37.5%, top-5 17.0% / ILSVRC-2012: top-5 15.3% (1등) |
> | 기여 | CNN을 대규모 데이터·GPU와 결합 → 기존 feature-engineering 기반 방법보다 획기적으로 우수한 성능 달성 |
>
>   - **아키텍쳐**
> ![AlexNet Architecture](/assets/img/posts/alexnet/image.png)

<br>



-----

### 💡 결론 & 고찰

> | 항목 | 내용 |
> | --- | --- |
> | ILSVRC-2010 | CNN: top-1 37.5%, top-5 17.0% (기존 최고: 47.1%, 28.2%) |
> | ILSVRC-2012 | 단일 CNN: top-5 18.2% → 5개 앙상블: 16.4% → 사전학습+앙상블: 15.3% (2위는 26.2%) |
> | ImageNet 2009 (10k 클래스) | CNN: top-1 67.4%, top-5 40.9% (기존 78.1%, 60.9%) |
> | Qualitative 분석 | CNN은 색상·방향·주파수 selective kernel 학습 / GPU별 specialization 발생 |
> | Feature space | 4096차원 feature vector로 이미지 간 semantic similarity 반영, raw pixel distance보다 의미 있는 검색 가능 |
> | 추가 제안 | auto-encoder로 feature vector 압축 → 효율적 이미지 검색(Image retrieval) |
>
>   - 네트워크를 깊게 만든다면 효과가 더 클것으로 예상되어 진다. 또한 unsupervised pretraining도 효과적으로 작동할것으로 판단된다.
>   - CNN의 깊이가 성능향상에 기여한다.
>
> | 항목 | 내용 |
> | --- | --- |
> | 깊이의 중요성 | convolution layer 하나만 제거해도 top-1 성능 약 2% 저하 → 깊은 구조가 핵심 |
> | 비지도 사전학습 | 본 논문에서는 사용하지 않았음. 하지만 더 큰 네트워크·라벨 부족 환경에서 유용할 것으로 예상 |
> | 성능 스케일링 | 네트워크를 크게 하고 학습 시간을 늘리면 성능이 계속 향상됨 (scale-up 효과) |
> | 한계 | 인간 뇌의 시각 피질(infero-temporal pathway)에 비하면 아직 멀었다고 언급 |
> | 미래 방향 | 정적 이미지가 아닌 **비디오 학습**으로 확장 → 시계열적 단서(temporal structure) 활용 |

-----

### 🗃️ 데이터

> >   - **Amazon Mechanical Turk** : **작업을 잘게 쪼개어(HITs, Human Intelligence Tasks) 온라인으로 다수의 사람(worker)에게 할당**, 소액 보상으로 처리하게 하는 시스템. 적은 데이터셋으로 대규모데이터셋 구성가능
>
>   - CNN모델에 input으로 넣기 위해서, **256x256으로 다운샘플링** 진행
>       - 직사각형의 경우에는 이미지가 짧은변을 256으로 resizing한 후, 그 결과물에서 중앙 256x256패치를 잘라냄
>   - 전처리는 하지 않고, 훈련셋 전체에서 **픽셀 단위 평균값을 빼는 작업**만 수행해서, 네트워크는 중심화된 원시 RGB값으로 학습되어짐
>
> | 항목 | 내용 |
> | --- | --- |
> | 원본 데이터셋 | ImageNet (15M+ 이미지, 22k 클래스) |
> | ILSVRC 버전 | 2010/2012 대회용 하위셋 (1000 클래스, 약 120만 train, 50k val, 150k test) |
> | 특징 | ILSVRC-2010만 test 라벨 공개 (실험 검증용) |
> | 오류 지표 | Top-1 (최고 확률 예측이 틀린 경우), Top-5 (상위 5개 예측 안에 정답 없음) |
> | 전처리 | 모든 이미지를 256×256으로 리사이즈 후 중앙 crop / train set 픽셀 평균값을 빼고 raw RGB 사용 |
> | 의의 | 당시로서는 unprecedented(전례 없는) 대규모 labeled dataset. CNN 학습 가능케 한 기반. |

-----

### 📌 서론

> > ### 1\) **Stationarity of statistics (통계적 특성의 불변성)**
>
> >   - 의미: **이미지의 통계적 패턴(경계, 질감, 색 분포 등)은 공간 전체에서 비슷하다**는 가정.
> >   - 예: 고양이의 귀(edge 패턴)든 자동차의 바퀴(circle-like edge)든, **이미지의 어느 위치**에 있더라도 같은 특징 검출기가 적용 가능해야 함.
> >   - CNN에서 이를 구현하는 방식:
> >       - **가중치 공유(weight sharing)** → 같은 convolution filter(커널)를 이미지 전역에 적용.
> >       - 덕분에 파라미터 수가 급격히 줄고, 학습된 특징은 위치에 무관하게 사용 가능 (**translation invariance**).
>
> > 
>
> > -----
>
> > ### 2\) **Locality of pixel dependencies (픽셀 간 의존성의 지역성)**
>
> >   - 의미: **픽셀은 멀리 떨어진 것보다 가까운 픽셀끼리 더 강한 상관관계**를 가진다는 가정.
> >   - 예: 이웃한 픽셀은 같은 물체(고양이의 털, 자동차 표면)에 속할 확률이 높음. 반면, 왼쪽 위 픽셀과 오른쪽 아래 픽셀은 독립적일 가능성이 큼.
> >   - CNN에서 이를 구현하는 방식:
> >       - **지역적 연결(local receptive field)** → 각 뉴런은 이미지 전체가 아닌 \*\*작은 영역(예: 3×3, 5×5)\*\*만 본다.
> >       - 이 지역적 특징들이 계층적으로 조합되면서, 저수준(edge) → 중간수준(texture, parts) → 고수준(object) 표현으로 발전.
>
> > 
>
> > -----
>
> > ### 3\) **CNN의 장점으로 연결**
>
> >   - 위 두 가지 가정 덕분에:
> >       - **적은 파라미터**로도 큰 모델 표현력 확보 (weight sharing으로 절약).
> >       - **학습 효율**이 개선 → 전체 이미지를 다 보지 않아도, 작은 필터로 국소 특징을 잡아내고, 이걸 전체 이미지에 반복 적용.
> >       - **일반화 성능 향상** → 모델이 데이터셋에만 특화되지 않고, 위치·배경 변화에도 강인.
>
>   - 최근까지 라벨링된 데이터는 규모가 작다. 또한 이러한 규모의 데이터셋은 단순한 문제로 충분하며 인간의 성능에 근접했다.
>   - 그러나 실제환경은 큰 변화를 보이므로 큰 데이터셋이 필요하고 **ImageNet**같은 대규모 데이터셋으로 인해 가능해졌다.
>   - 큰 데이터셋을 학습하려면 큰 용량을 가진 모델이 필요한데, CNN은 이미지의 특성을 반영하여 올바른 가정을 가지며, 깊이와 폭을 조절할 수 있고, 일반적인 레이어와 비교하면 파라미터수가 적어 훨씬 학습이 용이하다.
>   - 이러한 장점에도 불구하고, 아직까지도 CNN은 계산비용이 크나 **GPU의 발전**과 **CNN 2D의 고도로 최적화된 구현**으로 학습이 가능해졌고, 대규모 데이터셋 덕분에 모델을 과적합없이 훈련시킬 수 있게 되었다.
>
> | 항목 | 내용 |
> | --- | --- |
> | 기존 한계 | MNIST, CIFAR 등 소규모 데이터셋 → 단순 과제에는 충분하지만, 실제 사물 인식은 불가능 |
> | 대규모 데이터 | ImageNet (15M+ 이미지, 22k 클래스) → 본격적인 CNN 학습 가능 |
> | 필요한 모델 특성 | 큰 용량(capacity), 사전 지식 내장(translation invariance, locality 반영) |
> | CNN 장점 | 적은 파라미터, 지역적 연결, 이미지 통계 가정 활용 → 일반 신경망 대비 효율적 |
> | 기술적 돌파구 | GPU + 고속 2D convolution 구현 → 대규모 CNN 훈련 현실화 |
> | 논문 기여 | - 대규모 CNN 학습 성공 |

-----

-----

## 🔬 실험과정

### 📚 3. The Architecture

>   - **ReLU** : 포화 비선형 함수는 경사하강법에서 비포화함수에 비해 훨씬 느리다. ReLU는 같은 깊이의 Tanh에 비해 몇배나 학습속도가 빠르다.
>
> <br>
>
> ![ReLU vs Tanh convergence speed](/assets/img/posts/alexnet/image%201.png)
>
> | 항목 | 내용 |
> | --- | --- |
> | 네트워크 계층 | 총 8개 학습 계층 → 5 convolution + 3 fully-connected |
> | 출력 | 마지막 fully-connected layer → 1000-way softmax (ImageNet 클래스 확률) |
> | ReLU 정의 | f(x) = max(0, x) (비포화형 비선형 함수) |
> | 기존 방식 | sigmoid, tanh → 포화 영역에서 gradient ≈ 0 → 학습 느림 (경사 소실 문제) |
> | 장점 | ReLU는 양수 영역에서 gradient=1 → 학습 속도 수배 향상 |
> | 실험 근거 | CIFAR-10 실험에서 tanh 대비 6배 빠른 수렴 속도 확인 |
> | 의의 | GPU + 대규모 데이터셋과 결합하여, “실제로 훈련 가능한” 대규모 CNN을 가능하게 만든 핵심 |
>
> >   - **columnar**
> >       - “Column” = 독립적으로 학습된 하나의 CNN.
> >       - 여러 개 column을 병렬로 두고, 각각의 CNN이 같은 입력 이미지를 처리.
> >       - 마지막에 각 column의 softmax 출력(probability distribution)을 평균하거나 투표하여 최종 예측.
> >       - 장점: 단일 CNN보다 일반화 성능이 뛰어나고, 과적합을 줄일 수 있음.
> >       - 단점: 각 column이 완전히 독립적이므로, 계산량이 크고 column 수에 비례해 자원이 필요.
>
>   - GPU는 서로간의 메모리에서 직접 읽고 쓰기가 가능하기 때문에, 호스트 메모리를 거치지 않고 교차 GPU를 잘 지원한다.
>   - GPU 2개에 대해 절반의 커널을 위치시켜 사용하며 특정 층에서는 gpu간 공유가 가능하다. 2-3번 layer가 해당부분으로 3번 layer는 2번의 모든 gpu에 대해서 입력을 받지만, 4번 layer는 같은 gpu내에서만 입력을 받는다.
>   - 이러한 패턴은 cross-validation으로 조절한다. 성능이 향상되었으며, 시간또한 약간 짧아졌다.
>   - **columnar == 독립구조**, **AlexNet == 협력구조**(featuremap을 나눔)
>   - **Local Response Normalization**을 사용하였음
>       - 뉴런간 경쟁을 유도(생물학적 뉴런의 측면억제에서 영감을 받음, 서로 다른 커널 출력간의 경쟁을 유도)
>       - CNN → ReLU → LRN (몇몇 층에 사용)
>
> <br>
>
> ![Local Response Normalization formula](/assets/img/posts/alexnet/스크린샷_2025-08-27_오후_3.54.26.jpg)
>
> | 항목 | 내용 |
> | --- | --- |
> | GPU 병렬 학습 | 2 GPU에 절반씩 커널 분산, 특정 계층만 교차 연결 → 성능 향상, 학습시간 단축 |
> | 비교 | 단일 GPU 대비 top-1 오류율 1.7%↓, top-5 1.2%↓ |
> | 구조적 특성 | Cireşan et al.의 columnar CNN 유사, 하지만 column 간 완전히 독립X |
> | 정규화 아이디어 | Local Response Normalization (LRN), lateral inhibition에서 영감 |
> | 정규화 식 | bᵢ = aᵢ / (k + α Σⱼ (aⱼ²))^β, 하이퍼파라미터 (k=2, n=5, α=1e-4, β=0.75) |
> | 효과 | ILSVRC top-1 1.4%↓, top-5 1.2%↓ / CIFAR-10 오류율 13%→11% |
>
>   - **overlapping pooling**을 진행하였다. 이러한 기법으로 일반화 성능을 향상시켰다.
>
> | 항목 | 내용 |
> | --- | --- |
> | Pooling 정의 | 일정한 영역의 뉴런 출력을 요약 (대표적으로 max-pooling) |
> | 전통 방식 | non-overlapping (s = z) → 격자 영역이 겹치지 않음 |
> | AlexNet 방식 | overlapping pooling (s \< z) 사용 → s=2, z=3 |
> | 효과 | top-1 오류율 0.4%↓, top-5 오류율 0.3%↓ |
> | 추가 관찰 | 겹치는 pooling은 네트워크가 학습 데이터에 과적합되기 조금 더 어려움 |
>
> <br>
>
> ![AlexNet layer connection diagram](/assets/img/posts/alexnet/image%202.png)
>
>   - 큰 필터와 큰 stride로 시작 → 점차 작은 필터로 깊이 있게 연결 → fully-connected layer에서 high-level representation 형성 → softmax 출력
>   - Conv3는 두 GPU 모두 연결, Conv4·5는 같은 GPU 내부 연결” 정도를 표에 넣어주면 더 완성도가 높습니다.
>
> | 항목 | 내용 |
> | --- | --- |
> | 전체 구조 | 5 convolutional + 3 fully-connected → 마지막 softmax(1000-way) |
> | 입력 | 224×224×3 RGB 이미지 |
> | Conv1 | 96개의 11×11×3 필터, stride=4 |
> | Conv2 | 256개의 5×5×48 필터 (이전 layer 출력 일부만 연결, GPU 분산 구조) |
> | Conv3 | 384개의 3×3×256 필터 (양쪽 GPU 모두 연결) |
> | Conv4 | 384개의 3×3×192 필터 (GPU 내부 연결) |
> | Conv5 | 256개의 3×3×192 필터 (GPU 내부 연결, 이후 pooling) |
> | FC1–2 | 각각 4096 뉴런 (dropout 적용) |
> | FC3 | 1000 뉴런 → softmax |
> | ReLU | 모든 conv, FC layer에 적용 |
> | Normalization/Pooling | Conv1, Conv2 뒤에 LRN + pooling, Conv5 뒤에 pooling |

### 📚 4. Reducing Overfitting

> ## 4.1. 데이터 증강 (Data Augmentation)
>
>   - 이미지 **crop 및 좌우 반전** : 256x256 이미지에서 임의의 224x224 패치 및 그 좌우 반전본을 잘라내어 학습에 사용
>       - 약 2048배의 데이터 증가 효과. 테스트 시에는 네 구석 + 중앙 5개 패치와 각 좌우 반전본을 포함해 총 10개 패치의 softmax 출력을 평균하여 최종 예측
>   - **RGB 채널 강도 변환** : 훈련셋 전체 픽셀 값에 대해 PCA를 수행하여, 각 이미지 픽셀 값에 PCA의 주성분 벡터를 일정 비율로 더함 (난수 활용)
>   - 이 효과를 통해 조명 세기나 색이 달라져도 물체 정체성이 유지되는 자연 이미지의 성질을 모방
>
> <br>
>
> ![RGB channel intensity variation example](/assets/img/posts/alexnet/image%203.png)
>
> ## 4.2. 드롭아웃 (Dropout)
>
>   - 학습 시 각 hidden neuron의 출력을 확률 0.5로 0으로 만들어버림. 따라서 전파 및 역전파에 일정 뉴런들이 참여하지 않음.
>   - 매번 입력시 다른 아키텍쳐를 가지는 효과를 가짐 (but 가중치는 공유).
>   - 즉, 특정 뉴런이 다른 뉴런의 존재에 지나치게 의존(co-adaptation)하지 못하게 하여 **일반화 성능을 높인다.**
>   - 테스트 시에는 모든 뉴런을 사용하되, 출력에 0.5를 곱해줌.
>   - 첫 두 fully-connected layer에 적용. Dropout이 없을 때 네트워크가 심각하게 과적합되었으며, 적용 시 학습 수렴 속도가 약 두 배 느려지지만 과적합은 크게 줄어듦.
>
> ## 정리
>
> | 항목 | 내용 |
> | --- | --- |
> | 문제 | 파라미터 60M → 데이터(120만 장)로는 과적합 심각 |
> | 해결책1 | **Data Augmentation** |
> | – 무작위 crop & 좌우 반전 (훈련 데이터 2048배 확장) |  |
> | – PCA 기반 색상 증강 (조명·색 변화 불변성 반영) |  |
> | 효과1 | crop 없으면 과적합 심각 / 색상 증강 시 top-1 오류율 1%↓ |
> | 해결책2 | **Dropout** |
> | – 학습 시 뉴런을 확률 0.5로 제거 |  |
> | – 뉴런 간 공적응(co-adaptation) 방지 |  |
> | – 일반화 성능 ↑, 하지만 수렴 속도 약 2배 느려짐 |  |
> | 적용 위치 | Fully-connected 1, 2층 |

### 📚 5. Details of learning

> ## 5\. 학습 세부사항
>
>   - \*\*확률적 경사하강법(SGD)\*\*으로 학습. 배치 크기 128, 모멘텀 0.9, **가중치 감쇠(weight decay) 0.0005**로 설정.
>   - 작은 weight decay는 단순한 정규화 이상의 역할을 하였음 → 모델이 실제로 학습을 하도록 도와주었음.
>
> <br>
>
> ![Weight update rule](/assets/img/posts/alexnet/image%204.png)
>
>   - **가중치 초기화**: 각 계층의 가중치는 평균 0, 표준편차 0.01의 Gaussian 분포에서 샘플링.
>   - **Bias 초기화**: 두 번째, 네 번째, 다섯 번째 convolutional layer와 fully-connected hidden layer의 뉴런 bias는 상수 1로 초기화. (bias = 1 : ReLU 뉴런이 초기 학습 단계에서 양수 입력을 더 잘 받게 되어 학습이 빨라지는 효과). 나머지 layer의 bias는 0으로 초기화.
>   - **학습률 스케줄**: 모든 layer에 동일한 학습률을 사용했으며, 학습 중 수동으로 조정. 검증 오류(validation error)가 개선되지 않으면 학습률을 10배 낮춤. 초기 학습률은 0.01이었고, 종료 전까지 총 세 번 낮춤.
>   - 전체 학습은 약 120만 장의 이미지 데이터셋을 **90 epoch** 순환하는 동안 진행되었으며, 3GB 메모리를 가진 NVIDIA GTX 580 GPU 두 장에서 **5\~6일**이 소요됨.