---
title: "(AlexNet) ImageNet Classification with Deep Convolutional Neural Networks"
date: 2025-08-29 18:00:00 +0900
categories: [AI-ML-DL, etc.]
tags: [Computer-Vision, paper]
---
## 📚 정리

### 📌 제목

> **Visualizing and Underst  - **분석 구조**

![Deconvnet Process]((ZFNet)%20Visualizi### 📚 4. Convnet Visualization

![Feature Visualization across Layers]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%202.png)

**4. 시각화**g%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%201.png)

**✅ 5줄 요약**nding  - **분석 구조**

![Deconvnet Process]((ZFNet)%20Visualizi### 📚 4. Convnet Visualization

![Feature Visualization across Layers]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%202.png)

**4. 시각화**g%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%201.png)

**✅ 5줄 요약**Convolutional Networks**

-----

### 🌟 초록

> **0. 초록**
>
>   - 왜 ConvNet이 작동하는지 명확한 이해가 부족하다, 그리고 추가적으로 어떻게 개선될 수 있는지에 대한 이해가 부족하다.
>   - **진단적 도구** 제안 : 중간 특징 추출(intermediate feature layers)의 기능과 분류기의 동작을 이해 → 기존 모델을 능가하는 아키텍처를 찾아내는 데 도움을 준다. 즉 **눈으로 이해**할 수 있다(블랙박스 부분을 화이트박스로)
>   - **ablation study** 수행
>   - **ZFNet개발 - 전이능력** 확인
>
> **정리**
>
> | 항목 | 내용 |
> | --- | --- |
> | 데이터셋 | ImageNet 2012, Caltech-101, Caltech-256 |
> | 모델 구조 | AlexNet 기반, stride/filter 개선, softmax 출력 |
> | 연구 기여 | Deconvnet 시각화, 구조 최적화, ablation, 전이학습 |
> | 평가 결과 | ImageNet에서 AlexNet보다 낮은 오류율, Caltech에서 SOTA 달성 |

-----

### 💡 결론 & 고찰

> **6. Discussion**
>
>   - 시각화하는 새로운 방법을 제시함으로서, feature들이 무작위적이거나 해석 불가능한 패턴이 아님이 들어났음
>       - 계층이 깊어질수록 compositionality(조합성), invariance(불변성), class discrimination(클래스 구별) 등 직관적인 특성들이 보였다.
>       - 시각화 기법이 모델 디버깅에 사용될 수 있다. AlexNet 성능을 향상
>   - Occlusion(가림) 실험을 통해, clf.가 scene context(장면의 광범위한 맥락)을 사용하는것이 아니라, 이미지의 local structure(국소적 구조)에 매우 민감하게 반응한다는 것을 입증함.
>   - Ablation study를 통해 개별 계층이 아니라 네트워크가 가지는 minimum depth(최소한의 깊이)가 성능에 필수적이다.
>   - 전이 학습의 효용성을 증명했다(일반화 성능), 그러나 PASCAL과 같은 데이터에서 dataset bias때문에, 일반화가 약했으나 그 마저도 낮은 감소였다. → 객체탐지까지 자연스럽게 확장될 수 있다.
>
> **핵심**
>
>   - convnet은 해석가능하고, 계층이 깊어질수록 추상화, 불변성이 증가
>   - 시각화 : 단순 시각 설명도구가 아니라, 모델 디버깅(개선용)에 사용가능
>   - Occlusion : 모델은 맥락보다, 진짜 객체 구조(local structure)에 집중
>   - Ablation : 모델은 층의 뉴런수보다, 충분한 깊이가 성능에 핵심
>   - 전이학습 특정 데이터셋에서 효과적, 그러나 데이터의 bias를 처리할 수 있다면 다른 데이터셋에서도 효과적일거라 예상
>       - ImageNet → Caltech(작은 데이터셋) : 효과적
>       - 역은 효과적인지 의문
>       - 대규모데이터셋이더라도 모든도메인에 완전히 전이되지 않음
>         → 손실함수를 손본다면 더 향상되지 않을까?
>
> **5줄 요약**
>
>   - ConvNet 특징은 무작위가 아니라 점진적으로 추상화되는 의미 있는 표현
>   - Deconvnet 시각화는 모델 성능 개선에 유용한 진단 도구
>   - 모델은 국소적 구조를 잘 잡아내며 깊이가 필수적임을 확인
>   - ImageNet 학습 모델은 Caltech 계열에서 SOTA 달성, 작은 데이터셋 벤치마크의 한계 제기
>   - PASCAL에서는 일반화가 제한적 → dataset bias, loss function 개선 필요
>
> **정리**
>
> | 항목 | 내용 |
> | --- | --- |
> | 시각화 발견 | 특징은 추상화·불변성·클래스 구별을 점차 강화 |
> | Occlusion | 객체 위치에 민감 → 배경 맥락만 이용하지 않음 |
> | Ablation | 특정 층보다 깊이(depth) 자체가 핵심 |
> | 전이 성능 | Caltech-101/256에서 SOTA, PASCAL은 dataset bias로 다소 저하 |
> | 시사점 | 작은 벤치마크의 유효성 재검토, loss function 개선 시 객체 탐지로 확장 가능 |

-----

### 🗃️ 데이터

> **데이터셋(생략)**
>
> | 데이터셋 | 크기/구성 | 특징 | 활용 목적 |
> | --- | --- | --- | --- |
> | ImageNet 2012 | 130만 학습 / 5만 검증 / 10만 테스트, 1000 클래스 | 대규모, 객체 중심 | ConvNet 학습 및 성능 평가 |
> | Caltech-101 | 101 클래스, 클래스당 15\~30 학습, 최대 50 테스트 | 소규모, 단순 객체 | 전이학습 효과 검증 |
> | Caltech-256 | 256 클래스, 클래스당 15\~60 학습 | 클래스 수 많고 다양성 큼 | 전이학습 강건성 평가 |
> | PASCAL VOC 2012 | 20 클래스, 장면 내 다중 객체 포함 | 복잡한 장면, multi-object | ConvNet 일반화 한계 확인 |

-----

### 📌 서론

> **1. 서론**
>
>   - 1990년 초 처음 제안된 CNN은 AlexNet(2012)부터 획기적인 모델로 발전해 왔다. 그 이유는
>       - 대규모 학습 데이터셋의 이용 가능성
>       - GPU의 구현
>       - 정규화 기법(Dropout, etc.)
>   - 그러나 Blackbox(내부 메커니즘 불명확)이기 때문에, 더 나은 모델 개발이 단순한 시행착오에 의존할 수 밖에 없다. 그래서 어떤 계층에서 feature map을 시각화하는 기법을 제공하고, 이는 어떤 특징이 어떻게 진화하는지 관찰하고 모델을 진단할 수 있다.
>   - Deconvolutional Network(deconvnet, (Zeiler et al., 2011))을 활용하여, feature activation을 다시 픽셀 공간으로 투영한다. 또 이미지의 일부를 가려서 분류의 민감도를 분석하여 어떤 부분이 분류에 중요한지 확인한다.
>   - 이러한 기법들을 활용하여 AlexNet에서 좀 더 발전한 ZFNet을 만들었다.
>
> **✅ 5줄 요약**
>
>   - ConvNet은 최근 ImageNet 등에서 성능을 혁신적으로 개선
>   - 그러나 내부 메커니즘은 여전히 불명확
>   - 본 논문은 deconvnet 기반 시각화로 이를 분석
>   - 모델 구조 개선 및 진단 가능성을 제시
>   - ImageNet 학습 특징이 전이학습에서도 탁월함을 보임
>
> **📌 정리**
>
> | 항목 | 내용 |
> | --- | --- |
> | 배경 | ConvNet 성능 급상승 (CIFAR-10, ImageNet 등) |
> | 한계 | 내부 동작 원리에 대한 이해 부족 |
> | 기여 | 시각화 기법 제안 (deconvnet, occlusion) |
> | 연구 전략 | AlexNet 구조 → 개선 → 시각화 기반 진단 → 전이 성능 확인 |
> | 사전학습 구분 | 지도 사전학습(supervised pre-training) vs 비지도 사전학습(unsupervised pre-training) 대비 |

-----

-----

## 🔬 실험과정

### 📚 관련 연구

> **1.1. 관련 연구**
>
>   - 대부분은 첫번째 레이어만 직접 시각화한다. 더 깊은 층에서는 이러한 접근이 제한적이다.
>   - 각 뉴런 유닛의 활성화를 최대화하기 위해서, 이미지 공간에서 각 유닛의 optimal stimulus(최적 자극)을 찾았으나 이는 초기화에 민감하여 유닛의 invariances(불변성)에 대한 정보는 제공하지 못한다. → 이러한 단점을 해결하기 위해 Hessian을 수치적으로 계산하여 일부 통찰을 제공했으나 깊어질수록 통찰을 제공하지 않는다(이차 근사의 단점)
>   - 이를 해결하기 위해 비모수적 관점의 불변성 시각화를 제공, 이미지를 잘라내는게 아니라 top-down projection을 통해 특정 피쳐맵을 자극하는 패치 내부의 드러낸다
>
> > **헤세 행렬(Hessian Matrix)**
>
> >   - 어떤 함수 f(x)의 \*\*Hessian 행렬(Hessian matrix)\*\*은 \*\*이차 도함수(이계 미분)\*\*를 모아놓은 **정방행렬**입니다.
> >   - Hessian은 함수의 \*\*곡률(curvature)\*\*을 나타내며, 함수가 특정 지점에서 볼록(convex)한지, 오목(concave)한지, 또는 안장점(saddle point)인지를 판별하는 데 사용됩니다.
> >   - **최적화(Optimization)**: 2차 최적화 기법(Newton's method 등)은 Hessian을 이용해 더 빠르게 수렴합니다.
> >   - **민감도 분석(Sensitivity Analysis)**: 특정 입력 변화가 출력에 어떤 영향을 주는지, 즉 모델의 \*\*국소적 불변성(local invariance)\*\*을 분석할 때 Hessian을 사용합니다.
> >   - 뉴런의 출력이 입력 변화에 따라 얼마나 민감하게 달라지는지(곡률)를 분석
> >   - 곡률이 낮은 방향 → 뉴런이 그 방향의 입력 변화에는 **불변(invariant)**
> >   - 곡률이 높은 방향 → 민감하게 반응 → 중요한 패턴 방향
>
> **핵심**
>
>   - 시각화 연구는 초기에 시작층이나 초기층에만 국한되어 깊은 층은 해석 불가능
>       - **최적 자극 탐색 (Erhan et al., 2009)**: 이미지 공간에서 경사하강법 → 활성화 극대화
>         → 단점: 초기화 민감, 불변성 정보 없음
>       - **Hessian 기반 불변성 분석 (Le et al., 2010)**: Hessian 근사로 불변성 파악
>         → 단점: 고차원 층의 복잡한 불변성을 단순 이차식으로 설명 불가
>       - **패치 기반 시각화 (Donahue et al., 2013)**: 데이터셋에서 강한 활성화를 일으키는 패치 식별
>         → 단점: 단순 crop, feature map 내부 구조는 설명하지 못함
>
> **✅ 5줄 요약**
>
>   - 과거 시각화 연구는 주로 첫 번째 층에 집중
>   - Erhan et al. (2009): 경사하강법으로 최적 자극 탐색, 불변성 설명 부족
>   - Le et al. (2010): Hessian 근사로 불변성 분석, 고층에서는 부정확
>   - Donahue et al. (2013): 데이터셋 패치 기반 시각화, 구조적 해석 제한
>   - 본 논문: Deconvnet을 통해 **비모수적, 구조적 시각화** 제공 → 고층 feature 해석 가능
>
> **📌 정리**
>
> | 연구 | 방법 | 한계 |
> | --- | --- | --- |
> | Erhan et al. (2009) | 이미지 공간 경사하강 → 최적 자극 | 초기화 민감, 불변성 설명 불가 |
> | Le et al. (2010) | Hessian 근사 → 불변성 분석 | 고차원 층의 복잡성 반영 못함 |
> | Donahue et al. (2013) | 패치 식별 → 활성화 해석 | 단순 crop, 구조 설명 한계 |
> | 본 논문 | Deconvnet 기반 top-down projection | 고층 feature 구조적 해석 가능 |

### 📚 2. Approach

> **2. Approach**
>
>   - **지도학습**
>   - **Layer구조 : Conv → ReLU → (옵션) Max Pooling : Local에 대해 → (옵션) Local Contrast Normalization : feature map 전반을 정규화**
>   - 네트워크가 깊어지면 소수의 fc layer로 구성, 마지막은 **softmax clf.**
>   - **아키텍쳐**
>
> ![ZFNet Architecture]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image.png)
>
>   - 손실함수 : **Cross-entropy**
>   - Optimizer : **SGD(mini-batch)**
>
> **✅ 5줄 요약**
>
>   - ConvNet은 입력 이미지를 계층적 변환을 통해 클래스 확률로 매핑
>   - 계층은 합성곱, ReLU, 풀링, 정규화로 구성
>   - 상위 계층은 fully-connected + softmax 분류기
>   - 교차 엔트로피 손실과 backpropagation으로 학습
>   - 확률적 경사하강법(SGD)으로 파라미터 최적화
>
> **📌 정리**
>
> | 요소 | 설명 |
> | --- | --- |
> | 입력 | 컬러 2D 이미지 (xi) |
> | 출력 | 클래스 확률 벡터 (ŷi) |
> | 계층 구성 | 합성곱 + ReLU + (풀링) + (정규화) |
> | 상위 구조 | Fully-connected layers |
> | 최종 분류기 | Softmax |
> | 손실 함수 | Cross-entropy |
> | 학습 알고리즘 | Backpropagation + SGD |
>
> **2.1 Visualization with a Deconvnet**
>
>   - **Deconvolutional Network : Conv와 동일한 구성이지만 픽셀 → 특징 투영과 반대로 특징 → 픽셀 투영을한다.**
>       - 기존에는 비지도학습용으로 제안되었지만, 여기서는 **이미 학습된 모델을 probe**하는 용도로 사용된다.
>   - **분석과정**
>     1.  **convnet에 deconvnet을 연결한다.**
>     2.  **이미지를 convnet에 넣은 후 feature을 계산하고, 특정 활성을 선택하고 나머지는 모두 0으로 만든다.**
>     3.  **이 feature을 deconvnet에 넣어서 복원한다.**
>         1.  **unpool**
>         2.  **rectify**
>         3.  **filter 과정을 거쳐 바로 아래 계층의 activity를 복원**
>   - **핵심연산**
>       - **Unpooling** : max pooling은 비가역적인데, 이를 해결하기 위해 **pooling시 각 영역의 최대값의 위치를 switch변수로 기록**하고, 그 값을 바탕으로 복원 결과를 정확한 위치에 배치
>       - **Rectification** : ConvNet처럼 **ReLU**를 사용하여 복원(**복원된 feature가 양수**)
>       - **Filtering** : **ConvNet의 학습된 필터를 전치 버전**을 사용한다.(수직, 수평방향)
>   - 한개의 활성으로부터 얻는 결과는 이미지의 어떤 구조의 일부분과 유사하다. **모델은 판별적으로 학습되므로, 입력이미지에서 어떤부분이 중요했는지를 드러**낸다. 다만 **생성모델의 샘플이 아니라, 단순히 역투영**된다는 점이다. → **생성모델이 아니라 역투영**이다. → 즉 모델이 입력구조의 어떤부분을 가지고 판단했는지 직관적으로 보여준다.
>   - **분석 구조**
>
> \<img src="(ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image 1.png" alt="Deconvnet Process"\>
>
> **✅ 5줄 요약**
>
>   - Deconvnet을 이용해 중간층 feature map을 입력 픽셀 공간으로 복원
>   - 과정: (Unpool → ReLU → Filter) 반복
>   - pooling switch로 원래 자극 위치 복원
>   - 결과: 특정 activation이 입력 이미지의 어떤 부분에 의해 유발되었는지 확인 가능
>   - 이는 생성이 아닌 판별 기반 projection → ConvNet의 판별 근거를 직관적으로 시각화
>
> **📌 정리**
>
> | 단계 | ConvNet (정방향) | Deconvnet (역방향) |
> | --- | --- | --- |
> | Pooling | Max pooling | Unpooling (switch 사용) |
> | ReLU | ReLU | ReLU |
> | Filtering | Learned filter | Transposed filter (flip) |
> | 결과 | Feature map | 입력 공간 복원 |

### 📚 3. Training Detail

> **3. 학습 세부 사항**
>
>   - AlexNet의 GPU분산 학습을 했기때문에 3, 4, 5층은 \*\*sparse connections(희소 연결 - 우리가 기존에 아는 sparse가 아니라, GPU로 인한 모델 아키텍쳐 분산을 의미)\*\*을 사용했지만, 본 모델에서는 **dense connections**로 대체
>       - 또한 1층과 2층에 세부 수정이 이루어짐(시각화 결과를 기반으로)
>   - AlexNet과 동일하게 증강
>   - 1층 필터를 시각화 한결과, 일부 필터가 지나치게 지배적이기 때문에 이를 방지하기 위해 \*\*RMS값이 10^-1을 초과하는 필터는 강제로 renormalization(재정규화, 0.1로 만듦, 균형 유지)\*\*한다. 이는 [-128, 128]인 1층에서 중요하다.
>
> **✅ 5줄 요약**
>
>   - ImageNet 2012에서 학습 (130만 장, 1000 클래스)
>   - 전처리: 리사이즈·크롭·평균 제거·데이터 증강
>   - 학습: SGD, mini-batch=128, learning rate=0.01 시작, momentum=0.9, Dropout=0.5
>   - 필터 재정규화로 특정 필터 지배 방지
>   - 70 epoch, GPU 1장, 약 12일 소요
>
> -----
>
> **📌 정리**
>
> | 항목 | 설정 |
> | --- | --- |
> | 데이터 | ImageNet 2012 (1.3M, 1000 클래스) |
> | 전처리 | 리사이즈 256, 중앙 크롭, 평균 제거, 224x224 서브 크롭 10개, flip |
> | 최적화 | SGD, batch=128, lr=0.01, momentum=0.9 |
> | 정규화 | Dropout(0.5), filter RMS clipping(0.1) |
> | 구조 차이 | AlexNet sparse → dense 연결 |
> | 학습 시간 | 70 epoch, GTX580 GPU, 12일 |

### 📚 4. Convnet Visualization

> \<img src="(ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image 2.png" alt="Feature Visualization across Layers"\>
>
> **4. 시각화**
>
>   - 각 층별 특징 : 그 레이어에서의 복원이 아니라, 역으로 전부다 거친 후 복원, 따라서 깊어질수록 해상도가 높아진다. e.g. 2층의 경우 2 → 1, 5층의 경우 5 → 4 → … → 1 즉 하위계층(초기층)과 상위계층(후반층)을 비교
>       - **2층(layer 2)**: 모서리(corner), 색상/엣지 결합 구조에 반응
>       - **3층(layer 3)**: 텍스처(texture)와 같은 복잡한 불변성 패턴 포착
>       - **4층(layer 4)**: 클래스 특이적(class-specific) 패턴 (예: 개 얼굴, 새 다리)
>       - **5층(layer 5)**: 포즈 변화가 큰 전체 객체 (예: 키보드, 개 전체 모습)
>   - **입력 변형(input deformation)에 대한 불변성(invariance)을 확인 → 작은 변화는 하위층에서 큰 효과를 주지만 상위층에는 quasi-linear(안정적)인 반응을 보여줌**
>
> ![Feature Invariance Visualization]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/스크린샷_2025-08-28_오전_12.47.55.jpg)
>
>   - **계층적 성격(hierarchical nature)**
>   - **하위 계층은 소수의 epochs만에 수렴, 상위는 오래걸림**
>
> ![Feature Evolution during Training]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%203.png)
>
>   - **즉 ConvNet은 깊게 학습될수록 추상적인 특징을 학습**
>
> **✅ 5줄 요약**
>
>   - Deconvnet으로 각 층의 feature map을 픽셀 공간으로 복원
>   - 하위층: 저수준 특징(엣지, 코너), 중간층: 텍스처, 상위층: 객체/부위
>   - 훈련 초기에 하위층은 빠르게 수렴, 상위층은 늦게 발달
>   - 입력 변형에 대해 상위층은 더 안정적 → 불변성 확보
>   - ConvNet은 계층적으로 의미 있는 표현을 학습함을 실증
>
> -----
>
> **📌 정리**
>
> | 층 (Layer) | 주요 특징 | 시각화 결과 |
> | --- | --- | --- |
> | Layer 2 | 코너, 엣지+색 결합 | 기본 기하학적 구조 감지 |
> | Layer 3 | 텍스처, 반복 패턴 | 메시(mesh), 텍스트 인식 |
> | Layer 4 | 클래스 특이적 부위 | 개 얼굴, 새 다리 등 |
> | Layer 5 | 전체 객체 | 개, 키보드 등 다양한 포즈 |
>
> **4.1. Architecture Selection**
>
>   - **AlexNet의 문제점**
>       - **1층** : 매우 고주파(high frequency)와 저주파(low frequency) 정보가 혼합되어 있으며, **중간 주파수(mid frequency) 영역의 커버가 부족**
>       - **2층** : 1층 합성곱에서 stride=4를 사용한 탓에 \*\*aliasing artifact(샘플링 왜곡)\*\*이 발생
>   - **ZFNet 수정사항**
>       - **1층 필터 크기**를 11x11에서 7x7로 줄이고,
>       - **합성곱 stride**를 4에서 2로 축소하였다.
>
> ![AlexNet vs ZFNet Layer 1 and 2 Visualization]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%204.png)
>
> **✅ 5줄 요약**
>
>   - 시각화를 통해 AlexNet의 1·2층에서 문제점 발견
>   - 1층: 중간 주파수 부족, 2층: stride=4로 인한 aliasing
>   - 개선: 11x11 필터 → 7x7, stride=4 → 2
>   - 결과: 정보 보존 ↑, aliasing ↓
>   - 성능 또한 개선됨 (Section 5.1)
>
> -----
>
> **📌 정리**
>
> | 문제 (AlexNet) | 해결책 (Zeiler & Fergus) | 결과 |
> | --- | --- | --- |
> | 1층: 고·저주파 위주, mid-frequency 부족 | 필터 크기 11x11 → 7x7 | 더 균형 잡힌 필터 |
> | 2층: stride=4 → aliasing 발생 | stride=4 → 2 | aliasing 제거, 정보 보존 ↑ |
> | 성능 | 개선 전 AlexNet | 개선 후 더 낮은 오류율 |
>
> **4.2 Occlusion Sensitivity**
>
>   - 객체의 위치를 인식하는지, 주변 맥락만 사용하는지를 확인해보는 실험으로, 결론적으로 **객체를 지우는 경우 올바른 클래스 확률이 크게 떨어진다**.
>   - 즉 모델은 **객체 자체에 집중하여,** 객체 탐지의 가능성에 대해 시사
>
> ![Occlusion Sensitivity Experiment]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%205.png)
>
> **✅ 5줄 요약**
>
>   - Occlusion으로 모델이 배경이 아닌 객체 위치에 의존함을 확인
>   - 객체 부위가 가려지면 올바른 클래스 확률 급락
>   - top conv layer의 feature map 활성도도 함께 급락
>   - Deconvnet 시각화가 진짜 자극 구조를 보여준다는 검증
>   - ConvNet은 장면 context보다 객체 local structure에 민감
>
> -----
>
> **📌 정리**
>
> | 질문 | 방법 | 결과 | 의미 |
> | --- | --- | --- | --- |
> | 객체 위치를 보는가, 배경 context를 보는가? | 이미지 영역을 순차적으로 가림 | 객체 부분이 가려지면 확률 급락 | ConvNet은 객체 local structure에 집중 |
> | 시각화 신뢰성 검증 | top conv layer feature map 활성도 관찰 | occluder가 해당 구조를 가리면 활성도 급락 | Deconvnet 시각화 결과가 실제 feature와 일치 |
>
> **4.3. Correspondence Analysis**
>
>   - 기존의 모델들은 \*\*특정 객체 부위간의 대응(얼굴 속, 코와 눈)\*\*을 명시적으로 설정하게 되는데, 딥러닝 모델의 경우에는 이러한 메커니즘이 **암묵적으로 대응**한다는 점이다.
>
> **✅ 방법**
>
> 1.  개 얼굴 이미지 5장 선택
> 2.  동일한 위치(예: 왼쪽 눈)를 가려서 원본과 feature 차이 εli 계산
> 3.  이미지 쌍(i, j) 간의 차이 벡터 일관성을 해밍 거리로 측정 (Δl)
> 4.  특정 부위 vs 무작위 부위 비교
>
> **✅ 결과**
>
>   - **Layer 5**: 눈·코 같은 의미 있는 부위에서 Δ 값이 낮음 → 대응성 확보
>   - **Layer 7**: breed 판별에 집중하므로 Δ 값이 무작위 부위와 유사 → 부위 대응 정보 약화
>
> **✅ 의미**
>
>   - ConvNet은 명시적으로 correspondence를 정의하지 않아도, 중간층에서 **객체 부위 간 암묵적 대응**을 학습
>   - 그러나 깊은 층으로 갈수록 이 정보는 사라지고, **클래스 구분에 더 특화**됨
>
> > **해밍 거리**
> > 두 벡터가 있을때, 서로 다른 위치의 원소 개수를 새는 거리 측도
> > 값이 작을수록 서로 다른 이미지에서도 같은 부위가 공통된 역할을 하고 있음을 의미
>
> ![Correspondence Analysis Experiment]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%206.png)
>
> **✅ 5줄 요약**
>
>   - ConvNet이 암묵적으로 객체 부위 대응을 학습하는지 실험
>   - 동일 부위를 가려 feature 변화량을 비교
>   - Layer 5: 눈·코에서 변화가 일관적 → 대응성 존재
>   - Layer 7: breed 판별로 치중 → 대응성 감소
>   - ConvNet은 중간층에서 correspondence를 형성하지만, 깊은 층에서는 판별적 특징에 집중
>
> -----
>
> **📌 정리**
>
> | 층 (Layer) | 부위 | Δ 값 결과 | 의미 |
> | --- | --- | --- | --- |
> | Layer 5 | 눈·코 vs 무작위 | 눈·코 Δ 더 낮음 | 부위 간 대응성 확보 |
> | Layer 7 | 눈·코 vs 무작위 | 유사 | breed 구분에 집중, correspondence 약화 |

### 📚 5. Experiments

> **5.1. ImageNet 2012**
> **✅ 5줄 요약**
>
>   - ImageNet 2012: 학습 130만 / 검증 5만 / 테스트 10만, 1000 클래스
>   - AlexNet 구조 재현 → 보고된 성능과 동일
>   - stride 4→2, filter 11×11→7×7 → 성능 향상
>   - 단일 모델: Top-5 error 1.7% 개선
>   - 앙상블: 14.8% error → 당시 최고 성능, 비-ConvNet의 절반 수준
>
> -----
>
> **📌 정리표 (5.1 ImageNet 2012)**
>
> | 모델 | Top-5 Error (%) | 비고 |
> | --- | --- | --- |
> | AlexNet (2012) | 16.4 | Krizhevsky et al. |
> | Zeiler & Fergus (단일) | 약 14.7–15.0 | stride=2, filter=7×7 적용 |
> | Zeiler & Fergus (앙상블) | 14.8 | 2012 학습셋 기준 최고 성능 |
> | 비-ConvNet (Gunji et al.) | 26.2 | 같은 대회 상위 entry |
>
> ![ImageNet 2012 Results Table]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%207.png)
>
> **5.2 Feature Generalization**
> **✅ 5줄 요약**
>
>   - ImageNet 사전학습 feature는 소규모 데이터셋에서도 강력
>   - Caltech-101: 86.5%, 기존 최고치보다 +2.2%
>   - Caltech-256: 74.2%, 기존 최고치보다 +19%
>   - PASCAL: 평균 79.0%, 최고치 82.2%에 근접, 일부 클래스는 우위
>   - ConvNet은 범용적 전이 학습 도구임을 입증
>
> -----
>
> **📌 정리표 (5.2 Feature Generalization)**
>
> | 데이터셋 | 사전학습 모델 성능 | 기존 최고치 | Scratch 학습 | 의미 |
> | --- | --- | --- | --- | --- |
> | Caltech-101 | 86.5% | 81.4% | 46.5% | 소규모에서도 SOTA |
> | Caltech-256 | 74.2% (60 imgs/class) | 55.2% | 38.8% | 대규모/소규모 모두 압도 |
> | PASCAL VOC 2012 | 79.0% (mean) | 82.2% | - | 다중 객체 장면, 일부 클래스는 더 우위 |
>
> **5.3 Feature Analysis**
>
>   - **깊어질수록 판별 성능 증가**
>
> **✅ 5줄 요약**
>
>   - ConvNet의 feature 판별력은 깊을수록 향상
>   - Layer 1: 저성능 (엣지·색 기반)
>   - Layer 5: 큰 향상 (중간 특징이 매우 강력)
>   - Layer 7: Caltech-256에서는 최고, Caltech-101에서는 plateau
>   - ConvNet은 계층적으로 점점 강력한 특징 표현을 학습
>
> -----
>
> **📌 정리**
>
> | 데이터셋 | Layer 1 | Layer 3 | Layer 5 | Layer 7 | 의미 |
> | --- | --- | --- | --- | --- | --- |
> | Caltech-101 | 44.8% | 72.3% | 86.2% | 85.5% | 중간\~상위층에서 큰 향상, 최상위층은 plateau |
> | Caltech-256 | 24.6% | 46.0% | 65.6% | 71.7% | 층이 깊을수록 계속 향상, 최상위층이 최강 |