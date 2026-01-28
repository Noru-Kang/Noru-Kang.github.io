---
title: Chronos-2 From Univariate to Universal Forecasting
date: 2026-01-28 14:00:00 +0900
categories:
  - AI-ML-DL
  - etc.
tags:
  - time-series
  - TS_model
math : true
---
### 🔗 출처
> https://arxiv.org/abs/2510.15821

---
## 🗓️ 요약
### 📌 3줄 요약

1. 기존 Pretrained모델의 Univariate forecasting한계를 넘어, multivariate 및 covariate-informed forecasting task를 zero-shot방식으로 처리하는 범용 모델
2.  group attention mechanism을 통해 in-context learning(ICL)을 구현해 관련 TS간 효율적인 정보 공유를 가능하게 하며, 주로 synthetic datasets를 통해 학습
3. 다양한 벤치마크에서 SOTA를 달성했으며, covariate-informed task에서 기존 chronos 보다 큰 폭의 개선을 보여줌

### 📍요약

 기존 Chronos가 가졌던 <b>단변량(Univariate) 예측</b>의 한계를 넘어, <strong>다변량(Multivariate), 공변량(Covariate)</strong>정보를 zero-shot으로 처리할 수 있도록 진화한 foundation모델
 
#### 1. 전처리 및 토큰화 : 수치적 안정성과 구조화
  단순히 숫자를 나열하는 것이 아니라, 시계열의 구조적 정보를 보존하면서도 모델이 처리하기 쉬운 형태로 변환하는데 집중
##### 1.1. Robust Scailing($sinh^{-1}$)
 $$\tilde{v}_{t,d} = \text{sinh}^{-1}\left(\frac{v_{t,d} - \mu_d}{\sigma_d}\right)$$  기존의 표준화(standardization)는 이상치에 취약함. 따라서 이를 해결하기 위해 <strong>아크사인 변환</strong>을 도입하였음. 이 함수는 원점 근처에서 선형적으로 작동하고, 값이 커질수록 로그함수처럼 작동하여 <strong>분산을 안정화</strong>하고 이상치의 영향을 억제한다
 
##### 1.2. Pathcing
 시계열 데이터를 한 점씩 처리하는 대신, 패치로 묶어 처리함. 계산 복잡도롤 낮추고, <strong>Local Shape</strong>를 더 잘 파악하도록 해줌
#### 2. 아키텍처
 핵심적인 부분은 <strong>Dual Attention</strong>구조이다. 이를 통해 시계열 내의 시간적 흐름뿐만 아니라, 서로 다른 변수 간의 관계까지 학습

##### 2.1. Time Attention (시간축 정보 집계)
 기존 트랜스포머와 마찬가지로, 동일한 시계열 내에서 과거 패치들이 미래 예측에 얼마나 중요한지 계산한다. 이때 <strong>RoPE(Rotary Position Embeddings)</strong>을 활용하여 위치 정보를 회전 행렬로 인코딩하여 상대적 거리를 보존함으로서 시간순서를 파악한다.
##### 2.2. Group Attention (변수 간 정보 집계)
 다변량과 공변량을 처리하는 핵심
 - <strong>In-Context Learning(ICL)</strong> : 여러 시계열을 하나의 <strong>그룹</strong>으로 묶어 배치에 넣으면, 모델은 그룹 내 다른 시계열로부터 힌트를 얻음(공변량 혹은 다변량으로부터)
 - e.g. "기온"을 예측할때 "습도"와 "일사량"을 같은 그룹 ID로 묶어주면, 모델은 이들 사이의 Dynamic을 참조하여 더 정확한 예측을 수행
 - 인반적인 트랜스포머의 Time Attention이 하나의 시계열 안에서 "어제와 오늘"의 관계를 본다면, <strong>Group Attntion은 같은 시간대에 존재하는 "변수 A와 변수 B"의 관계를 봄</strong>. 즉, 그룹으로 묶인 서로 다른 변수들끼리 어텐션을 수행
	 - 입력 시계열들에 Group ID를 부여 : e.g. 목표 주가, 거래량, 금리 등을 같은 ID로 부여
	 - 위치 임베딩을 사용하지 않음(변수사이에는 순서가 없음)

#### 3. 확률론적 예측 : Quantile Head
 점예측을 하지 않고, <strong>21개의 분위수</strong>를 예측하여 값의 분포를 제시

##### 3.1. Quantile Regression Loss
$$\sum_{q \in Q} \text{check\_loss}(z - \hat{z}_q)$$
  check_loss는 예측이 실제보다 높거나 낮을 때 비대칭적인 페널티를 주는 함수
 이를 통해 예측의 <strong>불확실성</strong>을 정량화할 수 있으며, 실제 비즈니스 의사결정에서 중요한 역할을 함

#### 4. 전략 : 합성 데이터
 실제로 제공된 다변량 데이터의 양이 적기때문에, <strong>Multivariatizers</strong>라는 통계적 기법을 사용
 - <strong>상관관계 주입</strong> : 독립적인 단병량 데이터들을 생성한 후, 선형/비선형 변환을 통해 <strong>강제로 상관관계를 만듦</strong>
 - <strong>시차 효과(Lead-Lag)</strong> : 한 시계열이 다른 시계열보다 조금 늦게 움직이는 패턴 등을 수학적으로 생성하여, 모델이 복잡한 인과관계를 배울 수 있도록 함

### Chronos vs Chronos-2
| 비교 항목     | Chronos (v1)                           | Chronos-2                            |
| ------------- | -------------------------------------- | ------------------------------------ |
| **모델 구조** | T5 (Encoder–Decoder) / GPT-2           | T5 **Encoder-Only** (인코더만 사용)  |
| **기본 단위** | 토큰 (Token, 개별 수치 하나)           | 패치 (Patch, 여러 수치의 묶음)       |
| **전처리**    | Mean Scaling & Binning (구간 나누기)   | $\sinh^{-1}$ Scaling (아크사인 변환) |
| **출력 형태** | 범주형 분포 (Categorical Distribution) | 분위수 예측 (Quantile Regression)    |
| **지원 범위** | 단변량(Univariate) 전용                | 단변량 · 다변량 · 공변량 (Zero-shot) |
| **위치 정보** | Absolute / Relative Bias               | RoPE (Rotary Position Embedding)     |
| **핵심 기작** | 단순 자기회귀 (Autoregressive)         | Group Attention (ICL의 핵심)         |


---
## 📚 정리

### 📌 제목

#### Chronos-2: From Univariate to Universal Forecasting

---
---
### 🌟 초록
### 번역

사전 훈련된 시계열 모델은 작업별 학습 없이 정확한 예측을 생성하는 추론 전용 예측 시스템을 가능하게 했습니다.
그러나 기존 접근 방식은 주로 단변량 예측에 초점을 맞춰, 다변량 데이터와 공변량이 중요한 역할을 하는 실제 시나리오에서의 적용 가능성을 제한합니다.
본 논문에서는 제로샷(zero-shot) 방식으로 단변량, 다변량 및 공변량 기반 예측 작업을 처리할 수 있는 사전 훈련 모델인 Chronos-2를 제시합니다.
Chronos-2는 그룹 내에서 관련 시계열 집합, 다변량 시계열의 변량, 또는 예측 대상과 공변량을 나타낼 수 있는 그룹 내에서 다중 시계열 간의 효율적인 정보 공유를 통해 인컨텍스트 학습(ICL, in-context learning)을 촉진하는 그룹 어텐션 메커니즘을 사용합니다.
이러한 일반적인 기능은 단변량 시계열에 다양한 다변량 구조를 부과하는 합성 데이터셋으로 학습하여 달성됩니다.
Chronos-2는 세 가지 포괄적인 벤치마크인 fev-bench, GIFT-Eval, 그리고 Chronos Benchmark II에서 최첨단 성능을 제공합니다.
다변량 및 공변량 기반 예측을 강조하는 fev-bench에서 Chronos-2의 보편적인 ICL 기능은 기존 모델에 비해 상당한 개선을 가져옵니다.
공변량이 포함된 작업에서는 기존 베이스라인 모델보다 일관되게 훨씬 뛰어난 성능을 보입니다.
에너지 및 소매 분야의 사례 연구는 그 실질적인 이점을 더욱 부각합니다.
Chronos-2의 인컨텍스트 학습 기능은 이를 실제 예측 파이프라인에서 "그대로" 사용할 수 있는 범용 예측 모델로 자리매김하게 합니다.

---
### 내용

단변량 → 다변량, 공변량으로 확장
다중 시계열 간의 효율적인 정보 공유를 통한 ICL(in-context learning)을 촉진하는 <strong>Group Attention</strong>을 사용
<strong>단변량 시계열에 다양한 다변량 구조를 부과하는 합성 데이터셋</strong>을 활용

---
### 포인트

- <strong>Group Attention</strong> : 다중 시계열 간의 효율적인 정보 공유를 통한 ICL 촉진
- <strong>단변량 → 다변량 합성</strong> : 단변량에 다변량 구조를 부과

---
---
### 📌 서론 & 결론 & 고찰

### 번역
#### 서론

사전 훈련된 모델(기초 모델이라고도 함)의 등장은 시계열 예측 분야에 패러다임의 전환을 가져왔습니다.
각 시계열(로컬 모델, local models) (Hyndman & Athanasopoulos, 2018) 또는 각 데이터셋(작업별 모델, task-specific models) (Lim et al., 2021; Challu et al., 2023)에 대해 개별적으로 모델을 학습시키는 대신, 단일 모델을 대규모 시계열 데이터로 한 번 학습한 뒤 다양한 예측 문제에 적용할 수 있게 되었습니다 (Ansari et al., 2024; Das et al., 2024b).
이러한 사전 훈련 접근법은 각 사용 사례마다 모델을 처음부터 학습할 필요를 제거함으로써 예측 파이프라인을 크게 단순화합니다.
더욱 주목할 만한 점은, 사전 훈련된 모델이 종종 작업별 모델과 유사하거나 이를 능가하는 예측 정확도를 달성한다는 사실입니다 (Aksu et al., 2024).

이러한 발전에도 불구하고 근본적인 한계는 여전히 존재합니다.
대부분의 사전 훈련된 모델은 단변량 시계열에만 적용 가능하며, 예측 시 단일 시계열의 과거 관측치만을 활용합니다.
단변량 예측은 중요한 문제이지만, 실제 운영 환경에서 요구되는 예측 과제는 이보다 훨씬 복잡합니다.
현실적으로는 여러 개의 시계열이 함께 진화하는 상황에서 이를 동시에 예측해야 하는 다변량 예측(multivariate forecasting) 문제 (Bańbura et al., 2010; Cohen et al., 2025)나, 예측 대상이 외부 요인에 의존하는 공변량 기반 예측(covariate-informed forecasting) 문제에 직면하는 경우가 많습니다.

예를 들어, CPU 사용량, 메모리 소비량, 스토리지 I/O와 같은 클라우드 인프라 지표들은 상호 의존적으로 변화하며 공동 모델링을 통해 더 나은 예측 성능을 얻을 수 있습니다 (Cohen et al., 2025).
또한 소매 수요는 프로모션 활동에 크게 영향을 받으며, 에너지 소비 패턴은 기상 조건에 의해 좌우됩니다 (Petropoulos et al., 2022).
이처럼 다변량 및 공변량 기반 예측을 지원하지 못하는 점은 사전 훈련된 모델이 실제 운영 시스템에 광범위하게 채택되는 데 있어 주요한 장애 요인으로 작용합니다.

다변량 종속성과 공변량 정보를 모두 처리할 수 있는 범용 사전 훈련 모델의 개발은 두 가지 이유로 여전히 어려운 과제로 남아 있습니다.
첫째, 예측 문제의 이질성으로 인해 모델 아키텍처에 대한 근본적인 재고가 필요합니다.
각 다운스트림 작업은 변수의 개수와 그 의미론적 해석이 상이하며, 사전에 알려지지 않은 작업에서는 변수 간 상호작용 구조를 미리 가정할 수 없습니다.
따라서 모델은 주어진 맥락만을 기반으로 이러한 상호작용을 스스로 추론할 수 있어야 합니다.
둘째, 다변량 종속성과 유의미한 공변량 정보를 모두 포함하는 고품질 사전 훈련 데이터 자체가 매우 제한적입니다.

본 연구에서는 단변량, 다변량, 공변량 기반 예측을 모두 제로샷(zero-shot) 방식으로 처리할 수 있도록 설계된 사전 훈련 모델 Chronos-2를 제안합니다.
Chronos-2는 인컨텍스트 학습(in-context learning, ICL)을 활용하여 다변량 예측을 지원하며, 과거 데이터만 주어지는 경우뿐만 아니라 미래 값이 알려진 공변량, 실수값 또는 범주형 공변량을 모두 처리할 수 있습니다.
또한 향상된 ICL 능력은 배치 내의 단변량 시계열들 간 정보 공유를 가능하게 하는 크로스 학습(cross learning)을 통해 단변량 예측 성능 역시 향상시킵니다.

Chronos-2의 ICL 기능의 핵심은 그룹 어텐션 메커니즘에 있습니다.
이 메커니즘은 관련 시계열들의 임의의 집합, 다변량 시계열의 각 변량, 혹은 예측 대상과 공변량(과거 혹은 미래 값이 알려진 경우)을 하나의 그룹으로 묶어 그룹 내부에서 정보 교환을 가능하게 합니다.
타겟과 공변량을 단순히 연결(concatenation)하여 입력 맥락을 확장하는 대신, 그룹 어텐션 레이어는 배치 차원을 가로질러 그룹 단위로 정보를 공유함으로써 변량 수가 증가하더라도 우아하게 확장됩니다.
Chronos-2의 또 다른 핵심 기여는 학습 전략에 있으며, ICL 기능을 활성화하기 위해 기본 단변량 생성기로부터 샘플링한 시계열에 인위적으로 다변량 구조를 부여한 합성 시계열 데이터를 활용합니다.
토큰화 과정과 모델링을 포함한 Chronos-2의 전체 추론 파이프라인은 그림 1에 제시되어 있습니다.

fev-bench (Shchur et al., 2025), GIFT-Eval (Aksu et al., 2024), Chronos Benchmark II (Ansari et al., 2024)를 포함한 포괄적인 예측 벤치마크 실험 결과, Chronos-2는 최첨단 성능을 달성함을 확인하였습니다.
단변량, 다변량, 공변량 기반 예측을 모두 포함하는 fev-bench에서 Chronos-2는 모든 범주에서 기준 모델들을 일관되게 능가하였으며, 특히 공변량 정보를 포함하는 과제에서 가장 큰 성능 향상을 보였습니다.
이는 실제 활용도가 높은 설정에서 Chronos-2의 강점을 명확히 보여줍니다.
이러한 성능 향상에도 불구하고 Chronos-2는 높은 계산 효율성을 유지하며, 단일 중급 GPU(NVIDIA A10G) 환경에서 초당 약 300개의 시계열을 처리할 수 있습니다.

본 기술 보고서는 다음과 같이 구성되어 있습니다.
2장에서는 시계열 예측의 배경과 기존 방법론을, 특히 사전 훈련된 모델을 중심으로 설명합니다.
3장에서는 Chronos-2의 아키텍처와 학습 및 추론 파이프라인을 상세히 다룹니다.
4장에서는 학습에 사용된 코퍼스를 소개하며,
5장에서는 세 가지 예측 벤치마크 결과와 함께 에너지 및 소매 도메인 사례 연구, 그리고 ablation 실험 결과를 제시합니다.
마지막으로 6장에서는 결론과 향후 연구 방향을 논의합니다.

---
#### 논의

본 연구에서는 단변량, 다변량, 공변량 정보가 포함된 작업을 포함하여 광범위한 예측 시나리오를 제로샷 방식으로 처리할 수 있도록 설계된 사전 학습 시계열 모델인 Chronos-2를 제시하였습니다.
세 가지 포괄적인 예측 벤치마크 전반에 걸쳐 Chronos-2는 기존의 기초 모델들에 비해 일관되게 우수한 성능을 보였으며, 인컨텍스트 학습이 다양한 예측 작업 유형에 걸쳐 예측 성능을 효과적으로 향상시킬 수 있음을 보여줍니다.

특히 공변량 정보가 포함된 예측 작업에서 현저한 성능 격차가 관찰되었는데, 이는 Chronos-2가 기존 기초 모델들의 성능을 크게 상회하기 때문입니다.
이러한 결과는 기존 모델들이 지닌 한계를 드러내는 동시에, 정확한 예측을 위해 공변량과 같은 문맥 정보가 수행하는 핵심적인 역할을 강조합니다.
비록 Chronos-2는 현재 숫자형 및 범주형 공변량만을 지원하지만, 사전 학습된 시계열 모델을 확장하여 텍스트와 같은 멀티모달 입력을 통합하는 방향은 향후 연구를 위한 유망한 과제로 남아 있습니다 (Zhang et al., 2025).

또한 본 연구 결과는 일반화 가능한 예측 성능을 달성하는 데 있어 합성 데이터의 중요성을 다시 한 번 부각합니다.
Chronos-2가 단변량 예측을 넘어서는 능력을 획득하는 데에는 합성 데이터가 핵심적인 역할을 하며, 제거 실험 결과 합성 데이터만으로 학습된 모델조차 실제 데이터와 합성 데이터를 혼합하여 학습한 모델에 비해 성능 저하가 제한적임을 확인하였습니다.
이는 합성 데이터가 향후 사전 학습 시계열 모델의 발전 과정에서 점점 더 중심적인 역할을 수행할 가능성을 시사합니다.

마지막으로, Chronos-2의 유연한 그룹 어텐션 메커니즘은 추가적인 응용 가능성을 제공합니다.
예를 들어, 희소한 메타데이터나 밀집 임베딩을 활용하여 시계열을 그룹화함으로써 검색 증강 예측(retrieval-augmented forecasting)을 구현할 수 있으며, 이는 데이터가 제한적인 환경이나 콜드 스타트 시나리오에서 예측 성능을 향상시키는 데 기여할 수 있습니다.

---
---
### 내용
#### 서론

대규모 시계열 데이터 모델  : Chronos(Ansari et al., 2024; Das et al., 2024b)을 통해 예측 파이프라인을 크게 단순화 시켰다.
추가적으로, pretrained model이 특화 모델과 비슷하거나, 더 나은 경우를 보이는 경우도 있다.(GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation, Ansari et al., 2024; Das et al., 2024b)

그러나, 다부분의 모델은 <strong>단변량 시계열에만 적용 가능하고, 예측 시 단일 시계열의 과거 관측치만 활용</strong>한다. 이는 실제 운영 환경에서 효과적이지 않다.
실제로 우리가 직면하는 문제는 아래와 같다.
- <strong>다변량 문제</strong> : 여러 시계열을 동시에 예측
- <strong>공변량 문제</strong> : 예측 대상이 외부 요인에 의존
e.g. 클라우드 인프라 지표 : CPU 사용량 + 메모리 소비량 + 스토리지 I/O etc.
즉, 이를 단변량 모델을 채택하는것은 주요한 장애 요인이다.

다변량 모델 설계의 어려움은 아래와 같다.
1. <strong>예측 문제의 이질성</strong> : 모델 아키텍처에 대한 근본적인 재고가 필요 → 개별 다운스트림 task는 변수의 개수와 그 의미론적 해석이 상이하거나, 사전에 알려지지 않은 작업에서는 변수 간 상호작용 구조를 미리 가정할 수 없다.
2. <strong>데이터가 없다</strong> : 고품질 사전 훈련 데이터가 없다.

<strong>Chronos-2는 단변량, 다변량, 공변량 기반 예측을 모두 zero-shot</strong>으로 처리할 수 있도록 설계하였다.
- <strong>ICL</strong>
	- 다변량 예측을 지원(과거 데이터만 주어진 경우)
	- 미래 값이 알려진 공변량, 실수 혹은 범주형 공변량을 모두 처리
	- cross learning : 배치 내의 단변량 시계열들 간 정보를 공유하여 단변량 예측 성능 역시 크게 향상시킨다.
	- <strong>Group Attention</strong> : 관련 시계열들의 임의의 집합, 다변량 시계열의 각 변량, 예측 대상과 공변량(과거 or 미래의 값이 알려진 경우)를 <strong>하나의 그룹으로 묶어 그룹 내부에서 정보 교환을 가능</strong>하게 만들었다.
		- 타겟과 공변량을 concat하는 대신, 배치 차원을 가로질러 그룹 단위로 정보를 공유하여 변량 수가 증가하더라도 확장가능해짐
- <strong>학습 전략</strong>
	- 단변량 생성기로부터 샘플링한 시계열 → 인위적인 다변량 구조를 부여

공변량 정보를 포함하는 과제에서 가장 큰 성능 향상을 보임
높은 계산 효율성 : 단일 중급 GPU(NVIDIA A10G) 환경에서 초당 약 300개의 시계열을 처리

![](/assets/img/posts/chronos2/81bed9e47691a91b1ea2e79ec9e8c7cc.png)


---
#### 논의

<strong>단변량, 다변량, 공변량 정보</strong>가 포함된 작업을 포함한 광범위한 예측 시나리오를 <strong>zero-shot</strong>으로 처리할 수 있는 chronos-2를 개발하였음
- <strong>ICL</strong>이 다양한 예측 작업 유형에 예측 성능을 효과적으로 향상시킴
- 특히, <strong>공변량 문제에서 크게 효과적이다.</strong> 이는 공변량과 같은 문맥 정보가 핵심적인 기능을 한다고 볼 수 있다.
비록 chronos-2는 숫자 및 범주형 공변량만 지원하지만, 추후 멀티모달로도 업데이트 할 수 있을것이다.

일반화 성능을 향상시키는데, <strong>합성 데이터</strong>가 몹시 중요하다.
이는 chronos-2가 단변량 예측을 넘어서는 예측을 하게 하는 핵심적인 역할이었고, 이는 추후 합성 데이터가 향후 시계열 모델에서 핵심적인 역할을 수행할것을 시사한다.

<strong>Group Attention</strong>을 통해, 희소한 메타데이터나, 밀집한 임베딩을 활용하여 시계열을 그룹화 시키고 RAG를 구현할 수 있으며, 이는 데이터가 제한적인 상황에서 에측 성능을 향상시키는데 기여할 수 있다.

---
---
### 포인트
#### 서론

<!-- > [!NOTE] -->
> **NOTE**
> <strong>In-Context Learning</strong>
> 모델의 가중치(Weight)를 새로 학습시키지 않고도, 입력값(Context)으로 주어진 예시나 데이터를 보고 즉석에서 문제 해결 방식을 깨우치는 능력으로
> chronos-2에서는, 새로운 도메인의 데이터가 들어왔을때 이 데이터의 과거 패턴을 context로 삼아 예측
> 즉, <strong>별도의 튜닝 없이 "눈치껏 맞추는"능력</strong>

<strong>다변량 vs 공변량</strong>

| 구분           | 다변량 (Multivariate) | 공변량 (Covariate) |
| -------------- | --------------------- | ------------------ |
| 예측 타겟 수   | 여러 개               | 하나               |
| 변수의 지위    | 모두 주인공           | 타겟 1개만 주인공  |
| 다른 변수 역할 | 서로 결과             | 보조 정보          |
| 예측 대상 여부 | 전부 예측             | 타겟만 예측        |
| 실무 핵심 질문 | “이것도 맞춰야 하나?” | “이건 힌트인가?”   |


---
#### 논의

(없음)

---
---
## 🔬 실험과정

### 📚 2 Background and Related Work
### 번역

시계열 예측은 과거 관측값을 기반으로 시간 순서 데이터의 미래 값을 예측하는 것을 목표로 합니다.  
형식적으로,  
$Y_{1:T} = [y_1, \dots, y_T]$는 길이가 $T$인 과거 시계열을 나타내며, 각 관측값 $y_t \in \mathbb{R}^D$는 단변량 시계열의 경우 $D=1$, 다변량 시계열의 경우 $D>1$입니다.  
이러한 과거 맥락이 주어졌을 때, 예측 호라이즌 $H$에 대해 다음 $H$개의 시간 단계 $Y_{T+1:T+H}$를 예측하는 것이 목표입니다.

예측 과정은 공변량(외생 변수라고도 함)에 의해 지원될 수 있습니다.  
공변량 시계열은  
$X_{1:T+H} = [x_1, \dots, x_{T+H}]$로 표현되며, 각 $x_t \in \mathbb{R}^M$은 추가적인 정보를 나타냅니다.  
이때 공변량은 과거 구간($t \le T$)뿐만 아니라 미래 구간($t > T$)에 대해서도 제공될 수 있습니다.

이 예측 문제는 각 시간 단계에서 단일 값을 추정하는 포인트 예측(point forecasting)으로 정의될 수 있으며, 또는 예측 불확실성을 포착하기 위해 다음의 조건부 분포를 추정하는 확률론적 예측(probabilistic forecasting)으로도 정식화될 수 있습니다.
$$
P(Y_{T+1:T+H} \mid Y_{1:T}, X_{1:T+H})
$$

제로샷 예측(zero-shot forecasting)은 추가적인 학습, 적응 또는 미세 조정 없이, 이전에 관측되지 않은 시계열 데이터셋에 대해 모델이 직접 예측을 생성하는 설정을 의미합니다.


사전 학습 모델 패러다임 이전의 시계열 예측 방법론은 크게 로컬 모델과 글로벌 모델로 구분됩니다.  
로컬 모델은 데이터셋 내 각 시계열마다 개별적인 매개변수 집합을 학습하며, ARIMA, 지수 평활법(Exponential Smoothing) (Hyndman & Athanasopoulos, 2018), Theta 방법 (Assimakopoulos & Nikolopoulos, 2000)과 같은 고전적인 접근 방식이 이에 해당합니다.  
반면 글로벌 모델은 하나의 모델이 데이터셋 내 모든 시계열에 걸쳐 매개변수를 공유하는 방식으로, 지난 10여 년간 딥러닝 기반 접근법을 중심으로 널리 활용되어 왔습니다.

대표적인 글로벌 모델로는 DeepState (Rangapuram et al., 2018), DeepAR (Salinas et al., 2020), TimeGrad (Rasul et al., 2021)와 같은 순환 신경망 기반 모델, N-BEATS (Oreshkin et al., 2020) 및 N-HITS (Challu et al., 2023)와 같은 스택형 아키텍처, 그리고 TFT (Lim et al., 2021)와 PatchTST (Nie et al., 2023)와 같은 트랜스포머 기반 아키텍처가 있습니다.


최근에는 사전 학습된 예측 모델이 시계열 예측 분야에서 새로운 패러다임으로 부상하고 있습니다.  
기존 연구들에서도 예측을 위한 전이 학습의 제한적인 가능성이 관찰되었지만 (Orozco & Roberts, 2020; Oreshkin et al., 2021; Jin et al., 2022; Nie et al., 2023), 사전 학습 모델은 대규모 언어 모델과 유사한 원칙을 채택함으로써 다양한 데이터셋에 대한 제로샷 일반화를 가능하게 합니다.  
초기 연구들은 언어 모델을 시계열 문제에 직접 적용하는 데 초점을 맞추었으나 (Gruver et al., 2023; Jin et al., 2024), 최근 접근법들은 주로 LLM의 아키텍처적 아이디어를 차용하여 시계열 데이터에 대해 직접 사전 학습을 수행합니다 (Das et al., 2024b; Garza et al., 2024; Ansari et al., 2024).


대부분의 사전 학습된 예측 모델은 단변량 예측에 국한되며, 다변량 설정에서는 각 차원을 독립적으로 처리하거나 공변량 정보를 무시하는 경우가 많습니다 (Rasul et al., 2023; Das et al., 2024b; Ansari et al., 2024; Liu et al., 2025; Auer et al., 2025b).  
예외적으로 Moirai-1 (Woo et al., 2024)과 Toto (Cohen et al., 2025)는 다변량 구조를 아키텍처에 통합합니다.  
그러나 Moirai-1은 다변량 입력을 내부적으로 평탄화하여 고차원 확장성에 한계가 있으며, Toto는 교차 변수 어텐션을 도입하지만 알려진 공변량이나 범주형 공변량을 지원하지 않습니다.  
COSMIC (Auer et al., 2025a)은 합성 증강을 통해 공변량 활용을 확장하지만 단변량 타겟에만 적용됩니다.  
또한 TabPFN-TS (Hoo et al., 2025)는 알려진 공변량을 통합할 수 있으나, 과거 공변량이나 다변량 타겟을 모델링하지는 못합니다.

이러한 연구들이 제안되었음에도 불구하고, 경험적 분석에 따르면 대부분의 접근 방식은 단변량 모델 대비 제한적인 성능 향상만을 제공하며 (Żukowska et al., 2024; Auer et al., 2025a), 제로샷 설정에서 다변량 종속성과 공변량을 효과적으로 통합하는 문제는 여전히 해결되지 않은 과제로 남아 있습니다.


본 연구에서는 이러한 격차를 해소하기 위해 그룹 어텐션 메커니즘을 제안합니다.  
이 메커니즘은 다변량 예측을 위한 교차 어텐션 아키텍처 (Zhang & Yan, 2023; Rao et al., 2021; Arnab et al., 2021)와 여러 단변량 시계열 간의 교차 학습 (Das et al., 2024a) 개념을 일반화한 것입니다.  
기존 접근 방식과 달리, 그룹 어텐션은 관련 시계열의 그룹 단위로 작동하며, 아키텍처 수정이나 작업별 적응 없이도 단변량, 다변량, 공변량 기반 예측 작업을 하나의 통합된 프레임워크에서 자연스럽게 처리할 수 있습니다.  
표 1은 Chronos-2와 기존 사전 학습 모델들의 기능을 비교합니다.

![](/assets/img/posts/chronos2/a5b3a2624b9f1054792c1af928ce925a.png)


---
### 내용

- 시계열 : $Y_{1:T} = [y_1, \dots, y_T]$, $y_t \in \mathbb{R}^D$
	- 단변량 : $D=1$
	- 다변량 : $D>1$
- 예측 구간 : $H$, $Y_{T+1:T+H}$

- 공변량 시계열 : $X_{1:T+H} = [x_1, \dots, x_{T+H}]$, $x_t \in \mathbb{R}^M$
	- 이때 공변량은, 과거 구간($t \le T$), 미래 구간($t > T$) 둘 다 제공되 수 있음.

- 예측 문제 : 점 추정(단일 값), 확률론적 예측
$$P(Y_{T+1:T+H} \mid Y_{1:T}, X_{1:T+H})$$
- <strong>zero-shot</strong> : <strong>추가적인 학습, 적응 또는 fine tunning 없이, 관측되지 않은 시계열 데이터셋에 대해 모델이 직접 예측</strong>

#### Local vs Global model
#### Local
데이터셋 내 <strong>각 시계열마다 개별적인 매개변수 집합을 학습</strong>
- ARIMA, Exponential Smoothing, Theta... 와 같은 고전 통계 모델

#### Global
하나의 모델이 데이터셋 내 <strong>모든 시계열에 걸쳐 매개변수를 공유</strong> 
- DL모델
- RNN : DeepState, DeepAR, TimeGrad
- Stack : N-BEATS, N-HITS
- TF : TFT, PatchTST

#### 사전 학습 모델
기존에도 전이 학습 자체의 제한적인 가능성이 관찰되었고, 사전 학습 모델을 통해서 LLM과 유사한 원칙을 채택하여 zero-shot 일반화를 가능하게 한다.
초기 연구는 언어 모델을 시계열 문제에 직접 적용하는데 초점을 맞추었으나, 최근에는 <strong>LLM의 주요 아키텍처 아이디어를 차용하여 시계열 데이터에 직접 사전 학습을 시키는 트랜드</strong>
→ <strong>대부분 단변량 예측</strong>에 국한되어, <strong>각 차원을 독립적으로 처리하거나 공변량 정보를 무시</strong>하는 경우가 많다.

<strong>다양한 다변량 모델들의 한계</strong>(2025)
- Moirari-1의 경우 다변량 입력을 내부적으로 평탄화하여 처리 → 고차원 확장에 한계가 있음
- COSMIC의 경우, 합성 증강을 통해 공변량 활용을 확장하지만 이는 단변량 타겟에만 적용
- TabPFN-TS의 경우, 알려진 공변량 정보를 통합할 수 있으나, 과거의 공변량 혹은 다변량 타겟을 모델링할 수 없다.

#### GroupAttention
이러한 격차를 해소하기 위해 그룹 어텐션을 제안함 → 교차 어텐션 아키텍처 (Zhang & Yan, 2023; Rao et al., 2021; Arnab et al., 2021)와 여러 단변량 시계열 간의 교차 학습 (Das et al., 2024a) 개념을 일반화
기존의 접근과 달리, group attention은 시계열의 그룹 단위로 작동하며 아키텍처의 수정이나 adaptation없이 단변량, 다변량, 공변량을 하나의 통합된 프레임워크에서 처리할 수 있음.

![](/assets/img/posts/chronos2/c60bf9bc9e16dc7a63e51943a9e935b8.png)


---
### 포인트

> [!NOTE]
> <strong>DeepAR</strong>
> 확률적 자기회귀 RNN 기반 모델
> - 순환신경망을 통해 과거의 시계열로부터 <strong>미래 확률분포를 직접적으로 예측</strong>
> - 여러 시계열을 global하계 학습, 분포 현태로 표현하므로 불확실성을 모델링 가능
> - 과거값과 공변량을 입력받아 예측 분포를 출력
> 
> <strong>강점</strong>
> - 불확실성 모델링 가능
> - 대규모 시계열에 강함
> - 전처리 필요성이 비교적 낮음

> [!NOTE]
> <strong>DeepState</strong>
> 상태 공간 모델(SSM) + 딥러닝
> <strong>SSM → 동적 상태(state)와 관측(obs.)을 동시에 학습</strong>하며 딥러닝을 통해 데이터를 통해 자동으로 학습하되, 구조적 해석 가능성을 남겨둠
> 
> <strong>강점</strong>
> - 적은 데이터로 구조적 학습 가능
> - 모델 해석에 유리

> [!NOTE]
> <strong>TimeGrad</strong>
> 확산(diffusion) 기반 확률 시계열 모델
> diffusion model을 시계열에 적용하여 <strong>확률적 분포 샘플링 기반 예측</strong>, <strong>다변량 확률적 패턴을 추정</strong>
> 
> <strong>강점</strong>
> - <strong>불확실성이 큰 문제</strong>에 유리
> - 자연스럽게 다변량 시계열 구조를 다룸
> 
> <strong>단점</strong>
> - 계산량
> - 샘플링 비용

> [!NOTE]
> <strong>N-BEATS</strong>
> MLP 기반 Residual 네트워크
> 완전한 <strong>MLP</strong>기반 구조 + <strong>Residual link</strong>를 활용
> 트렌드/계절성 분해를 진행하는 <strong>해석가능한 모드</strong>와 일반적인 모드 둘 다 가능
> 
> <strong>강점</strong>
> - RNN, TF에 비해 복잡하지 않으면서 강력한 성능
> - 낮은 데이터 가용성에도 훌륭한 성능

> [!NOTE]
> <strong>N-HITS</strong>
> Neural Hierarchical Interpolation 모델
> N-BEATS에서 확장된 모델, <strong>계층적 보간</strong>을 이용해 장기적 패턴을 예측하며 서로 다른 스케일 패턴을 분리하여 예측에 반영
> 
> <strong>강점</strong>
> - TF보다 좋을떄도 있음
> - 장기예측 성능에 강점
> - 복잡한 시계열 패턴을 다중 스케일로 포착
> 

> [!NOTE]
> <strong>TFT</strong>
> Transformer 기반 다변량 시계열 예측
> TF기반으로, 정적 변수와 시계열 변수를 동시에 처리하며 gating과 variable selection을 포함해 <strong>해석가능성을 제공</strong>
> 
> <strong>강점</strong>
> - 여러 미래 시점에 대한 예측에 강점(MultiHorizon Forecasting)
> - 다양한 입력 형태에 대해 처리

> [!NOTE]
> <strong>PatchTST</strong>
> Transformer Long-term forecasting
> 시계열 데이터를 패치화 시켜서 TF에 적용하여 긴 시계열에 대한 의존성을 잘 학습
> 
> <strong>강점</strong>
> - 장기 예측의 효율성
> - TF 계산력 향상

---
### 📚 3 The Chronos-2 Model
### 번역

### 3 Chronos-2 모델

본 절에서는 Chronos-2 모델을 소개합니다.  
먼저 스케일링 및 토큰화 과정을 설명하고, 이어서 Chronos-2의 인컨텍스트 학습 능력을 가능하게 하는 그룹 어텐션 메커니즘을 포함한 아키텍처를 다룹니다.  
마지막으로 Chronos-2의 학습 및 추론 파이프라인을 논의합니다.  
Chronos-2의 전체 추론 파이프라인은 그림 1에 시각화되어 있습니다.

---

#### 3.1 스케일링 및 토큰화

##### 입력 구성

모델은 타겟 시계열 $Y_{1:T}$와 공변량 $X_{1:T+H}$에서 파생된 두 개의 입력으로 작동합니다.  
모든 과거 값은
$$
V = [v_1, \dots, v_T]
$$
로 정의되며, 각
$$
v_t \in \mathbb{R}^{D+M}
$$
은 타겟 관측값 $y_t$와 해당 공변량 벡터 $x_t$를 연결한 것입니다.

마찬가지로, 미래 값은
$$
W = [w_{T+1}, \dots, w_{T+H}]
$$
로 정의되며, 각
$$
w_t \in \mathbb{R}^{D+M}
$$
은 알려진 미래 공변량 값을 포함합니다.  
타겟 또는 과거 전용 공변량에 해당하는 항목은 미래 구간에서 누락된 값으로 설정됩니다.

공변량 $X_{1:T+H}$에 포함된 범주형 변수는 $V$와 $W$로 연결되기 전에 실수값 표현으로 변환됩니다.  
단변량 타겟의 경우, 타겟과의 관계를 기반으로 각 범주를 수치 값으로 매핑하는 타겟 인코딩을 적용합니다.  
다변량 타겟의 경우에는 각 범주에 고유한 정수를 할당하는 순서형 인코딩을 사용합니다.

---

##### 강건한 스케일링

입력 값 $V$와 $W$는 임의의 스케일을 가질 수 있으므로, 토큰화 파이프라인은 시계열 정규화부터 시작합니다.  
본 연구에서는 표준화 이후 추가적으로 $\sinh^{-1}$ 변환을 적용합니다.  
이 로그 유사 변환은 분산을 안정화하고 이상치가 목적 함수에 미치는 영향을 줄이는 데 효과적입니다.

형식적으로, 각 과거 값과 미래 값은 다음과 같이 정규화됩니다.
$$
\tilde{v}_{t,d} =
\frac{\sinh^{-1}(v_{t,d} - \mu_d)}{\sigma_d},
\quad t \in \{1, \dots, T\}
$$

$$
\tilde{w}_{t,d} =
\frac{\sinh^{-1}(w_{t,d} - \mu_d)}{\sigma_d},
\quad t \in \{T+1, \dots, T+H\}
$$

여기서 $\mu_d$와 $\sigma_d$는 각각 과거 값 $[v_{1,d}, \dots, v_{T,d}]$의 평균과 표준편차입니다.  
결측값은 $\mu_d$와 $\sigma_d$ 계산에서 제외됩니다.

정규화된 과거 값과 미래 값을 연결하여 다음 입력 행렬을 구성합니다.
$$
U = [\tilde{V}, \tilde{W}] \in \mathbb{R}^{(T+H) \times (D+M)}
$$

---

##### 메타 특징 (Meta Features)

토큰화 과정에서 $U$의 각 차원은 독립적으로 처리됩니다.  
하나의 차원 $d$에 해당하는 열
$$
u_d = [u_{1,d}, \dots, u_{T+H,d}]^\top
$$
에 대해 두 가지 메타 특징을 추가합니다.

첫째, 시간 인덱스
$$
j = -\frac{T}{C}, \dots, 0, \dots, \frac{H-1}{C}
$$
는 각 시간 단계의 상대적 위치를 인코딩하며, $C$는 모델이 지원하는 최대 컨텍스트 길이입니다.

둘째, 마스크 $m_d$는 값이 관측되었을 때 1, 그렇지 않을 때 0인 이진 지표로,  
과거의 결측값과 미래에 알려진 공변량을 구분하는 역할을 수행합니다.  
마스크 적용 이후 모든 결측값은 0으로 대체됩니다.

---

##### 패칭 및 임베딩

입력 시계열과 메타 특징은 길이 $P$의 비겹침 패치로 분할됩니다.  
컨텍스트와 미래 구간은 별도로 패치화되며, $T$ 또는 $H$가 $P$의 배수가 아닌 경우 제로 패딩이 적용됩니다.

각 패치 $(u_p, j_p, m_p)$는 다음 잔차 네트워크를 통해 임베딩됩니다.
$$
h_p = f_{\text{in},\phi}(u_p, j_p, m_p),
\quad
f_{\text{in},\phi} : \mathbb{R}^{3P} \rightarrow \mathbb{R}^{D_{\text{model}}}
$$

여기서 $\phi$는 네트워크 파라미터이며, $D_{\text{model}}$은 트랜스포머의 은닉 차원입니다.  
컨텍스트와 미래 패치 사이에는 구분 토큰이자 어텐션 싱크 역할을 하는 REG 토큰을 삽입합니다.

---

#### 3.2 아키텍처

Chronos-2는 인코더 전용 트랜스포머 모델로, T5 인코더 설계를 따른 구조입니다.

##### 시간 어텐션

시간 어텐션 레이어는 시간 축을 따라 셀프 어텐션을 적용하여 동일 차원의 패치들 간 정보를 집계합니다.  
상대 위치 임베딩으로는 RoPE를 사용합니다.

##### 그룹 어텐션

그룹 어텐션 레이어는 Chronos-2의 인컨텍스트 학습을 가능하게 하는 핵심 요소입니다.  
동일한 패치 인덱스에서 동일 그룹에 속한 시계열 간의 정보만을 집계합니다.

그룹은 다음과 같이 정의될 수 있습니다.

- 단일 시계열 (순수 단변량 예측)
- 관련 시계열 집합 (교차 학습)
- 공유 동적 특성을 가진 변수 집합 (다변량 예측)
- 타겟과 공변량 집합 (공변량 기반 예측)

각 항목은 그룹 ID $g$로 식별되며, 그룹 어텐션은 이를 기반으로 2차원 마스크를 구성합니다.  
그룹 내에는 순서 개념이 없으므로 위치 임베딩은 사용되지 않습니다.

---

##### Quantile Head

트랜스포머 스택 이후, 미래 패치 임베딩은 잔차 블록을 통과하여 분위수 예측
$$
\hat{Z} \in \mathbb{R}^{H \times D \times |Q|}
$$
을 생성합니다.

Chronos-2는 다음 21개 분위수를 예측합니다.
$$
Q = \{0.01, 0.05, 0.1, \dots, 0.9, 0.95, 0.99\}
$$

#### 3.3 훈련

훈련 배치는 단변량, 다변량, 공변량 기반 예측 작업을 혼합하여 구성됩니다.  
각 작업은 $(D, M)$과 각 차원의 역할로 정의되며, 작업별 그룹 ID가 할당됩니다.

모델은 분위수 회귀 손실을 사용하여 학습됩니다.
$$
\sum_{q \in Q}
\left[
q \cdot \max(z - \hat{z}_q, 0)
+ (1-q) \cdot \max(\hat{z}_q - z, 0)
\right]
$$

손실은 타겟 차원에 대해서만 계산되며, 알려진 공변량과 결측 타겟은 제외됩니다.  
출력 패치 수는 훈련 중 무작위로 샘플링됩니다.

훈련은 두 단계로 진행됩니다.  
첫 단계에서는 컨텍스트 길이 2048로 사전 학습을 수행하고,  
두 번째 단계에서는 컨텍스트 길이를 8192로 확장하여 장기 의존성을 학습합니다.

---

#### 3.4 추론

분위수 예측은 다음 역정규화를 통해 원래 스케일로 복원됩니다.
$$
\hat{y}_{q,t,d} = \mu_d + \sigma_d \cdot \sinh(\hat{z}_{q,t,d})
$$

추론 시 그룹 ID를 통해 다음 설정을 처리할 수 있습니다.

- 단변량 예측: 각 시계열에 고유한 그룹 ID
- 다변량 예측: 동일 시계열의 변수에 동일 그룹 ID
- 공변량 기반 예측: 타겟과 공변량에 동일 그룹 ID

Chronos-2는 모든 항목이 동일 그룹에 속하는 전체 교차 학습 모드도 지원합니다.

---
### 내용

#### Input
타겟 시계열 $Y_{1:T}$와 공변량 $X_{1:T+H}$에서 파생된 두 개의 입력으로 작동

##### 과거값
각 $v_t$는 타겟 관측값 $y_t$와 해당 공변량 벡터 $x_t$를 연결
$$V = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_T \end{bmatrix} = \begin{bmatrix} y_1 & x_1 \\ y_2 & x_2 \\ \vdots & \vdots \\ y_T & x_T \end{bmatrix} \in \mathbb{R}^{T \times (D+M)}$$

##### 미래값 : 예측해야할 값 + 이미 알고 있는 값
각 $w_t$는 타겟 값 $y_{T+i}$와 알고 있는 공변량 벡터 $x_{T+i}$를 연결
$$W = \begin{bmatrix} w_{T+1} \\ w_{T+2} \\ \vdots \\ w_{T+H} \end{bmatrix} = \begin{bmatrix} * & x_{T+1} \\ * & x_{T+2} \\ \vdots & \vdots \\ * & x_{T+H} \end{bmatrix} \in \mathbb{R}^{H \times (D+M)}$$
<strong>범주형 변수</strong>
공변량에 포함된 범주형 변수는 V와 W로 연결되기 전에 <strong>실수값 표현</strong>으로 변환
- 단변량 : 타겟과의 관계를 기반으로 각 범주를 수치 값으로 매핑하는 <strong>타겟 인코딩</strong>
- 다변량 : 각 번주의 고유한 정수를 할당 <strong>순서형 인코딩</strong>

#### Robust Scailing

입력값 $V, W$는 임의의 스케일을 가질 수 있으므로
- standardization + $\sinh^{-1}$ 변환
	- 특히 아크사인변환의 경우, 분산을 안정화시키고 이상치가 목적 함수에 미치는 영향을 줄이는데 효과적임
$$
\tilde{v}_{t,d} =
\frac{\sinh^{-1}(v_{t,d} - \mu_d)}{\sigma_d},
\quad t \in \{1, \dots, T\}
$$

$$
\tilde{w}_{t,d} =
\frac{\sinh^{-1}(w_{t,d} - \mu_d)}{\sigma_d},
\quad t \in \{T+1, \dots, T+H\}
$$

- $\mu_d$와 $\sigma_d$는 각각 과거 값 $[v_{1,d}, \dots, v_{T,d}]$의 평균과 표준편차
- 결측값은 $\mu_d$와 $\sigma_d$ 계산에서 제외
- 
정규화된 과거 값과 미래 값을 연결하여 Input을 구성
$$
U = [\tilde{V}, \tilde{W}] \in \mathbb{R}^{(T+H) \times (D+M)}
$$

#### Meta Features
토큰화 과정에서 $U$의 각 차원을 <strong>독립적으로 처리</strong>
하나의 차원 $d$에 해당하는
$$
u_d = [u_{1,d}, \dots, u_{T+H,d}]^\top
$$
에서 메타 feature을 추가함
1. <strong>시간 인덱스</strong>
$$
j = -\frac{T}{C}, \dots, 0, \dots, \frac{H-1}{C}
$$
	각 시간 단계의 <strong>상대적 위치를 인코딩</strong>하며, $C$는 모델이 지원하는 최대 컨텍스트 길이
2. <strong>마스크</strong> $m_d$
	값이 관측되었을 때 1, 그렇지 않을 때 0인 이진 지표. 과거의 결측값과 미래에 알려진 공변량을 구분하는 역할을 수행. 마스크 적용 이후 모든 결측값은 0으로 대체
#### 패칭 및 임베딩
입력 시계열과 메타 특징은 길이 $P$의 윈도우로 분할
과거와 미래 구간은 별도로 패치화, $T$ 또는 $H$가 $P$의 배수가 아닌 경우 제로 패딩이 적용
각 패치 $(u_p, j_p, m_p)$는 다음 잔차 네트워크를 통해 임베딩
$$
h_p = f_{\text{in},\phi}(u_p, j_p, m_p),
\quad
f_{\text{in},\phi} : \mathbb{R}^{3P} \rightarrow \mathbb{R}^{D_{\text{model}}}
$$
- $\phi$ : 네트워크 파라미터
- $D_{\text{model}}$ : 트랜스포머의 은닉 차원  
컨텍스트와 미래 패치 사이 어텐션 싱크 역할을 하는 REG 토큰을 삽입

#### 아키텍처
Chronos-2는 인코더 전용 트랜스포머 모델로, T5 인코더 구조
##### Time Attention
- 시간 축을 따라 self-attention, 동일 차원의 패치 간 정보를 집계
- 위치 임베딩 : <strong>RoPE</strong>
##### Group Attention
<strong>동일 그룹에 속한 시계열 간의 정보만 집계</strong>
그룹의 정의
- 단변량 시계열
- 관련 시계열 집합(교차 학습)
- 공유된 동적인 특징을 가지는 변수 집합(다변량 시계열)
- 다켓과 공변량 집합(공변량)
각 그룹은 그룹 ID $g$로 식별되며, 이를 기반으로 2차원 마스크를 구성하여 어텐션
그룹 내에는 순서개념이 없으므로 위치 임베딩은 사용하지 않음
##### Quantile Head
미래 패치 임베딩은 잔차블럭을 통과하며 생성
$$
\hat{Z} \in \mathbb{R}^{H \times D \times |Q|}
$$

Chronos-2는 다음 21개 분위수를 예측
$$
Q = \{0.01, 0.05, 0.1, \dots, 0.9, 0.95, 0.99\}
$$
#### 훈련
위의 다양한 그룹 정의를 혼합하여 구성시킴
각 작업은 $(D, M)$과 각 차원의 역할로 정의, 작업별 그룹 ID가 할당
#### Loss : quantile reg. loss
$$
\sum_{q \in Q}
\left[
q \cdot \max(z - \hat{z}_q, 0)
+ (1-q) \cdot \max(\hat{z}_q - z, 0)
\right]
$$
target에 대해서만 계산, 미래의 공변량과 결측 타겟은 제외
패치 수는 무작위 샘플링(훈련중)
<strong>과정</strong>
1. 컨텍스트 길이 2048 : pretrain
2. 컨텍스트 길이를 8192 : 장기 의존성을 학습
#### 추론

분위수 예측 역정규화를 통해 원래 스케일로 복원
$$
\hat{y}_{q,t,d} = \mu_d + \sigma_d \cdot \sinh(\hat{z}_{q,t,d})
$$

추론 시 그룹 ID를 통해 다음 설정을 처리
- 단변량 예측: 각 시계열에 고유한 그룹 ID
- 다변량 예측: 동일 시계열의 변수에 동일 그룹 ID
- 공변량 기반 예측: 타겟과 공변량에 동일 그룹 ID
Chronos-2는 모든 항목이 동일 그룹에 속하는 전체 교차 학습 모드도 지원


---
### 포인트
#### Input
##### Raw
$$V = \begin{bmatrix} y_1 & x_1 \\ y_2 & x_2 \\ \vdots & \vdots \\ y_T & x_T \end{bmatrix} \in \mathbb{R}^{T \times (D+M)}, \quad W = \begin{bmatrix} * & x_{T+1} \\ * & x_{T+2} \\ \vdots & \vdots \\ * & x_{T+H} \end{bmatrix} \in \mathbb{R}^{H \times (D+M)}$$
##### after Concat with scailing
$$U = \begin{bmatrix} \tilde{V} \\ \hline \tilde{W} \end{bmatrix} = \begin{bmatrix} \tilde{v}_{1,1} & \dots & \tilde{v}_{1,D+M} \\ \vdots & \ddots & \vdots \\ \tilde{v}_{T,1} & \dots & \tilde{v}_{T,D+M} \\ \hline \tilde{w}_{T+1,1} & \dots & \tilde{w}_{T+1,D+M} \\ \vdots & \ddots & \vdots \\ \tilde{w}_{T+H,1} & \dots & \tilde{w}_{T+H,D+M} \end{bmatrix} \in \mathbb{R}^{(T+H) \times (D+M)}$$
##### Input D
$$\text{Input for Dim } d = \begin{bmatrix} 
u_{1,d} & j_1 & m_{1,d} \\
\vdots & \vdots & \vdots \\
u_{T,d} & j_T & m_{T,d} \\
\hline
u_{T+1,d} & j_{T+1} & m_{T+1,d} \\
\vdots & \vdots & \vdots \\
u_{T+H,d} & j_{T+H} & m_{T+H,d} 
\end{bmatrix} \in \mathbb{R}^{(T+H) \times 3}$$
##### Patching : u를 P개씩
###### 하나의 패치
$$\text{Patch}_p = [\underbrace{u_{t}, \dots, u_{t+P-1}}_{P \text{ values}}, \quad \underbrace{j_{t}, \dots, j_{t+P-1}}_{P \text{ indices}}, \quad \underbrace{m_{t}, \dots, m_{t+P-1}}_{P \text{ masks}}]^\top \in \mathbb{R}^{3P}$$
###### x
$$\mathbf{X}_{\text{sequence}} = \Big[ \underbrace{\mathbf{h}_1, \dots, \mathbf{h}_n}_{\text{Context Patches}} , \quad \mathbf{e}_{\text{REG}} , \quad \underbrace{\mathbf{h}_{n+1}, \dots, \mathbf{h}_{n+m}}_{\text{Future Patches}} \Big]$$
$$\mathbf{h}_p = f_{\text{in}} \left( \left[ \underbrace{u_{t:t+P, d}}_{\text{Values}} , \underbrace{j_{t:t+P}}_{\text{Indices}} , \underbrace{m_{t:t+P, d}}_{\text{Masks}} \right] \right) \in \mathbb{R}^{D_{\text{model}}}$$
> [!NOTE]
> <strong>RoPE(Rotary Position Embedding)</strong>
> 입력 벡터를 특정 각도만큼 <strong>회전</strong>시켜 벡터를 더해주는 방법
> e.g. t번째 위치 → t번째와 비례하는 각도 $\theta$만큼 회전 변환
> 이를 통해 두 토큰 사이의 어텐션 계산을 진행할때, 결과값이 두 토큰의 <strong>상대적 거리</strong>에만 의존하게됨

#### Attention
- <strong>Time Attention</strong> : 어제와 오늘 - 시간 관계 학습
- <strong>Group Attention</strong> : 같은 시간대에 있는, 변량간 학습
	- <strong>단변량</strong> : 각 시계열마다 서로 다른 ID로 자기의 과거를 통해 학습
	- <strong>관련 시계열 집합(교차 학습)</strong> : 비슷한 상품, 지역의 데이터를 같은 ID로 묶음, 데이터가 부족한 시계열이 다른 데이터의 패턴을 context로 빌려올 수 있음(few-shot)
	- <strong>다변량</strong> : 한 시스템의 여러 변수를 같은 ID로 묶음, 변수들이 어떻게 묶여있는지 배움
	- <strong>타겟과 공변량</strong> : 예측하려는 값과 힌트를 같은 그룹으로 묶음

### 📚 4 Training Data
### 번역

Chronos-2와 같은 범용 사전 학습(pretrained) 모델의 경우, 학습 데이터는 모델의 특정 아키텍처보다도 더 결정적인 역할을 하는 경우가 많습니다.  
최근 대규모 시계열 데이터셋의 가용성이 확대되고 있음에도 불구하고(Woo et al., 2024; Ansari et al., 2024; Aksu et al., 2024), 이러한 데이터셋들은 주로 단변량 시계열로 구성되어 있습니다.  
이러한 한계를 극복하고 Chronos-2에 인컨텍스트 학습(in-context learning) 능력을 부여하기 위해, 본 연구에서는 합성 데이터에 크게 의존하였습니다.

---

#### 4.1 단변량 데이터

Chronos(Ansari et al., 2024) 및 GIFT-Eval(Aksu et al., 2024)의 사전 학습 코퍼스에서 선별한 데이터셋을 Chronos-2의 학습 코퍼스에 통합하였습니다.  
전체 데이터셋 목록은 부록의 표 6에 제시되어 있습니다.

![](/assets/img/posts/chronos2/ed9bc2ce6de7c42d809cf5ec09d2bb4d.png)

데이터의 다양성을 더욱 향상시키기 위해, 다음의 두 가지 접근법을 사용하여 단변량 합성 데이터를 생성하였습니다.

- **TSI (Trend, Seasonality, Irregularity)**  
  Bahrpeyma et al.(2021)에 기반한 생성기로, 다양한 추세(trend), 계절성(seasonality), 불규칙성(irregularity) 구성 요소를 무작위로 조합하여 폭넓은 합성 시계열을 생성합니다.

- **TCM (Temporal Causal Model)**  
  시간적 인과 모델(Runge et al., 2023)에서 무작위 인과 그래프를 샘플링한 뒤, 이를 기반으로 자기회귀(auto-regressive) 과정을 통해 시계열을 생성합니다.
#### 4.2 다변량 데이터

다변량 예측 및 공변량(covariate) 기반 작업의 경우, 본 연구에서는 전적으로 합성 데이터에 의존하였습니다.  
다양한 다변량 구조를 표현하기 위해, 우리는 **멀티베리아타이저(multivariatizer)**라는 개념을 도입합니다.

멀티베리아타이저는 기본 단변량 생성기로부터 여러 시계열을 샘플링한 뒤, 이들 사이에 종속성을 부여함으로써 다변량 동역학을 생성하는 메커니즘입니다.  
기본 단변량 생성기로는 자기회귀(AR) 모델, 지수 평활(ETS) 모델, TSI, 그리고 KernelSynth(Ansari et al., 2024) 등 다양한 모델을 사용하였습니다.

우리는 다음의 두 가지 주요 멀티베리아타이저 클래스를 사용하였습니다.

- **동시 멀티베리아타이저 (Cotemporaneous Multivariatizer)**  
  기본 단변량 생성기에서 샘플링된 시계열들에 대해 동일한 타임스텝에서 선형 또는 비선형 변환을 적용합니다.  
  이를 통해 시계열 간의 즉각적인 상관관계를 도입한 다변량 시계열을 생성합니다.

- **순차 멀티베리아타이저 (Sequential Multivariatizer)**  
  시간 축을 따라 종속성을 유도함으로써 선행–지연 효과(lead–lag effect)나 공적분(cointegration)과 같은 보다 풍부한 다변량 특성을 생성합니다.

멀티베리아타이저를 통해 생성된 다변량 시계열은  
(1) 모든 변량을 예측해야 하는 다변량 예측 태스크와,  
(2) 변량의 일부가 무작위로 관측 가능한 공변량으로 지정되는 공변량

![](/assets/img/posts/chronos2/9b4b22dc3f79e2201f9e29cfae4493e0.png)

---
### 내용

#### 단변량

Chronos, GIFT-Eval의 사전 학습에서 선변한 데이터셋을 Chronos-2에 사용
##### 합성
- <strong>TSI(Trend, Seasonality, Irregularity)</strong> : trend(추세), seasonality(계절성), irregularity(불확실성)을 구성 요소를 무작위로 조합
- <strong>TCM(Temporal Casual Model)</strong> : 시간적 인과 모델에서 무작위 인과 그래프를 샘플링, AR(자귀 회귀)방식으로 시계열을 생성

![](/assets/img/posts/chronos2/463b9365fa14860ea1fbd2c1f917d98e.png)

#### 다변량
다변량 예측 및 공변량 task의 경우, 전적으로 합성 데이터에 의존
<strong>Multivariatizer</strong>라는 개념을 도입
	단변량 생성기로부터 여러 시계열을 샘플링 → 이들 사이에 종속성을 부여 = 다변량을 생성하는 메커니즘
	단변량 생성기
	- AR model
	- ETS model
	- KernelSynth 등
- <strong>Cotemporaneous Multivariatizer(동시 멀티베리아타이저)</strong> : 기본 단변량 생성기에서 샘플링된 시계열들에 대해 동일한 시간에서 선형 또는 비선형 변환을 적용 → 이를 통해 시계열간 즉각적인 상관관계를 도입한 시계열을 생성
- <strong>Sequential Multivariatizer(순차 멀티베리아타이저)</strong> : 시간 축을 따라 종속성을 유도 → 선행-지연효과 혹은 공적분과 같은 다변량 특성을 생성

---
### 포인트

> [!NOTE]
> <strong>TSI(Trend, Seasonality, Irregularity)</strong>
> 시계열 데이터를 구성하는 3가지 성분(추세, 계절성, 불확실성)을 무작위로 조합(기존데이터로부터 유래 X)하여 단변량 데이터를 생성 → 모델이 학습해보지 못한 희귀한 패턴, 데이터문제 해결
> 

> [!NOTE]
> <strong>TCM(Temporal Causal Model)</strong>
> 시간에 따른 인과 관계를 모델링
> <strong>무작위 인과 그래프</strong>를 샘플링 후 자기회귀, A사건이 발생하면 일정 시간 뒤 B사건에 영향을 준다는 시간적 인과 구조를 데이터에 부여
> - 무작위 인과 그래프 : 변수들 사이의 "원인과 결과"관계를 나타내는 지도를 무작위로 그려낸 것, 즉 컴퓨터가 어떤 변수가 어떤 변수의 원인 혹은 영향력이 얼마나 강할지를 무작위로 결정하며 이는 현실세계의 <strong>What-if</strong>시나리오를 위함

> [!NOTE]
> <strong>선행 지연 효과</strong>
> <strong>시간차 관계</strong>를 의미. 한 시계열(Lead)의 움직임이 다른 시계열(Lag)에 일정한 시간 간격을 두고 나중에 나타나는 현상
> e.g. 가격이 오르면(Lead)→완제품의 가격이 오르는(Lag)관계
> 

> [!NOTE]
> <strong>공적분</strong>
> 각각은 불안정(Non-Stationary)하여 제멋대로 움직이는 것 같지만, <strong>장기적으로는 일정한 관계를 유지하며 함께 움직이는 성질</strong>
> 개별 시계열 데이터가 추세를 가지고 있어 불안정하더라도, 이들의 선형 조합이 안정적인 상태가 될 때 공적분 관계에 있다고 한다.
> e.g. 술취한 사람과 개 : 서로 비틀거리지만 목줄로 연결되어 같은 방향으로는 간다

---
### 📚 5 Experiments
### 번역

본 절에서는 Chronos-2를 세 가지 포괄적인 벤치마크(5.1절)에 대해 최첨단 접근법들과 비교 평가한 실험 결과를 제시합니다.  
이후 단변량, 다변량, 그리고 공변량 정보 기반 예측 태스크에서 인컨텍스트 학습(in-context learning, ICL)을 통해 얻은 성능 향상을 분석합니다(5.2절).  
다음으로, 공변량이 정확한 예측에 핵심적인 역할을 하는 에너지 및 소매 도메인 태스크에서 Chronos-2의 성능을 조사합니다(5.3절).  
마지막으로, 더 작은 모델, 합성 데이터로만 학습된 모델, 그리고 장기 문맥 후속 학습 이전 모델을 포함한 Chronos-2의 다양한 축소·변형 모델에 대한 실험 결과를 보고합니다(5.4절).

#### 벤치마크 결과

**표 3: fev-bench 결과**  
평균 승률과 스킬 점수는 스케일링된 분위수 손실(Scaled Quantile Loss, SQL) 지표를 기준으로 계산되며, 두 값 모두 클수록 성능이 우수함을 의미합니다.  
Chronos-2는 단변량, 다변량, 공변량 정보 기반 예측 태스크를 모두 포함하는 fev-bench에서 기존의 모든 사전 학습 모델을 상당한 차이로 능가합니다.  
기본 결과와 특정 태스크의 데이터 누수를 처리하기 위한 대체 전략은 Shchur et al.(2025)에서 가져왔으며, 추가 예측 지표에 대한 결과는 부록의 표 7–9에 제시되어 있습니다.

본 연구에서는 120M 매개변수를 가진 기본 Chronos-2 모델을 다음의 세 가지 포괄적인 예측 벤치마크에서 평가했습니다.

- fev-bench (Shchur et al., 2025)  
- GIFT-Eval (Aksu et al., 2024)  
- Chronos Benchmark II (Ansari et al., 2024)

성능 비교를 위해, 각 벤치마크에서 가장 강력한 결과를 보고한 최첨단 시계열 파운데이션 모델들과 비교했습니다. 여기에는 TiRex, TimesFM-2.5, Toto-1.0, Moirai-2.0, TabPFN-TS, COSMIC, Sundial, 그리고 Chronos의 최신 공개 버전인 Chronos-Bolt가 포함됩니다.  
또한 통계적 예측 문헌(Hyndman & Athanasopoulos, 2018)에 기반한 AutoARIMA, AutoETS, AutoTheta 및 이들의 앙상블도 추가적인 기준선으로 포함했습니다.

이전 연구(Aksu et al., 2024; Ansari et al., 2024)에서 사전 학습된 모델이 평균적으로 태스크 특화 딥러닝 모델과 유사하거나 더 나은 성능을 보였음을 확인하였기 때문에, 본 연구에서는 Chronos-2를 사전 학습 모델들과만 비교하고 태스크별 딥러닝 모델은 평가에서 제외했습니다.

Shchur et al.(2025)을 따라, 모든 모델에 대해 평균 승률(W)과 스킬 점수(S)를 함께 보고합니다.  
이 지표들은 이전 연구에서 사용된 평균 순위(R) 및 기하 평균 상대 오차(G)와 수학적으로 동등하며, 다음과 같이 정의됩니다.

- $R = 1 + (1 - W / 100)(N - 1)$  
- $G = 1 - S / 100$

여기서 $N$은 비교된 모델의 수입니다.  
승률은 모델이 다른 모델을 능가하는 쌍대 비교의 비율을 의미하며, 스킬 점수는 기준 모델(Seasonal Naive) 대비 평균 백분율 성능 개선을 나타냅니다.

##### fev-bench

fev-bench는 공변량을 포함한 100개의 예측 태스크로 구성되어 있으며, 다양한 실제 시나리오를 가장 포괄적으로 포괄하는 벤치마크입니다.  
이 데이터셋들은 Chronos-2의 학습 과정에서 사용되지 않았습니다.

표 3은 SQL 지표 기준의 fev-bench 결과를 보여주며, Chronos-2는 승률과 스킬 점수 모두에서 기존 시계열 파운데이션 모델을 크게 능가합니다.  
또한 fev-bench는 모델 간 성능 차이가 통계적으로 유의미한지를 평가할 수 있는 도구를 제공합니다. 그림 2에 제시된 95% 신뢰구간 분석 결과는 Chronos-2가 TiRex 및 TimesFM-2.5를 통계적으로 유의미한 차이로 능가함을 보여줍니다.
##### GIFT-Eval

GIFT-Eval은 55개의 데이터셋에서 파생된 97개의 태스크로 구성되며, 고주파 시계열과 장기 예측에 중점을 둡니다.  
표 4의 결과에 따르면 Chronos-2는 가중 분위수 손실(WQL)과 평균 절대 스케일 오차(MASE) 모두에서 이전 최고 성능 모델을 능가합니다.

사전 학습 코퍼스를 구성할 때 GIFT-Eval의 테스트 구간과 중복되지 않도록 주의했으나, 일부 데이터셋의 학습 구간과는 부분적인 중복이 존재합니다.  
엄격한 제로샷 결과는 5.4절에서 합성 데이터만으로 학습된 모델을 통해 분석합니다.

##### Chronos Benchmark II

Chronos Benchmark II는 27개의 태스크로 구성되며, 평균적으로 300 타임스텝 미만의 짧은 이력을 포함합니다.  
이 벤치마크에서도 Chronos-2는 WQL 및 MASE 기준에서 모든 기존 모델을 일관되게 능가합니다(표 5 참조).

종합하면, Chronos-2는 세 가지 벤치마크 전반에서 모든 경쟁 모델을 능가하며, Chronos-Bolt 대비 상당한 성능 향상을 보여줍니다. 이는 아키텍처 개선과 학습 전략의 효과를 명확히 보여줍니다.
#### 인컨텍스트 학습을 통한 개선

5.1절의 결과는 전체 교차 학습 모드에서 ICL이 활성화된 Chronos-2의 성능입니다.  
본 절에서는 단변량 추론 모드와 비교하여 ICL로 인한 성능 향상을 분석합니다.

이를 위해 fev-bench를 다음의 세 부분집합으로 분할합니다.

- 단변량 서브셋: 단일 대상 시계열, 공변량 없음 (32개 태스크)  
- 다변량 서브셋: 다중 대상, 공변량 없음 (26개 태스크)  
- 공변량 서브셋: 하나 이상의 과거 또는 알려진 공변량 포함 (42개 태스크)

단변량 모드에서는 각 시계열을 독립적으로 예측하며 공변량은 무시됩니다.

##### 단변량 태스크

그림 3에서 볼 수 있듯이, ICL은 단변량 태스크에서 스킬 점수를 향상시킵니다.  
특히 짧은 이력을 포함하는 태스크가 많은 Chronos Benchmark II에서 효과가 두드러집니다. 이는 제한된 관측 구간에서도 관련 시계열 정보를 활용할 수 있음을 의미합니다.
##### 다변량 태스크

fev-bench의 다변량 서브셋에서는 ICL의 이득이 상대적으로 작습니다.  
흥미롭게도 단변량 모드의 Chronos-2는 네이티브 다변량 모델인 Toto-1.0보다도 우수한 성능을 보입니다.

이는 Takens의 임베딩 정리(Takens, 2006)가 시사하듯, 충분히 긴 단변량 이력이 시스템의 주요 동역학을 포착할 수 있기 때문일 수 있습니다.  
유사한 관찰은 Nie et al.(2023)에서도 보고되었습니다.
##### 공변량 포함 태스크

가장 큰 성능 향상은 공변량을 포함하는 태스크에서 관찰됩니다.  
ICL을 통해 Chronos-2는 공변량 정보를 효과적으로 활용하여 단변량 추론 대비 큰 성능 개선을 달성합니다.  
이 서브셋에서 두 번째로 우수한 성능을 보인 모델은 공변량을 지원하는 TabPFN-TS입니다.
#### 도메인별 사례 연구

에너지 및 소매 도메인에서 공변량을 포함하는 fev-bench 태스크를 대상으로 추가 분석을 수행했습니다.  
각각 16개(에너지)와 17개(소매) 태스크를 사용했으며, 기준선으로 TabPFN-TS와 TiRex를 선택했습니다.

결과적으로 Chronos-2는 모든 경우에서 기준선 모델을 일관되게 능가하며, 공변량 통합이 실제 예측 성능에 매우 중요함을 보여줍니다.

독일 에너지 가격 예측 태스크와 Rossmann 소매 판매 예측 태스크 사례 분석에서도, ICL은 공변량을 활용해 예측의 정확성과 해석 가능성을 크게 향상시킵니다.

#### Ablation Studies

본 절에서는 다양한 설계 선택의 영향을 분석하기 위한 추가 실험을 제시합니다.

##### 모델 크기

28M 매개변수의 소형 모델은 성능 저하가 매우 제한적인 반면, 추론 속도는 약 2배 향상됩니다. 이는 자원이 제한된 환경에 적합합니다.

##### 합성 데이터만 사용

합성 데이터만으로 학습한 Chronos-2-Synth는 실제 데이터가 포함된 모델보다 성능이 다소 낮지만, 여전히 강력한 결과를 보입니다. 이는 합성 데이터의 중요성을 강조합니다.

##### 긴 문맥 후속 학습

문맥 길이를 2,048에서 8,192 타임스텝으로 확장한 후속 학습은 성능을 전반적으로 향상시키며, 특히 장주기 계절성이 강한 고주파 데이터에서 효과가 큽니다.


---
### 내용

![](/assets/img/posts/chronos2/98d07da16e1eeae188918e14b9f8f877.png)

<strong>평균 승률, 스킬 점수 : SQL기준으로 계산되며 두 값 모두 클수록 성능이 우수함</strong>

추가예측지표에 대한 결과(부록 표 7~9)

![](/assets/img/posts/chronos2/8781fe7f057ffadda0a12bc6a8f9170d.png)

![](/assets/img/posts/chronos2/250d38f78634ea4719e75c3a060e2482.png)

![](/assets/img/posts/chronos2/4ab2d1bc6ebc76f8499a1cf7c801894e.png)
120M 매개변수를 가진 기본 Chronos-2 모델로 평가
- fev-bench (Shchur et al., 2025)  
- GIFT-Eval (Aksu et al., 2024)  
- Chronos Benchmark II (Ansari et al., 2024)

성능 비교를 위해, 각 벤치마크에서 가장 강력한 결과를 보고한 최첨단 시계열 파운데이션 모델들과 비교를 진행(TiRex, TimesFM-2.5, Toto-1.0, Moirai-2.0, TabPFN-TS, COSMIC, Sundial, Chronos의 최신 공개 버전인 Chronos-Bolt)
통계모델들도 AutoARIMA, AutoETS, AutoTheta 및 이들의 앙상블도 추가
이전 크로노스에서 다운스트림 task는 비슷하거나, 더 나은 경향을 보였기때문에 해당 작업은 제외함
#### fev-bench

![](/assets/img/posts/chronos2/98d07da16e1eeae188918e14b9f8f877.png)

공변량을 포함한 100개의 예측 task로 구성, 실제 시나리오를 가장 포괄적으로 포괄하는 벤치마크(학습과정에서 사용 안함)
모든 모델에 대해서 Chronos2가 능가함 

![](/assets/img/posts/chronos2/b5f90a83b4542d647b6878f5d863eb06.png)

fev-bench는 모델 간 성능 차이가 통계적으로 유의미한지를 평가할 수 있는 도구를 제공
그림 2에 제시된 95% 신뢰구간 분석 결과는 Chronos-2가 TiRex 및 TimesFM-2.5를 통계적으로 유의미한 차이로 능가함

#### GIFT-Eval
위에서부터 a, b
![](/assets/img/posts/chronos2/b1bc24502338a049cda349318678e193.png)
![](/assets/img/posts/chronos2/53857c349c8b6681567f3d8e39bf1787.png)

GIFT-Eval은 55개의 데이터셋에서 파생된 97개의 태스크로 구성되며, 고주파 시계열과 장기 예측에 중점
이전모델을 다 능가함
다만 중복된 데이터가 존재함(일부, 엄격한 결과는 5.4. zeroshot에서 분석)
#### Chronos Benchmark II

![](/assets/img/posts/chronos2/44c6f2d1fa17e1a76340afc4c493a6f8.png)

![](/assets/img/posts/chronos2/d71a06e33d1f13cec5e8a26c0224bd2d.png)

Chronos Benchmark II는 27개의 태스크로 구성되며, 평균적으로 300 타임스텝 미만의 짧은것도 포함

### 인컨텍스트 학습을 통한 개선

##### ICL vs Full Cross Learning
ICL은 그룹 어텐션 메커니즘을 통해 실현되는 모델의 <strong>전반적인 성능</strong>을 의미하며, 전체 교차 학습은 이 능력을 극대화시키기 위해 추론시 사용하는 <strong>특정한 모드</strong>이다
###### Group ID할당 방식
- ICL : 관련 있는 데이터끼리만 같은 그룹으로 할당
- 교차 : <strong>배치 안에 있는 모든 데이터에 대해 동리한 그룹으로 할당</strong>하며 데이터의 종류와 상관엇ㅂ이 서로의 정보를 무차별적으로 참조

이를 위해 fev-bench를 다음의 세 부분집합으로 분할

- 단변량 서브셋: 단일 대상 시계열, 공변량 없음 (32개 태스크) 
	- 단변량 모드에서는 각 시계열을 독립적으로 예측하며 공변량은 무시
- 다변량 서브셋: 다중 대상, 공변량 없음 (26개 태스크)  
- 공변량 서브셋: 하나 이상의 과거 또는 알려진 공변량 포함 (42개 태스크)

##### 단변량 태스크

![](/assets/img/posts/chronos2/f504f1fa8fe6aeb28d9dcbe64bac794d.png)
그림 3을 보면(개선점을 누적 표시), ICL은 단변량에서 스킬 점수를 향상시킴
짧은 시계열인 Chronos Benchmakr2에서도 개선이 보이며, 이는 시계열의 정보를 활용한다면 예측을 개선할 수 있음을 의미
##### 다변량 태스크
![](/assets/img/posts/chronos2/8e5880527e7baf8278041ba802d5ae46.png)
![](/assets/img/posts/chronos2/c52e062dc4fe34fd500f848b8921802f.png)

다변량 문제에 대해서는 ICL의 효과가 낮음
흥미롭게도 단변량 모드의 Chronos-2는 네이티브 다변량 모델인 Toto-1.0보다도 우수한 성능
이는 Takens의 임베딩 정리(Takens, 2006)가 시사하듯, 충분히 긴 단변량 이력이 시스템의 주요 동역학을 포착할 수 있기 때문
유사한 관찰은 Nie et al.(2023)에서도 보고됨
##### 공변량 포함 태스크


![](/assets/img/posts/chronos2/3542cd9e64bbe7b9f49f66552402b895.png)
가장 큰 성능 향상은 공변량을 포함하는 태스크에서 관찰됨
ICL을 통해 Chronos-2는 공변량 정보를 효과적으로 활용하여 단변량 추론 대비 큰 성능 개선을 확인함

#### 도메인별 사례 연구

에너지 및 소매 도메인에서 공변량을 포함하는 fev-bench 태스크를 대상으로 추가 분석을 수행
각각 16개(에너지)와 17개(소매) 태스크를 사용했으며, 기준선으로 TabPFN-TS와 TiRex를 선택

독일 에너지 가격 예측 태스크와 Rossmann 소매 판매 예측 태스크 사례 분석에서도, ICL은 공변량을 활용해 예측의 정확성과 해석 가능성을 크게 향상

![](/assets/img/posts/chronos2/86c9b931722ac94b8e1f94257d77a112.png)

![](/assets/img/posts/chronos2/2749d4ba5c56e80da64308bd4147c402.png)

#### Abalation Study
![](/assets/img/posts/chronos2/9630d45d31bc55eba2f9d87705d60c04.png)


##### 모델 크기

28M 매개변수의 소형 모델은 성능 저하가 매우 제한적인 반면, 추론 속도는 약 2배 향상 이는 자원이 제한된 환경에 적합

##### 합성 데이터만 사용

합성 데이터만으로 학습한 Chronos-2-Synth는 실제 데이터가 포함된 모델보다 성능이 다소 낮지만, 여전히 강력한 결과 이는 합성 데이터의 중요성을 강조

##### 긴 문맥 후속 학습

문맥 길이를 2,048에서 8,192 타임스텝으로 확장한 후속 학습은 성능을 전반적으로 향상시키며, 특히 장주기 계절성이 강한 고주파 데이터에서 효과적

---
### 포인트

> [!NOTE]
> <strong>평균 승률</strong>
> 특정 모델이 다른 모델과의 1:1비교에서 얼마나 자주 이기는지를 측정
> N개의 모델이 있을때, 각 데이터셋(task)마다 모델 A와 B의 예측 오차(Loss by SQL)을 비교하여 더 낮은 오차를 기록한 모델이 승자로 판단.
> 이후 모델 A가 다른 모델들을 상대로 거둔 승리 비율을 계싼하고, 이를 전체 태스크에 대해 평균
> $$
R = 1 + \left(1 - \frac{W}{100}\right)(N - 1)
> $$
> - $W$: 평균 승률 (Avg. Win Rate)  
> - $N$: 비교 대상 모델의 총 개수  
> - $R$: 모델의 평균 순위 (Average Rank)
> 승률 100%에 가까울수록, 해당 모델이 거의 모든 데이터셋에서 이긴다는 의미

> [!NOTE]
> <strong>스킬 점수</strong>
> 베이스라인 모델 대비 성능이 얼마나 개선되었는지 <strong>백분율로 나타내는</strong>지표
> 베이스라인 : Seasonal Naive
> 평가 모델과 베이스라인 모델의 예측 오차를 비교, 오차가 얼마나 감소했는지를 측정
> 
> $$
S = \left(1 - \frac{Loss_{\text{model}}}{Loss_{\text{base}}}\right) \times 100
> $$
> - $S$: 스킬 점수 (Skill Score) 
> - $Loss_{\text{model}}$: 평가 대상 모델의 SQL(Scaled Quantile Loss) 
> - $Loss_{\text{base}}$: Seasonal Naive 모델의 SQL  
>   **의미**
> - $S = 0$: Seasonal Naive와 성능이 동일함  
> - $S > 0$: 베이스라인보다 성능이 우수함  
> 	- e.g. : $S = 47.3$은 베이스라인 대비 예측 오차를 약 47% 감소시켰음을 의미
> - $S < 0$: 베이스라인보다 성능이 열등함  

> [!NOTE]
> **대체 전략 (Imputation Strategy)**
> 시계열 예측 벤치마크(fev-bench)에서 미래의 정보가 입력 데이터에 포함되어 성능이 왜곡되는 **데이터 누수(Data Leakage)** 문제를 해결하기 위한 데이터 처리 규약
> 
> **핵심 원리**
> 모든 변수를 '미래를 아는 정보'와 '모르는 정보'로 엄격히 분류하여, 알 수 없는 정보는 미래 시점에서 강제로 제거(마스킹)
> 
> **세부 방법**
> 1. **변수 분류:**
> - **알려진 공변량(Known Covariates):** 휴일, 프로모션 등 미래 값을 그대로 유지
> - **타겟 및 과거 전용 공변량(Past-only Covariates):** 미래 시점의 값을 모두 **결측치(Missing Values, *)**로 설정
> 1. **마스킹 및 대체:**
> - **이진 마스크():** 값이 관측된 지점은 1, 결측치나 예측 대상 지점은 0으로 표시하는 지표를 생성
> - **0으로 대체:** 마스크를 생성한 후, 결측치로 표시된 모든 미래 값은 모델 입력 시 0으로 대체
>
> **의미**
> * **누수 차단:** 모델이 미래의 타겟 정보를 미리 보고 예측하는 '커닝'을 원천적으로 방지
> * **공정한 비교:** 모든 모델이 동일하게 제한된 정보(마스킹된 데이터)만을 사용하여 예측하게 함으로써 진정한 제로샷 예측 성능을 측정


---