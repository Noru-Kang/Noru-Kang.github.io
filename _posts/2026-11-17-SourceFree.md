---
title: SourceFree Basic
date: 2026-11-16 16:13:45 +09:00
categories:
  - AI-ML-DL
  - etc.
tags:
  - Source-Free
math: true
---
<strong>원본 학습 데이터에 접근없이 모델을 새로운 도메인에 적응시키는 기법</strong>

## Source-Free learning

<strong>원본 학습 데이터에 접근할 수 없다는 제약아래, 이미 학습된 모델을 활용하는 연구 설정</strong>이다. 대표 연구분야는 Source-Free Domain Adaptation(SFDA)이다. 예시는 "UDA(기존 Domain Adaptation) VS SFDA(Source-Free Domain Adaptation)"의 콜아웃을  참고하길 바란다.

> **Source-Free VS Fine-Tuning**
>
> 먼저, 서로 대립되는 개념이 아니다.
> - <strong>Fine-Tuning</strong> : 이미 학습된 모델의 파라미터를 추가로 학습시키는 <strong>방법</strong>
> - <strong>SFDA</strong> : Source 데이터 없이 Target 환경에 모델을 적응시켜야 하는 <strong>문제 설정</strong>
>
> | 구분 | Source 데이터 | Target 데이터 | Target Label |
> |---|---|---|---|
> | Supervised Fine-Tuning | 불필요 | 있음 | 있음 |
> | UDA | 있음 | 있음 | 없음 |
> | SFDA (Unsupervised) | 없음 | 있음 | 없음 |
> | Source-Free Supervised Adaptation | 없음 | 있음 | 있음 |
>
> **핵심:** SFDA에서도 Fine-Tuning을 사용할 수 있지만, SFDA 상황은 보통 Target Label이 없는 경우가 많으므로, 다른 접근이 필요하다.
{: .prompt-info }

> **UDA(기존 Domain Adaptation) VS SFDA(Source-Free Domain Adaptation)**
> 1. <strong>UDA</strong>의 예시
>    <strong>Source 데이터와 Target 데이터 모두 활용하는 경우</strong>
>    - <strong>병원 A(Source)</strong> : ECG 데이터(X)와 질병 Label(y)이 존재
>    - <strong>병원 B(Target)</strong> : ECG 데이터(X)만 있음
>    - <strong>기존 모델</strong> : 병원 A에서 이미 학습된 모델
>
>    서로 다른 측정장비를 사용하기 때문에 데이터 분포가 다르다.
>    학습 과정은 아래와 같을 수 있다.
>    1. 기존 모델에 병원 A와 B의 ECG를 입력하여 각각의 Feature Distribution을 추출
>    2. 두 병원의 Feature Distribution 차이를 줄이도록 기존 모델을 추가 학습
>    3. 이때 병원 A의 실제 Label을 활용하여 질병 분류 능력이 손상되지 않도록 학습
>
>    **Source 데이터를 다시 사용하는 이유** : Target에 적응하면서 기존 분류 능력이 손상되는 것을 방지하기 위함이다.
>
> 2. <strong>SFDA</strong>의 예시
>    <strong>Source 데이터 없이 기존 모델과 Target 데이터만 활용하는 경우</strong>
>    - <strong>병원 A(Source)</strong> : 개인정보 보호로 ECG 데이터와 Label 제공 불가
>    - <strong>병원 B(Target)</strong> : ECG 데이터(X)만 있음
>    - <strong>기존 모델</strong> : 병원 A에서 학습된 모델만 사용 가능
>
>    Source 데이터가 없으므로 기존 분류 Loss를 직접 계산할 수 없다.
>    학습 과정은 아래와 같을 수 있다.
>    1. 기존 모델로 병원 B의 ECG에 대한 Pseudo-label 생성
>    2. Pseudo-label을 이용해 병원 B의 데이터로 Fine-Tuning
>    3. 기존 모델의 지식을 유지하면서 Target 환경에 적응
>
>    **핵심 차이** : UDA는 Source 데이터를 함께 활용할 수 있지만, SFDA는 Source 데이터 없이 기존 모델의 지식과 Target 데이터만으로 적응해야 한다.
{: .prompt-info }

### 1.1. 수학적 문제 설정

<strong>Source Domain</strong>

$$
D_S = \{(x_i^S, y_I^S)\}_{i=1}^{n_s}
$$

<strong>Target Domain</strong>

$$
D_T = \{x_j^T\}_{j=1}^{n_T}
$$

- $x_i^S$ : Soucre의 입력 데이터
- $y_i^S$ : Source의 정답 레이블
- $x_j^T$ : Target의 입력 데이터
- $n_s, n_T$ : 각각의 데이터 개수
- $D_s, D_T$ : Source 및 Target 데이터 셋

<strong>Soucre 모델</strong>

$$
\theta_S = A(D_S)
$$

- $A$ : 모델을 학습시키는 알고리즘
- $\theta_s$ : 학습이 완료된 Source 모델의 파라미터

<strong>SFDA 문제의 목적함수</strong>

$$
\min_{\substack{\theta_T}}L_{SFDA}(\theta_T;D_T, \theta_S)
$$

<strong>단, Adaptation 단계에서는</strong> $D_S$<strong>에 접근할 수 없다.</strong>

즉, Source 모델이 가진 지식과 레이블 없는 Target 데이터만으로 Target에서의 성능을 높이는 것이 목표이다.

| 학습 설정                                | Source 데이터        | Target 데이터 | Target 레이블 |
| ------------------------------------ | ----------------- | ---------- | ---------- |
| Supervised Domain Adaptation         | 사용                | 사용         | 사용         |
| Unsupervised Domain Adaptation (UDA) | 사용                | 사용         | 미사용        |
| Source-Free UDA (SFUDA)              | 미사용               | 사용         | 미사용        |
| Domain Generalization (DG)           | 학습 시 사용           | 학습 시 미사용   | 미사용        |
| Test-Time Adaptation (TTA)           | 적응 시 미사용인 경우가 일반적 | 추론 시 사용    | 미사용        |

Source-Free UDA설정과 TTA 설정은 겹치는 부분이 많지만 동일한 설정은 아니다.
- SFUDA : Target 데이터셋으로 Adaptation 문제
- TTA : 추론 단계에서 들어오는 데이터를 이용해 모델을 적응하는 추론 시점에서의 문제

### 1.2. 정답 없이 어떻게 적응하는가?

<strong>일반적으로 Target 레이블</strong>이 없기 때문에 일반적은 Cross-Entropy(CE) 지도학습을 그대로 사용할 수 없다. 그래서 <strong>Target 데이터에서 추가적인 학습 신호를 추출</strong>한다.

#### Entropy Minimization

모델이 Traget 데이터에 대해 불확실한 예측을 하고 있다면, 그 <strong>불확실성을 낮추도록 학습</strong>

$$
\mathcal L_{\mathrm{ent}} =-\frac1{n_T} \sum_{i=1}^{n_T} \sum_{k=1}^K p_{ik}\log p_{ik}
$$

- $K$: 클래스 개수
- $p_{ik}$: Target 데이터 $i$가 클래스 $k$에 속할 예측 확률

e.g. 고양이, 말, 개의 예측 확률이 두 모델에서 각각 A : (0.34, 0.33, 0.33) / B : (0.95, 0.03, 0.02) 라고 하면, 둘 다 고양이로 예측하지만 확신도는 B가 높다

<strong>확신이 높다는 것이 정답이라는 뜻은 아니다.</strong> 잘못된 예측을 더욱 확신하게 되는 문제가 발생할 수 있다.

#### Pseudo-Labeling

모델이 <strong>직접 예측한 클래스를 임시 정답</strong>으로 사용하는 방식

$$
\hat y_i=\arg\max_k p_\theta(k\mid x_i^T)
$$

$$
\mathcal L_{\mathrm{PL}}
=-\frac1{n_T}
\sum_{i=1}^{n_T}
\log p_\theta(\hat y_i\mid x_i^T)
$$

즉, 정답이 없는 데이터를 모델이 스스로 라벨링 하고, 그 결과를 활용하여 추가 학습을 진행한다. 그러나 <strong>초기 모델이 틀린 정답을 생성하면 그 오류가 누적</strong>될 수 있다. 이를 흔히 Confirmation Bias 문제라고 한다.

#### Feature Alignment / Clustering

<strong>동일한 클래스로 추정되는 Target Feature들이 가까운 위치에 모이도록 학습</strong>

e.g. Target의 개 사진과 개 스케치가 Feature Space에서 비슷한 표현을 갖도록 만드는 것

Target 내부의 군집 구조를 이용하지만, <strong>실제 군집이 의미론적 클래스와 일치한다는 보장은 없다.</strong>

### 1.3. 대표 논문  : SHOT (ICML 2020)

#### Do We Really Need to Access the Source Data? Source Hypothesis Transfer for Unsupervised Domain Adaptation

[논문링크](https://proceedings.mlr.press/v119/liang20a/liang20a.pdf)

<strong>Source 모델의 Classifier를 고정한 상태에서 Target에 맞게 Feature Extractor를 학습</strong>

$$
f_\theta(x)=h_W(g_\phi(x))
$$

- $g_\phi$: Feature Extractor. 입력 데이터에서 특징을 추출
- $h_W$: Classifier. 추출한 특징으로 클래스를 예측
- $\phi$: Feature Extractor의 파라미터
- $W$: Classifier의 파라미터

학습된 Classifier의 W를 고정하고, Feature Extractor의 파라미터 $\phi$를 Target데이터에 맞게 업데이트

예를 들어, Classifier가 Source에서 개와 고양이를 구분하는 기준을 이미 학습했다고 생각하면, Target에서 추출한 Feature가 그 기준에 맞는 위치로 이동하도록 Feature Extractor만 수정하면 된다. 이를 진행하기위해 SHOT는 <strong>Information Maximization과 Self-Supervised Pseudo-Labeling</strong>을 사용한다.

$$
\mathcal L_{\mathrm{IM}}
=
\underbrace{H(\hat Y\mid X_T)}_{\text{개별 예측의 불확실성}}
-
\underbrace{H(\hat Y)}_{\text{전체 예측의 다양성}}
$$

Information Maximization은 위의 식과 같고 첫 번째 항을 최소화 하면서 개별 예측이 확신을 가지도록 유도한다. 두 번째 항은 전체 데이터가 한두 개의 클래스에만 몰리는 현상을 억제한다.

$$
\boxed{
\mathcal L_{\mathrm{SHOT}}
=
\mathcal L_{\mathrm{ent}}
+
\mathcal L_{\mathrm{div}}
+
\beta\mathcal L_{\mathrm{PL}}
}
$$

$$
\mathcal L_{\mathrm{div}}
=\sum_{k=1}^K \bar p_k\log\bar p_k
$$

Feature 클러스터링 기반의 의사 라벨을 추가하며 위의 전체 목적함수를 생성한다. 
- $\bar p_k$은 Target 전체의 클래스 k 평균 예측 확률
- $\beta$는 의사 라벨 손실의 가중치이다.

다만 diversity 항이 클래스 비율이 비교적 균등하다는 가정과 관련이 있기때문에 심하게 불균형하다면 정규화가 오히려 성능을 저하시킬 수 있다.

### 1.4. 주요 연구 과제

<strong>새로운 환경에 적응하려는 업데이트가 원래 학습된 지식을 훼손할 수 있다는 점</strong>이 주요 연구과제이다.

| 연구 문제 | 설명 |
|---|---|
| Large Domain Gap | Source와 Target이 크게 다르면 초기 Pseudo-Label이 부정확함 |
| Label Shift | Target 클래스 비율이 Source와 다름 |
| Open-Set Adaptation | Target에 기존 모델이 학습하지 않은 클래스가 존재 |
| Continual Adaptation | Target Domain이 시간에 따라 지속적으로 변함 |
| Catastrophic Forgetting | 새로운 도메인에 적응하면서 기존 도메인의 성능이 감소 |
| Model Access | 전체 Weight 접근과 API 기반 Black-Box 접근은 서로 다른 제약 |

이러한 문제는 적응 후 Target 정확도가 상승했지만, 동시에 기존의 정확도가 대폭 감소한다면 기존 지식의 보존에 실패했다고 볼 수 있다.

다만 <strong>표준 SFDA는 Adaptation 단계에서 원본 데이터에 접근할 수 없으므로,</strong> 원본 평가는 연구자가 별도로 보관한 평가 데이터로 수행해야 한다.