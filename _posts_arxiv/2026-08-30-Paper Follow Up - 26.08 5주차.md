---
title: Paper Follow Up - 26.08 5주차
date: 2026-08-19 13:00:00 +0900
categories:
  - AI-ML-DL
  - papers-follow-up
tags:
  - total
  - BioSignal/Bio_Time_series
  - Audio/Speech
  - Interest
  - AI/ML/DL
math: true
---
# Paper Follow Up - 26.08 5주차

>
> * **HALO: A Heterogeneity-Aware Language-Aligned IMU Foundation Model for Open-Set Human Activity Recognition**
> * **TRACE-CRC: Trajectory-Adaptive Conformal Risk Control for Multi-Step Channel State Information Prediction**
{: .prompt-info }

## 1. 이번 주 주요 이벤트

### Trend: 전체 흐름

**"배포 시 어떤 종류의 shift와 불확실성이 발생하는가"를 모델 구조 안에 명시적으로 넣는 흐름**

1. **생체/센서 분야**에서는 sampling rate, channel 구성, sensor placement, recording device처럼 지금까지 전처리로 뭉개던 차이를 **관측 과정 자체의 heterogeneity**로 취급하는 연구가 눈에 띈다.
2. **시계열**에서는 **point prediction → structured uncertainty / latent dynamics adaptation**으로 이동하는 흐름이 보인다.
3. **World model과 neural decoding**에서는 Transformer를 단순한 범용 sequence mixer로 사용하는 대신 **동역학에 맞는 operator 구조**를 넣으려는 움직임이 나타난다.
4. **통계 ML**에서는 Gaussian/RBF kernel을 GP uncertainty의 기본값으로 사용하는 관행을 비판하면서, **지나친 analyticity가 과신과 수치적 ill-conditioning을 동시에 유발할 수 있다**는 주장이 나왔다.

### A: 생체신호 / 의료 시계열

* `HALO`: sampling rate, channel 구성, sensor placement 등의 **센서 heterogeneity를 단순 nuisance로 제거하는 대신 모델이 처리하거나 conditioning해야 하는 관측 조건으로 명시**하는 방향을 보여준다.
* `Graph-CMMC`: ECG의 12 lead를 독립 채널로 처리하지 않고 **lead 관계를 graph로 표현**하면서, signal/image의 **pseudo-multimodal SSL**을 수행한다.
* `A Two-Site Retrospective Study`: 매 시간 severity label을 직접 만드는 대신 **환자 outcome을 trajectory-level ranking signal로 사용하여 credit이 각 timestep에 비균일하게 배분될 수 있도록 하는 weak temporal supervision**이 흥미롭다.
* `rPPG`: endpoint MAE 하나만으로 source physiology가 보존되었다고 결론내리면 안 된다는 점을 보여주며, **property-specific recoverability**라는 평가 관점을 제시한다.

> **property-specific recoverability**
> **원래 신호가 가지고 있던 여러 종류의 정보 중 어떤 정보가 변환된 신호에서도 살아남아 있는지를 property별로 따로 평가하는 것**이다.
> 예를 들어 PPG → rPPG 변환이라면 HR 하나만 보는 것이 아니라 temporal correlation, spectral structure, recurrence, nonlinear dynamics 등이 각각 얼마나 보존되는지를 따로 확인한다.
{: .prompt-info }

### B: 오디오 / 음성

* biomedical audio에서는 **cross-dataset failure가 병리 signal보다 recording device나 dataset identity를 학습했기 때문일 수 있다**는 결과가 나왔다. `cough-TB` 연구가 이를 직접적으로 보여준다.
* `AVSR`에서는 모델을 다시 학습하지 않고 audio-only와 audio-visual prediction의 차이 및 attention 정보를 이용해 **token별 contrastive decoding strength를 조절하는 reliability-aware routing** 기법이 제안되었다.
* Audio LLM 평가에서는 transcript만으로 풀 수 있는 shortcut을 제거하기 위해 **음향 cue와 텍스트가 서로 다른 답을 지지하는 conflict sample**을 별도로 만드는 평가 설계가 등장했다.

> **audio-only와 audio-visual prediction의 차이를 이용해 token별 contrastive decoding strength를 조절한다는 것?**
> 단순히 **"소리가 시끄러우면 video의 가중치를 직접 키운다"**는 의미는 아니다.
> 같은 AVSR 모델에서
>
> * audio-only prediction
> * audio + video prediction
>   을 각각 구한 뒤, 두 prediction의 차이를 contrastive correction에 사용한다.
>   audio가 충분히 reliable하다고 판단되는 token에서는 correction을 약하게 하고, audio가 불안정하다고 판단되는 token에서는 **AV prediction과 audio-only prediction의 차이를 더 강하게 반영**한다.
>   즉 소음이 심한 순간에는 `"audio-only가 지지하지만 AV는 덜 지지하는 token"`을 더 강하게 억제하는 방식이다.
>   여기서 **token별 조절**은 다음 token을 생성할 때마다 contrastive strength를 새로 결정한다는 의미이고, attention dynamics와 prediction disagreement는 현재 audio prediction의 reliability를 판단하기 위한 단서로 사용된다.
{: .prompt-info }

### C: 관심 분야

* Conformal prediction의 단위를 개별 timestamp가 아니라 **forecast trajectory 전체**로 올리는 것이 가장 눈에 띈다. 각 미래 시점을 따로 보장하는 것이 아니라, **미래 예측 경로 전체에서 한 시점이라도 실패하는 사건을 직접 제어**하는 방식이다.
* model-based inference와 NN의 결합도 단순한 KalmanNet을 넘어 **EM iteration 자체를 learned/unrolled adaptation loop**로 바꾸는 방향으로 진행되었다.

  * 기존 learned Kalman smoother는 model mismatch에는 어느 정도 robust하지만, **training distribution에서 벗어난 새로운 system parameter variation에 대해 inference 중 직접 parameter를 다시 추정하는 mechanism이 부족하다.**
  * EM-KalmanNet은 fixed small number의 EM-like iteration을 NN layer처럼 unfold한다.
  * E-step에는 **parameter-aware RTSNet**을 사용한다. 현재 추정 중인 $F$ 또는 $H$도 network condition으로 넣어 `"현재 system parameter를 이렇게 추정하고 있을 때 state를 어떻게 smoothing해야 하는가?"`를 학습한다.
  * M-step에는 **M-Net**을 사용한다. 현재 parameter가 어느 방향으로 얼마나 틀렸는지를 empirical moments, residual, gradient-related statistics로 판단하여 parameter correction을 출력한다.
* time-series + VLM에서는 LLM/VLM을 inference loop에 직접 넣기보다 **training-time semantic/visual supervision을 주고 실제 forecasting model은 가볍게 유지**하는 설계가 등장했다.
* 통계 ML에서는 kernel smoothness가 uncertainty calibration에 미치는 영향을 다시 봐야 할 이유가 생겼다.

> **Conformal prediction**
> 모델의 prediction error를 calibration data에서 측정하고 이를 이용해 **prediction interval/set의 크기를 보정하는 방법**이다.
> 중요한 점은 `"이 sample의 실제 값이 90% 확률로 이 구간 안에 있다"`라는 개별 sample의 conditional probability를 직접 의미하는 것이 아니다.
> 기본 split conformal에서는 exchangeability 가정 아래 새로운 sample에 반복적으로 적용했을 때 **실제 값이 prediction set 안에 포함되는 비율이 목표 coverage 이상이 되도록 finite-sample marginal coverage를 보장**한다.
{: .prompt-info }

> **KalmanNet**
> **Kalman Filter의 predict-update 구조는 유지하되, analytical model이나 covariance가 부정확할 때 계산하기 어려운 Kalman gain을 NN이 학습하도록 만든 구조**이다.
> ![](/assets/img/posts/20260830_1709_PAPER/601d23b512fac593193de5a4855946e3.png)
> [Kalman Filter 참고 영상](https://www.youtube.com/watch?v=LioOvUZ1MiM)
>
> ##### Kalman Filter
>
> **1. Predict**: 이전 state와 dynamics model을 이용해 다음 state를 예측한다.
> 예: $\hat{x}_{t|t-1}=12$
> **2. Update**: 실제 sensor measurement와 prediction이 다를 때 Kalman gain $K_t$를 이용해 어느 정도 수정할지 결정한다.
>
> $$
> \hat{x}_{t|t} = \hat{x}_{t|t-1} + K_t \left(y_t-\hat{y}_{t|t-1}\right)
> $$
>
> 예를 들어 predicted state가 $12$, sensor measurement가 $13$, $K_t=0.7$이라면
>
> $$
> 12+0.7(13-12)=12.7
> $$
>
> 로 수정된다.
>
> ##### KalmanNet
>
> 실제 환경에서는
>
> * noise가 Gaussian이 아닐 수 있고
> * dynamics model이 정확하지 않을 수 있고
> * noise covariance를 정확히 모를 수 있다.
>   KalmanNet은 이런 상황에서도 **Kalman Filter의 model-based 구조는 유지하면서 Kalman gain 계산을 recurrent NN이 데이터로부터 학습**한다.
{: .prompt-info }

> **EM (Expectation-Maximization)**
> **hidden variable과 model parameter를 번갈아 추정하면서 서로를 반복적으로 개선하는 알고리즘**이다.
> **0. Initialization**
> 처음에 parameter $F^{(0)}$를 초기화한다.
> **1. E-step**
> 현재 $F$가 맞다고 가정하고 observation과 현재 parameter를 이용해 hidden state $x_1,\ldots,x_T$의 posterior 또는 필요한 statistics를 추정한다.
> **2. M-step**
> E-step에서 얻은 state statistics를 이용해 더 나은 $F$를 추정한다.
> **3. 반복**
> 새 $F$를 다시 E-step에 사용하면서 state와 parameter를 반복적으로 개선한다.
>
> ##### 이 논문에서 비교하는 classical EM-KF의 한계
>
> EM 자체가 linear-Gaussian model에만 한정되는 것은 아니다. 다만 **이 논문에서 다루는 closed-form EM-KF는 linear-Gaussian state-space model과 정확한 probabilistic model에 크게 의존**한다.
> 주요 한계는 다음과 같다.
>
> 1. model mismatch나 non-Gaussian noise에서 성능이 저하될 수 있음
> 2. 수렴을 위해 여러 번의 forward-backward pass가 필요함
> 3. 짧은 observation block에서는 parameter 추정이 어려움
> 4. 빠르게 변하는 system에서는 iterative inference latency가 부담될 수 있음
{: .prompt-info }

### D: 주요 AI / DL / ML / RL

* world model에서는 latent representation만 신경 쓰는 것을 넘어 **latent transition operator 자체의 inductive bias**를 연구 대상으로 분리하는 흐름이 보인다.

  * **latent transition operator 자체의 inductive bias**란 world model이 `"무엇을 representation에 기억할 것인가?"`뿐 아니라 **latent state가 다음 latent state로 어떤 구조에 따라 변화해야 하는지에 동역학적 제약을 넣는 것**이다.
  * 기존에는 `좋은 latent representation을 만들자`가 중심이었다면, 이제는 `그 latent가 시간에 따라 어떻게 움직일지도 완전히 generic한 함수에 맡기지 말고 dynamics에 맞는 구조를 넣자`는 방향이다.
* offline RL에서는 복잡한 diffusion actor 대신 **배포할 때 사용하지 않는 critic에 capacity를 집중하는** 반대 방향의 설계가 나타났다.

  * 실제 deployment에서 계속 실행되는 actor는 단순하게 유지하고, **training에서만 사용하는 critic을 더 크고 강하게 만들어 actor에게 더 좋은 value/gradient signal을 제공**하는 전략이다.
  * actor: 실제로 어떤 action을 할지 결정하는 policy model
  * critic: 해당 state-action이 얼마나 좋은지를 평가하는 value model
* agent 연구에서는 execution trace를 매번 직접 재사용하는 대신 **trace → persistent knowledge → executable skill**을 분리하는 memory architecture가 등장했다.

  * 과거 행동 기록을 매번 통째로 다시 읽는 대신, **execution trace에서 반복적으로 사용할 지식을 추출하고 그 지식을 다시 실행 가능한 skill로 정리**하는 3단계 구조이다.

### E: 참고할 분야

* neural decoding에서는 저차원 latent state를 단순 feature로 사용하지 않고 **어떤 operator를 실행할지 지정하는 instruction pointer/controller**로 해석하는 방법이 등장했다.
* 계산신경과학에서는 neuronal activity와 pathological spreading 사이의 nonlinear feedback만으로 **hysteresis와 multistability가 발생할 수 있음**을 보였다. 즉 neuronal activity와 pathological spreading이 서로 영향을 주는 nonlinear feedback 때문에 같은 외부 조건에서도 시스템이 여러 stable state를 가질 수 있고, 한 번 다른 regime으로 전환된 뒤에는 조건을 단순히 원래 수준으로 되돌리는 것만으로 즉시 복귀하지 않을 수 있다.

> **Hysteresis**
> **상태가 한 번 바뀐 뒤에는 parameter를 반대 방향으로 되돌려도 같은 threshold에서 바로 원래 상태로 복귀하지 않는 현상**이다.
> 즉 현재 state가 현재 parameter만이 아니라 **이전에 어떤 경로를 거쳐 왔는가**에도 의존한다.
{: .prompt-info }

> **Multistability**
> **같은 외부 조건과 parameter에서도 둘 이상의 stable state가 존재할 수 있는 현상**이다.
{: .prompt-info }

---

## 2. 정독할 논문 정리

이번 주는 pass.

---

## 3. 상세 논문 정리

### A: 생체신호 / 의료 / 임상 시계열

#### HALO — Heterogeneity-Aware Language-Aligned IMU Foundation Model

##### 3줄 요약

1. 여러 IMU 데이터셋을 같이 학습하려 하면 **sampling rate, channel 수, sensor 종류·위치, activity label이 모두 달라** 고정된 하나의 encoder/classifier를 만들기 어렵다.
2. HALO는 먼저 sampling/channel/placement heterogeneity를 처리하는 IMU encoder를 SSL로 학습하고, 이후 signal representation을 **자연어 activity representation과 정렬**한다.
3. 센서 구성이 달라져도 같은 encoder를 사용하고, 학습할 때 보지 못한 activity도 새로운 text label bank를 사용해 zero-shot으로 분류할 수 있도록 설계했다.

##### 요약

HALO는 기존 HAR 모델이 실제로는 특정 데이터셋의 sensor format에 강하게 묶여 있다는 문제에서 출발한다. 예를 들어 모든 데이터를 공통 50 Hz로 resampling하면 downsampling에서는 high-frequency 정보를 잃고, upsampling에서는 존재하지 않는 sample을 만들어낼 수 있다. 또한 고정된 channel 수를 가정하는 architecture는 서로 다른 sensor suite를 하나의 model로 처리하기 어렵다.

HALO는 **물리적인 시간 길이를 기준으로 signal을 자르고**, sampling rate가 달라도 **adaptive pooling을 통해 고정 크기 token**을 만들 수 있도록 설계한다. 각 channel에는 동일한 tokenizer/Transformer/fusion parameter를 공유해서 channel 수가 model architecture에 직접 의존하지 않게 한다. variable channel 수를 batch에서 처리하기 위해 zero-padding은 사용하지만, **binary mask를 모든 attention/pooling layer에 전달하여 padded channel이 실제 계산에는 참여하지 않도록 한다.**

문제는 모든 channel에 같은 network를 사용하면 `wrist accelerometer X`와 `waist accelerometer X`처럼 sensor의 의미 차이를 알기 어렵다는 것이다. 이를 해결하기 위해 각 channel의 sensor 종류, axis, placement를 **자연어 embedding(Sentence-BERT embedding)**으로 만들어 signal token에 conditioning한다.

Stage 1에서는 encoder를 Masked Autoencoding + Contrastive Learning으로 SSL pretraining한다. Stage 2에서는 activity label도 Sentence-BERT로 embedding해 IMU representation과 정렬한다. 이때 의미가 비슷한 activity label을 hard negative로 처리하지 않도록 text-text similarity를 이용해 **soft target**을 만든다.

최종 zero-shot inference에서는 dataset-specific classifier를 새로 학습하지 않는다. 각 fused patch representation을 candidate text label embedding과 cosine similarity로 비교하여 patch별 prediction을 만들고, 논문의 zero-shot 평가는 이 patch prediction을 majority voting하여 최종 activity를 결정한다.

##### 핵심 아이디어 및 개념

* **Sampling rate 문제를 왜 adaptive pooling으로 해결?**: `100 samples = 1 patch`처럼 sample 수를 기준으로 patch를 정의하면 100 Hz에서는 1초지만 20 Hz에서는 5초가 된다. 즉 **같은 token이 나타내는 물리적 시간이 달라진다.**
  HALO는 반대로 **몇 초라는 실제 시간 기준으로 segment를 정의**한다. 따라서 sampling rate에 따라 들어오는 sample 수는 달라도 같은 물리적 시간 범위를 표현한다.

```text
1초 IMU segment

100 Hz → 100 samples ┐
 50 Hz →  50 samples ├─ adaptive pooling → same-size token
 20 Hz →  20 samples ┘
```

실제 tokenizer는 temporal branch의 multi-scale 1D CNN + adaptive average pooling과 spectral branch의 FFT + MLP를 함께 사용하고, 두 출력을 결합해 고정 차원 token을 만든다.

이를 이용하면 모든 데이터를 인위적으로 같은 Hz로 resampling하지 않아도 된다.

→ **sample index가 아니라 physical time을 invariant unit으로 선택한 것**

* **Channel 수가 다른데 어떻게 동일 encoder를 사용?**: 데이터셋마다 channel 수가 다르기 때문에 일반적인 fixed-width model처럼 `input = [C, T]`에 강하게 의존하면 `C=3`인 smartphone IMU와 `C=45`인 full-body IMU를 동일한 architecture로 처리하기 어렵다.
  HALO는 **각 channel에 동일한 tokenizer, Transformer, fusion parameter를 공유**한다. 즉 channel마다 별도 network가 존재하는 것이 아니라 같은 network가 모든 channel에 적용된다.

```text
channel 1 ─┐
channel 2 ─┤
channel 3 ─┼─ shared tokenizer / shared encoder
...       ─┤
channel C ─┘
```

Transformer layer에서는 두 종류의 attention을 번갈아 적용한다.

```text
1. temporal self-attention
   한 channel 내부에서
   patch1 ↔ patch2 ↔ patch3 ...
   → 이 sensor가 시간에 따라 어떻게 변하는가?

2. cross-channel self-attention
   같은 patch 위치에서
   ch1 ↔ ch2 ↔ ch3 ...
   → 서로 다른 sensor가 같은 시점에서 어떻게 관계되는가?
```

이후 learnable query가 여러 channel token에 attention을 수행하여 variable channel dimension을 하나의 fixed-width representation으로 fusion한다. 따라서 처음 channel이 3개였든 45개였든 downstream representation의 크기는 동일하다.

batch에서 variable channel 수를 맞추기 위해 zero-padding은 사용하지만 **binary mask를 attention과 pooling에 함께 전달해 padded channel은 계산에 기여하지 않도록 한다.**

→ 핵심적으로 **`channel 수에 architecture를 맞추는 것`이 아니라 `variable-length channel collection을 shared parameter로 처리한 뒤 fixed-size representation으로 fusion하는 방식`**이다.

같은 weight를 모든 channel에 사용하기 때문에 특정 dataset의 `"7번째 channel"` 자체를 외우는 shortcut도 줄이는 implicit regularization 역할을 한다.

* **그럼 같은 network를 사용하면 wrist와 waist sensor를 어떻게 구별?**: 모든 channel이 동일한 weight를 사용하므로 waveform만 보면 `wrist accelerometer X`인지 `waist accelerometer X`인지 알기 어렵다. 같은 가속도 waveform이라도 sensor 위치에 따라 biomechanical meaning이 다를 수 있다.
  HALO는 각 channel에 대해 다음과 같은 sensor description을 만든다.

```text
"Accelerometer X-axis, waist-mounted smartphone"
"Gyroscope Z-axis, wrist-mounted smartwatch"
```

이를 frozen Sentence-BERT로 embedding하고 learned residual projection을 거쳐 sensor embedding $\mathbf{e}_c$를 만든다. 이후 원래 signal token에 더한다.

$$
\mathbf{H}'_{b,p,c}
=
\mathbf{H}_{0,b,p,c}
+
\gamma \mathbf{e}_c
$$

여기서 $\gamma$는 learnable scale이고 0.1로 초기화한다. 처음부터 language information이 signal feature를 강하게 덮어쓰지 않고, training하면서 필요할 때 sensor context의 영향력을 키울 수 있도록 한 것이다.

```text
signal feature
      +
"어디에서 어떤 sensor로 측정했는가"
      ↓
context-aware signal representation
```

즉 **channel processing rule은 공유하지만 channel의 의미는 metadata conditioning으로 구분**한다.

→ 핵심적으로 **heterogeneity를 무조건 없애는 것이 아니라, 모델이 알아야 하는 heterogeneity는 context로 명시적으로 알려준다.**

실제 ablation에서도 sensor conditioning(channel-text fusion)을 제거했을 때 zero-shot open-set accuracy가 가장 크게 감소하여 HALO의 핵심 component임을 보여준다.

* **왜 Masked AE + Contrastive Learning을 같이 사용?**: Stage 1에서는 activity label을 사용하지 않고 여러 IMU dataset을 SSL로 pretraining한다. 두 objective는 서로 다른 역할을 한다.
  **Masked Autoencoding**에서는 50%의 patch를 learned mask token으로 바꾸고 주변 context를 이용해 원래 normalized signal을 복원한다.

```text
walking signal

patch1 | patch2 | MASK | patch4 | patch5
                 ↓
          patch3 reconstruct
```

이를 잘하려면 model이 **local waveform, periodicity, temporal continuity**처럼 signal 내부의 세부 구조를 representation에 보존해야 한다.

반면 Contrastive Learning에서는 같은 patch에 jittering, scaling, time warping 등을 적용해 두 augmented view를 만들고 positive pair로 사용한다.

```text
original motion
 ├─ augmentation A → z1
 └─ augmentation B → z2

z1 ≈ z2
```

즉 noise, amplitude scaling, 일부 temporal distortion처럼 activity identity와 직접 관계없는 변화가 있어도 representation은 비슷하게 만들도록 한다.

```text
Masked AE
→ 중요한 signal detail은 버리지 마라

Contrastive Learning
→ 중요하지 않은 corruption에는 invariant해져라
```

따라서 한 objective만 사용할 때보다 **`local information preservation`과 `robust/discriminative representation learning`을 동시에 학습**하도록 만든 구조이다.

* **Activity label을 왜 일반 one-hot contrastive target으로 정렬하지 않고 soft target으로 정렬?**: 여러 데이터셋을 합치면 activity label vocabulary가 통일되어 있지 않다.

```text
dataset A: walking
dataset B: normal walk
dataset C: strolling
dataset D: ambulating
```

일반 CLIP-style contrastive learning에서는 정확히 matching되는 pair만 positive이고 나머지는 negative가 된다.

```text
IMU walking ↔ "walking"   : positive
IMU walking ↔ "strolling" : negative
```

하지만 `walking`, `strolling`, `ambulating`은 의미적으로 유사하다. 이들을 hard negative로 밀어내면 **같은 종류의 motion을 나타내는 text embedding끼리 서로 멀어지도록 하는 contradictory gradient**가 생긴다.

HALO는 Sentence-BERT의 text-text similarity를 이용해 one-hot target 대신 **semantic similarity 기반 soft target distribution**을 만든다.

```text
일반 hard target
walking   : 1.00
strolling : 0.00
running   : 0.00
sitting   : 0.00

설명용 soft target 예시
walking   : 0.70
strolling : 0.22
running   : 0.06
sitting   : 0.02
```

위 숫자는 설명용 예시이고, 핵심은 semantic하게 비슷한 label에 partial target mass를 주어 **완전한 negative로 취급하지 않는다는 것**이다.

Stage 2에서는 synonym augmentation도 함께 사용하며, 최종 inference는 classifier가 아니라 text label embedding과의 cosine similarity retrieval로 수행한다.

```text
IMU patch
   ↓
IMU embedding

candidate labels
   ↓
text embeddings

cosine similarity
   ↓
patch prediction
   ↓
majority vote
   ↓
final activity
```

따라서 새로운 dataset에서도 classifier head를 다시 학습하지 않고 **candidate text label bank를 교체하여 open-vocabulary prediction**을 수행할 수 있다.

---

### C: 관심 분야

#### TRACE-CRC

##### 3줄 요약

1. multi-step forecasting에서 각 future timestamp를 따로 90% coverage로 보장해도 **전체 future trajectory가 동시에 coverage될 확률이 90%라는 뜻은 아니다.**
2. TRACE-CRC는 `미래 중 한 시점이라도 uncertainty ball 밖에 존재하면 trajectory failure`라고 정의하고 이 failure risk를 직접 제어한다.
3. uncertainty radius를 **horizon difficulty × trajectory difficulty × global risk multiplier**로 분해하여 adaptive하게 만들면서도 trajectory-level risk를 별도의 LTT 단계에서 certify한다.

##### 요약

TRACE-CRC는 새로운 forecasting model보다 **예측 결과 주위의 uncertainty를 어떻게 calibration할 것인가**가 핵심이다.

pretrained CSI predictor가 미래 CSI matrix를 여러 timestep에 대해 예측한 뒤, 각 predicted CSI matrix를 중심으로 Frobenius-norm uncertainty ball을 만든다. 기존 horizon-wise conformal에서는 각 future horizon의 residual distribution을 따로 calibration할 수 있다.

하지만 실제 의사결정에서는 `20개 미래 시점 각각의 marginal coverage가 높다`는 것과 `20개 미래 frame이 모두 uncertainty region 안에 있다`는 것은 다른 문제이다. downstream beamforming이나 scheduling은 하나의 미래 frame만 크게 틀려도 영향을 받을 수 있기 때문에 논문은 **trajectory 전체에서 적어도 하나의 horizon이 실패하는 사건**을 risk로 정의한다.

다만 trajectory 전체를 안전하게 만들기 위해 모든 horizon에 큰 동일 radius를 적용하면 uncertainty set이 지나치게 커진다. TRACE-CRC는 이를 줄이기 위해 horizon별 error scale과 trajectory별 difficulty를 따로 추정한다.

최종 radius는

$$
r_{i,j}^{\star}
=
\lambda^{\star} q_{g_i} w_j
$$

로 구성된다.

* $w_j$: horizon $j$의 상대적인 error difficulty
* $q_{g_i}$: 현재 trajectory가 속한 difficulty group의 conformal scale
* $\lambda^\star$: 전체 trajectory risk를 certify하기 위한 global multiplier

마지막으로 별도의 validation subset에서 Learn-then-Test 방식으로 여러 $\lambda$ 후보를 검정하고, target trajectory failure risk를 만족한다고 certify할 수 있는 가장 작은 multiplier를 선택한다.

##### 핵심 아이디어 및 개념

* **왜 point coverage와 trajectory coverage가 다른가?**

trajectory $i$의 future horizon $j$에서 true CSI가 uncertainty ball 내부에 존재하는 사건을

$$
A_{i,j}
=
\left\{
E_{i,j}\le r_{i,j}
\right\}
$$

라고 하자.

여기서 $E_{i,j}$는 predicted CSI와 true CSI 사이의 Frobenius residual이다.

horizon-wise conformal은 각 $j$에 대해

$$
P(A_{i,j})
\ge
1-\alpha_h
$$

를 만족시키는 문제다.

하지만 trajectory 전체가 성공하려면

$$
\mathcal{C}_i^{\mathrm{traj}}
=
\bigcap_{j=1}^{N_f} A_{i,j}
$$

즉 **모든 horizon이 동시에 성공해야 한다.**

따라서 trajectory failure indicator는

$$
L_i^{\mathrm{traj}}
=
\mathbf{1}
\left\{
\exists j:
E_{i,j}>r_{i,j}
\right\}
$$

로 정의한다.

예를 들어 단순히 10개 horizon이 서로 독립이고 각각 coverage가 90%라고 가정하면 전체가 동시에 성공할 확률은

$$
0.9^{10}
\approx
0.35
$$

에 불과하다.

실제 time series의 horizon들은 독립이 아니므로 이 숫자를 그대로 적용할 수는 없지만, **marginal coverage와 simultaneous trajectory coverage가 완전히 다른 문제**라는 점을 보여주는 예시이다.

→ **보장 단위를 실제 사용하는 decision unit과 맞춰야 한다.**

* **Radius를 왜 세 항의 곱으로 설계?**

핵심 식은

$$
r_{i,j}^{\star}
=
\lambda^{\star} q_{g_i} w_j
$$

이다.

```text
w_j
→ horizon j 자체의 상대적인 error scale은 어느 정도인가?

q_{g_i}
→ 이 trajectory는 easy/hard group 중 어디에 속하며,
  그 group에서 필요한 conformal scale은 어느 정도인가?

lambda*
→ 위 adaptive rule 전체를 얼마나 늘려야
  trajectory-level risk를 통계적으로 certify할 수 있는가?
```

**$w_j$: horizon difficulty**

단순히 `"미래로 갈수록 무조건 radius를 키운다"`는 방식은 아니다. 실제 calibration residual의 horizon별 upper quantile을 이용해 error profile을 추정한다.

$$
w_j^{\mathrm{raw}}
=
Q_{1-\alpha_h}
\left(
\left\{
E_{i,j}
:
i\in\mathcal{D}_{\mathrm{prof}}
\right\}
\right)
$$

이후 neighboring horizon을 이용한 moving average로 profile을 smoothing하고, 지나치게 작은 값에 lower floor를 둔 뒤 평균이 1이 되도록 normalize하여 최종 $w_j$를 만든다.

```text
w_j > 1
→ 평균보다 어려운 horizon

w_j < 1
→ 평균보다 쉬운 horizon
```

따라서 horizon별로 실제 관찰된 error scale에 맞춰 radius의 상대적 크기를 분배한다.

* **$q_{g_i}$: trajectory difficulty는 어떻게 결정?**

같은 horizon이라도 모든 trajectory의 난이도가 같지는 않다. 어떤 predicted trajectory는 안정적이고 smooth하지만, 어떤 trajectory는 magnitude와 변화량이 크고 복잡할 수 있다.

문제는 test 시점에는 true future residual $E_{i,j}$를 알 수 없다는 것이다. 따라서 TRACE-CRC는 **predicted trajectory에서 계산할 수 있는 feature만 이용해 difficulty를 미리 예측**한다.

predicted CSI norm sequence에서 mean, std, range, slope, total variation, curvature 등의 feature $\mathbf{x}_i$를 만든다.

calibration의 profile subset에서는 true residual을 사용할 수 있으므로 실제 trajectory difficulty target을

$$
d_i
=
\max_j
\frac{E_{i,j}}{w_j}
$$

로 정의한다.

즉 각 horizon의 일반적인 difficulty $w_j$를 제거하고도 **어느 한 horizon에서 unusually large error가 발생했는가**를 보는 값이다.

그 다음 ridge regression으로

$$
\hat{d}_i
=
\beta_0
+
\boldsymbol{\beta}^{\top}\mathbf{x}_i
$$

를 학습한다.

이후 conformal/test trajectory에서는 true residual 없이 predicted trajectory feature만으로 $\hat d_i$를 계산하고, threshold를 기준으로

```text
low-difficulty group  → g_i = 0
high-difficulty group → g_i = 1
```

으로 나눈다.

각 group마다 calibration score

$$
d_\ell
=
\max_j
\frac{E_{\ell,j}}{w_j}
$$

의 conformal quantile을 따로 계산해 $q_0$, $q_1$을 얻는다.

→ 결국 $q_{g_i}$는 **현재 trajectory의 예상 난이도에 맞춰 전체 radius scale을 조절하는 항**이다.

* **$\lambda^\star$: 최종 global risk control**

$w_j$와 $q_{g_i}$를 사용하면 horizon과 trajectory difficulty에 맞춰 adaptive radius를 만들 수 있지만, 이것만으로 최종 trajectory-level CRC guarantee가 자동으로 생기는 것은 아니다.

그래서 별도의 validation subset에서 finite candidate grid $\Lambda$의 각 $\lambda$에 대해

$$
r_{i,j}(\lambda)
=
\lambda q_{g_i}w_j
$$

를 만들고 trajectory failure를 계산한다.

각 $\lambda$에 대해

$$
H_0(\lambda)
:
\mathcal{R}_{\mathrm{traj}}(\lambda)
>
\alpha
$$

를 one-sided Hoeffding-Bentkus test로 검사하고 multiple-testing correction을 거친다.

그중 target risk 이하라고 certify할 수 있는 candidate 집합에서 **가장 작은 $\lambda$**를

$$
\lambda^\star
$$

로 선택한다.

즉 $w_j$와 $q_g$는 **효율적으로 radius를 배분하는 단계**, $\lambda^\star$는 **그 adaptive rule 전체에 최종 statistical certificate를 붙이는 단계**라고 보면 된다.

* **왜 calibration data를 여러 subset으로 나누는가?**

$$
\mathcal{D}_{\mathrm{cal}}
=
\mathcal{D}_{\mathrm{prof}}
\;\dot{\cup}\;
\mathcal{D}_{\mathrm{cp}}
\;\dot{\cup}\;
\mathcal{D}_{\mathrm{val}}
$$

```text
D_prof
→ horizon difficulty w_j 추정
→ trajectory difficulty regression model 학습

D_cp
→ difficulty group threshold 설정
→ group-wise conformal quantile q_g 계산

D_val
→ lambda 후보의 trajectory risk를 검정
→ lambda* certification
```

같은 data에서 adaptive rule을 설계하고 바로 그 data로 risk guarantee까지 확인하면 selection bias 때문에 validity argument가 복잡해질 수 있다.

그래서 **profiling / conformal calibration / risk certification을 서로 다른 trajectory subset에서 수행**한다.

이 방법의 guarantee에서 중요한 가정은 frame별 independence가 아니다. **완전한 CSI trajectory 단위의 exchangeability**가 필요하며, 한 trajectory 내부의 frame들 사이에는 temporal dependence가 있어도 된다.
