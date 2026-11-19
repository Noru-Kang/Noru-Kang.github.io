---
title: SAE(Sparse AutoEncoder), Steering Vector
date: 2026-07-26 14:00:00 +0900
categories:
  - AI-ML-DL
  - Representation-Learning
tags:
  - sae
  - steering-vector
  - interpretability
math: true
---
# SAE(Sparse AutoEncoder), Steering Vector
- <strong>SAE(Sparse AutoEncoder)</strong> : 모델의 복잡하게 섞인 내부 표현을 sparse하고 <strong>해석 가능한 feature로 분해</strong>하는 방법
- <strong>Steering Vector</strong> : 모델의 내부 표현에 특정 방향의 벡터를 더하거나 빼서 모델의 <strong>판단이나 출력을 원하는 방향으로 이동</strong>시키는 방법
- <strong>SAE + Steering Vector</strong> (cf. 두 기술의 역할이 다르다. 즉, 서로 다른 기술이라 개별적으로 사용할 수 있다.)
	1. SAE : "어떤 개념이 어느 방향에 있는지" 찾고 → 표현을 분석하고 좌표계를 만드는 도구
	2. Steering Vector : 그 "방향을 강화/약화"하여 모델에 개입 → 표현을 실제로 움직이는 도구

## 0. 모델의 내부 표현
 딥러닝 모델은 입력을 바로 분류하지 않고, 여러 층을 지나며 벡터 표현으로 의미를 분해한다.
 ```text
 data d
   │
   ▼
Model
   │
   ▼
hidden representation h
 ```
 $h \in R^d$ 이고 h는 d차원의 벡터(내부 표현)이다.
 e.g. `d=512`이면
 ```
 d = [1, 4, 2, ..., ..., ..., 5]
     └─── 1024 dimensions ─────┘
   │
   ▼
Model
   │
   ▼
 
 h = [0.81, -0.14, 1.22, ..., 0.07]
     └────── 512 dimensions ──────┘
 ```
 이때 문제는 벡터의 각 원소의 의미가 명확하지 않다는 점이다.
 - h[0] = 위치 + 성별 + 전력선
 - h[1] = 나이 + bmi + 위치
 - h[2] = 나이 + 신호의 특징 + ...

## 1. Polysemanticity와 Superposition
### 1.1. Polysemanticity
 하나의 뉴런 혹은 활성화된 차원이 <strong>서로 다른 여러 의미에 반응</strong>하는 현상을 의미한다.

 예를 들어 어떤 뉴런이 다음 상황에서 모두 활성화 될 수 있다.
 e.g. 노드 173 : 병원 A에서 측정한 데이터인가?, 고령환자 인가?, 특정 질환을 가지고 있는가?, ...

### 1.2. Superposition
 모델이 제한된 차원 내에 더 많은 feature를 넣기 위해 여러 feature을 서로 다른 비직교 방향으로 겹쳐 저장하는 가설이다. 예를 들어 2차원밖에 없지만, feature의 개수가 3개이상이면
			 뉴런1, 뉴런2
 - feature A = [1.0, 0.0]
 - feature B = [0.7, 0.7]
 - feature C = [0.2, 1.0]
 - ...
처럼 한 뉴런은 여러 feature의 영향을 동시에 받게 된다.
SAE에서는 이러한 Superposition때문에 개별 뉴런보다 <strong>activation space의 방향</strong>을 분석해야한다고 본다. 다만 SAE feature이 완전히 원자적이고 유일한 "진짜 feature"라는 보장은 없다.

---
## 2. SAE(Sparse AutoEncoder)
### 2.1. 일반적인 오토인코더
오토인코더는 입력을 압축한 뒤 다시 복원하도록 학습하는 모델이다. 
![](/assets/img/posts/SAE/a5f6cff9ecfced512775bd1504ace7b2.png)
| https://lilianweng.github.io/posts/2018-08-12-vae/

$$ x' \simeq x$$
학습 목표이며, latent dimension을 작게 만들어 정보를 압축(<strong>작은 차원이라는 bottleneck을 활용</strong>)
한다.

### 2.2. Sparse Autoencoder
![](/assets/img/posts/SAE/55d1aeed13b9e93ef25b4498eba8ef7e.png)
| https://www.lesswrong.com/posts/8YnHuN55XJTDwGPMr/a-gentle-introduction-to-sparse-autoencoders

원래 dim보다 훨씬 큰 latent space(hidden layer)를 사용하되, 아주 일부만 활성화되도록 만든다.
```
hidden h ∈ R^512
        │
        ▼
SAE latent z ∈ R^4096 : z = [0, 0, 0, 2.1, 0, 0, ..., 0.7, ..., 0]
        │
        ▼
reconstruction ĥ ∈ R^512
```
예를들어 4096개의 원소중에서 32개만 켜질 수 있다.

즉, SAE에서 정보 bottleneck은 작은 차원이 아니라 latent를 <strong>sparsity</strong>, 즉 "동시에 활성화될 수 있는 feature 수가 적다"는 제약을 통해 bottleneck을 구현하여, dense한 x를 sparse한 dictionary feature들의 조합으로 표현한다. 
최근에는 GPT계열에 수백만 개 이상의 latent를 갖는 SAE도 학습되었고, TopK방식으로 sparsity를 직접 제어하는 방법등이 제안되고 있다.


---
## 3. 수학적 구조
### 3.1. Encoder
dense embedding을 SAE feature로 변환한다.

### 3.2. Decoder
SAE feature을 다시 원래 dense embedding의 차원으로 복원한다.
이때 식을 보면
$$x' = decoder\ bias + \sum^{latent space의 차원}_{j=1}z_jd_j$$
- $z_j$ : j번째 SAE feature가 얼마나 활성화 되었는지
- $d_j$ : j번 feature가 원래 embedding 차원에서 가르키는 방향, <strong>보통 SAE decoder direction또는 dictionary atom</strong>이라 부른다.


---
## 4. SAE는 어떻게 희소성을 만드는가?
### 4.1. ReLU + L1 SAE
기본적인 방법
$$z = ReLU(encoder\ output)$$
손실함수
$$L = \underbrace{\|h - \hat{h}\|_2^2}_{\text{reconstruction loss}} + \lambda \underbrace{\|z\|_1}_{\text{sparsity penalty}}$$
- <strong>reconstrucion loss</strong> : 원래 dense embedding을 잘 복원하도록
- <strong>sparsity penalty</strong> : L1 loss로 많은 feature가 0이 되도록 유도

이 방식은 문제가 있음, L1은 불필요한 feature만 제거하는게 아니라 활성화 되어야 할 feature의 크기도 줄이기 때문이다. e.g. L1 적용 전 : 3.0 → L1 적용 후 : 1.8로 복구
L1때문에 <strong>shrinkage bias</strong>발생

### 4.2. TopK SAE
<strong>매 sample마다 가장 큰 K개의 feature</strong>만 남김
$$z = TopK(encoder\ output, K)$$
```text
encoder output = [0.1, 2.4, 0.7, 3.1, 0.2]

TopK, K = 2

z = [0.0, 2.4, 0.0, 3.1, 0.0]
```
이 방법을 통해 활성화된 feature 수를 직접 통제할 수 있다.
- K : 0이 아닌 원소의 개수

TopK는 L1처럼 지속적으로 0으로 당기지 않으므로 shrinkage를 줄일 수 있다. 또한 평균적인 feature 개수를 직관적으로 설정할 수 있어 최근 연구에 널리 사용됨.

### 4.3. BatchTopK SAE
일반 TopK는 모든 sampel에 정확히 같은 개수의 feature을 킨다. 그러나 실제로 복잡한 sample의 경우는 더 많은 feature을 필요할 수 있음. 따라서 이를 강제하지 않고 BatchTopK는 <strong>batch 전체에서 평균 sparsity</strong>를 맞춘다.
```
Sample A → 7개 활성화
Sample B → 15개 활성화
Sample C → 3개 활성화

Batch 평균 → 약 8개
```
즉, sample의 복잡도에 따라 feature수를 유연하게 할당 가능하다.
| https://arxiv.org/abs/2412.06410

### 4.4. JupReLU SAE
각 feature에 threshold $\theta_j$를 둔다.
$$z_j = \begin{cases} {encoder\ output}_j, & {encoder\ output}_j > \theta_j \\ 0, & {encoder\ output}_j \le \theta_j \end{cases}$$
- <strong>ReLU</strong> : 0보다 크면 활성화
- <strong>JumpReLU</strong> : threshold보다 크면 활성화
작고 불확실한 활성화는 제거하고, threshold를 넘은 feature만 유지
| https://arxiv.org/abs/2407.14435


---
## 5. SAE feature의 해석
SAE를 학습했다고 해서 feature의미가 자동으로 붙는 것은 아니다.
```
Feature 317
Feature 928
Feature 1503
```
즉 번호만 생긴다. 연구자는 이러한 feature가 어떤 상황에서 활성화 되는지 분석해야 한다.

### 5.1. Top-activating examples
특정 feature의 activation이 가장 큰 sample을 찾음
```
Feature 317이 강하게 활성화된 PSG window

1위: 병원 B, 고령, 높은 50/60Hz noise
2위: 병원 B, 고령, 높은 50/60Hz noise
3위: 병원 B, 중년, 높은 50/60Hz noise
4위: 병원 C, 고령, 낮은 50/60Hz noise
```
Feature 317은 다음 후보중 하나이다.
- 병원 B
- 고령
- 전원 잡음

### 5.2. Feature-label association
각 SAE feature와 label관계를 계산한다.
$$score_j = AUROC(z_j, label)$$
혹은
- Mutual information
- Point-biserial correlation
- ANOVA / Kruskal–Wallis
- Linear probe coefficient
- Logistic regression
- TCAV
- Mean activation difference
- Conditional association
- Mixed-effects model 등
label을 잘 예측한다는 사실이 label을 잘 표현한다고 결론 내리면 안된다.
이는 특정 label이 예를들어 고령환자가 많고, 특정 장비를 사용하고, 전처리 기법이 다를 수 있기때문이다. 즉 이는 실제로 나이나 질병을 나타낼 수 있다.

### 5.3. Causal intervention
해당 feature을 직접 줄이거나 제거한 후 모델 결과가 어떻게 변화하는지 확인하는 방법
```
Feature 317 원래 activation
z_317 = 2.8

Feature attenuation
z*_317 = 0.5 × 2.8 = 1.4
```
그 후 우리가 찾고자 하는 label(혹은 feature)의 성능을 확인한다.
- 해당 feature 분류 성능은 하락
- 그 feature을 제외한 다른 분류 성능은 비슷하게 유지 또는 상승/하강(단 너무 커서는 안된다.)

최근 SAE평가 연구도 reconstruction과 sparsity만으로 품질을 판단하기 어려워, 실제 downstream에 기반한 평가를 제안한다.


---
## 6. 좋은 SAE
### 6.1. Reconstuction fidelity(복원 충실도)
희소 분해 후 얼마나 잘 복원하는지
$$MSE=E[∥h−h^∥22]$$
혹은
- Explained variance
- Cosine Similarity
- Downstream loss recovred
- 원 classifier 성능 대비 reconstructed activation 성능

e.g.
```
Original h → classifier AUROC = 0.82
SAE ĥ      → classifier AUROC = 0.74
```
이는 SAE가 상당한 성능을 잃었다는걸 의미한다.

### 6.2. Sparsity
한 sample당 활성화된 feature 수
- K가 작음 : 너무 sparse하면 중요한 정보가 손실될 수 있음 → 해석이 쉬우나, 복원이 불량
- K가 큼 : 너무 dense하면 feature가 다시 섞일 수 있다. → 복원이 쉬우나, feature분해가 불명확
```
Reconstruction 
          Fidelity
             ▲
             │       Dense SAE (PCA/Identity에 가까워짐)
             │      /  [해석 불가능, 복원만 잘됨]
             │     /
             │    /   ★ Optimal Zone (Pareto Frontier)
             │   /     [해석 가능하면서 필수 정보 보존]
             │  /
             │ /     Sparse SAE (K가 너무 작음)
             │/       [해석은 쉬우나 중요 정보 손실]
             └──────────────────────────────► Sparsity (L0 감소)
```
#### L$_0$
입력 샘플 하나가 들어왔을 때 "몇 개의 피처만 커지는가(Non-zero)"를 나타낸다.
- Overcomplete Dictonary : 고차원 공간에 superposition이던 개념들 1개당 1개의 피처로 깔끔히 쪼개기 위해 엄청나게 키운다
- Dense Matrix의 위험 : $L_0$가 너무 크면(8192개중에서 1000개가 동시에 켜짐) 하나의 피처가 단일 개념을 뜻하는 게 아니라 <strong>여러 개념이 다시 뭉쳐서 켜지는 현상 발생</strong>

### 6.3. Dead Feature
학습 중 거의 한 번도 활성화 되지 않은 latent로, SAE dimension을 크게 만들었더라도 상당수가 dead라면 실제 dictionary capacity를 제대로 사용하지 못하는 것이다.
이는 SAE를 학습시킬때 L1이나 TopK의 제약이 너무 강한경우, 학습 초기 이후 단 한 번도 활성화되지 않을 수 있다는 의미이다.
- 이를 해결하기 위해 앤트로픽이나 OpenAI에서 학습 중간에 활성화 빈도가 극도로 낮아진 Feature을 감지하여, 현재 SAE가 잘 복원하지 못하는 잔차 방향으로 가중치를 강제로 재초기화 해주는 기법을 사용한다.

### 6.4. Interpretability와 Selectiviy
1. 사람이 feature의 의미를 설명할 수 있는가? → 사람이 쉽게 라벨링 할 수 있는가?
2. 모델이 실제 계산에서 그 feature을 사용하는가? → <strong>그 피처를 끄거나 키웠을때(Steering)</strong> 실제 출력에 아무 영향이 없다면?
라는 두가지 질문을 분리해서 생각해야 한다. 사람이 보기에는 특정 feature처럼 보여도 모델의 downstream task에서는 중요하지 않을 수 있다. 반대로 사람이 쉽게 설명하지 못하는 feature가 모델 계산에는 중요할 수 있다.
따라서 좋은 평가는
```
Reconstruction
+ Sparsity
+ Feature interpretability
+ Causal effect
+ Off-target preservation
+ Seed/dataset stability
```
를 포함해야 한다.


---
## 7. SAE의 한계
### 7.1. Feature Splitting
하나의 실제 개념이 여러 SAE Feature로 나뉠 수 있다. 이는 SAE의 Dictionary 크기(Hidden Dimension)를 크게 설정할수록, SAE는 단순한 개념 하나만 표현하기 보다 "<strong>상황별로 세분화된 조합</strong>"을 <strong>각각의 피처로 만드는 것이 Loss를 줄이는데 훨씬 유리하다고 판단한다.</strong> → 즉, <strong>단어를 형태소단위까지(의미 없는)</strong> 분해하는 현상이 발생한다.
```
실제 개념: 병원 B

SAE feature 31  = 병원 B의 EEG
SAE feature 82  = 병원 B의 RESP
SAE feature 143 = 병원 B의 고령 환자
```
- 📌 대응 : 단일 피처를 끄는 대신, 관련 피처들의 클러스터나 하위 공간 전체를 찾아내어 동시에 제어하는 <strong>Feature Clustering</strong>기법들이 연구되고 있다.

### 7.2. Feature absorption
```
Feature 31 = 병원 B feature처럼 보임

병원 B sample 1 → 활성화
병원 B sample 2 → 활성화
병원 B sample 3 → 비활성화
```
즉, concept feature가 존재하지만 일부 상황에서는 활성화되지 않는 현상이 발생할 수 있다.
이는 피처가 명확히 존재함에도 불구하고, 특정 조건이나 맥락에서 피처 B가 켜지지 않고 <strong>더 구체적인 다른 피처 A가 피처 B의 역할까지 흡수</strong>해버리는 현상이다.
e.g. 병원 B, Feature B : h[31] → 특수 상황(병원 B + 고령 + 노이즈 데이터) : h[317], h[0]은 비활성화
이는 L1, L0때문에 SAE는 피처 2개를 동시에 키우는것보다 복합 피처 하나를 키우는것을 선호하게 되는 단점이다. 불가피하다.

### 7.3. SAE dictionary는 유일하지 않음
같은 데이터로 학습해도
- Seed
- SAE크기
- sparsity
- train data
- optimziation
- layer 등
다양한 요소들에의해 유일성을 잃는다.



---
---
## 8. Steering Vector
모델이 데이터를 처리할 때, 내부에는 수많은 개념들이 <strong>activation space(고차원 지도)</strong>위의 특정 좌표에 존재한다.
steering vector는 모델 내부 activation space의 특정 방향인데, 이 벡터를 activation에 더하거나 빼서 <strong>모델 내부 상태를 이동시킨다.</strong> 즉, <strong>모델이 데이터를 처리하는 중간 과정을 직접 수정하여 원하는 출력을 유도하는 기법</strong>이다.

보통 모델의 성능이나 행동을 바꾸려면 파라미터 자체를 수정하는 
- fine tunning이 필요하지만, 이는 비용이 크고 원하지 않는 부작용을 일으킬 수 있다. 
- 반면 steering vector는 <strong>추론 과정에서 잠재 공간의 위치만 살짝 밀어주는 방식</strong>이다.

transformer계열에 사용했을 때
$$h_{\ell,t}^* = h_{\ell,t} + \alpha v_{\ell}$$
- $h_{l, t}$ : Transformer의 $l$번째 레이어, $t$번째 위치에 생성된 기본 hidden state vector
- $v_{l}$ : <strong>steering vector</strong>로 특정 개념, 속성, 또는 패턴(e.g. '긍정적 어조', '특정 노이즈 성분', '특정 편향' 등)을 나타내는 <strong>기하학적 방향 벡터</strong>
- $\alpha$ : <strong>steering strength</strong>로 조작의 강도를 조절하는 스칼라 값으로
	- $\alpha > 0$ : 해당 특징을 강제 주입 혹은 강화
	- $\alpha < 0$ : 해당 특징을 억제 혹은 제거

```
[입력 데이터] 
       │
       ▼
[이전 Layer들] ───► Original Activation (h)
                          │
                          │ + α * v (Steering Vector 추가)
                          ▼
                   Steered Activation (h*)
                          │
                          ▼
                  [다음 Layer들] ───► [변화된 최종 출력]
```

---
## 9. Steering Vector의 직관
hidden representation을 여러 특성을 조절하는 믹서라고 생각할때
```
┌─────────────────────────────┐
│ 내부 표현 믹서                  │
│                             │
│ 사실성       ███████░         │
│ 긍정성       ███░░░░░         │
│ 공격성       ██░░░░░░         │
│ 거절 성향    ██████░░          │
└─────────────────────────────┘
```
Steering vector는 그 믹서를 조절하는것을 의미한다.

---
## 10. Steering Vector를 추출하는 주요 기법
### 10.1 Contrastive Mean Difference : 대조 평균 차이
가장 직관적이고 널리 쓰이는 기본 기법으로, 원하는 속석을 가진 그룹(Postivie)과 그렇지 않은 그룹(Negative)의 내부 표현(Activation) 평균을 내어 그 <strong>차이(Difference) 벡터</strong>를 구한다.(대조 데이터의 차이 벡터들)
$$v = \mathbb{E}[h \mid y=1] - \mathbb{E}[h \mid y=0]$$
paired 샘플(동일 조건에서 단 하나의 변수만 다른 데이터 쌍)이라면, 개별 차이의 평균을 사용
$$v = \frac{1}{N} \sum_{i=1}^{N} (h_i^+ - h_i^-)$$
```
Positive examples
      │
      ▼
Frozen model
      │
      ▼
h⁺₁, h⁺₂, ..., h⁺ₙ
      │
      └────────────┐
                   │
                   ▼
             mean(h⁺)
                   │
                   ▼
              steering v : mean(h⁺) - mean(h⁻)
                   ▲
                   │
             mean(h⁻)
                   ▲
                   │
      ┌────────────┘
      │
h⁻₁, h⁻₂, ..., h⁻ₙ
      ▲
      │
Frozen model
      ▲
      │
Negative examples
```
- 장점 : 계산이 극도로 간단하며 빠른 연산이 가능
- 단점 : "평균"을 내는 방식이기 때문에 데이터 내부의 복잡한 비선형 구조나 고차원 변동성을 완전히 담아내지 못할 수 있음.

### 10.2 Linear Probe Direction
활성 공간(h)에서 특정 개념 혹은 라벨(y)을 구분하는 선형 분류기(Linear CLF.)를 먼저 학습시킨 뒤, 그 분류기의 가중치 벡터(w)를 Steering Direction으로 사용하는 방식

$$p(y=1 \mid h) = \sigma(w^\top h + b) \quad \implies \quad v = \frac{w}{\Vert{}w\Vert{}_2}$$
분류 평면의 법선 벡터 w를 $L_2$를 사용하여 방향만 추출해 이용
- 단점 : Linear Probe가 분류를 매우 잘 해내더라도, 가중치 w가 오직 "순수한 개념"만을 가르킨다고 보장할 수 없다, probe는 label을 "가장 잘 예측 하는 상관 방향"을 찾을 뿐이다. 즉 교란변수들이 혼합되어 있을 수 있다.
```text
w_site
   ├── 장비 차이
   ├── 연령 차이
   ├── 질환 비율 차이
   └── preprocessing 차이
```

### 10.3 PCA, Representation Engineering(Mean Diff.의 한계 극복)
Mean Difference의 한게를 극복하기 위해, pair 데이터의 차이 벡터들($\Delta h_i$)을 모아 <strong>PCA</strong>를 수행하고, 가장 설명력이 높은 첫 번째 주성분을 추출하는 방식
$$\Delta h_i = h_i^+ - h_i^- \quad \implies \quad v = \text{PC}_1(\{\Delta h_i\}_{i=1}^N)$$
- 장점 : 개별 샘플 간 차이의 "평균"만 보는게 아니라, 여로 대조 샘플들이 <strong>활성 공간상에서 공통적으로 움직이는 대표 분산 방향</strong>을 포착한다.
<strong>Representation Engineering</strong> : 단일 뉴런 단위가 아니라 전체 Activation 분포의 행동 패턴을 관찰하고 개입함으로서, 모델의 고수준 속성을 정밀하게 통제할 수 있음 | https://arxiv.org/abs/2310.01405

### 10.4 Optimization-Based Vector
pair 데이터를 이용해 간접적으로 구하는 것이 아니라, 목적 함수를 직접 최소화하도록 <strong>Steering Vector</strong> $v$ 자체를 경사하강법으로 직접 학습하는 기법
e.g. 예시 목적 함수
$$v^* = \arg\min_v \mathcal{L}_{\text{target}}(f(h + \alpha v)) + \lambda \Vert{}v\Vert{}_2^2$$
- 장점 : target에 직접 최적화
- 단점 : 데이터 leakage 위험, overfitting, 해석 어려움, shortcut 학습, off-target representation 손상
- 

### 10.5 SAE feature direction
SAE decoder의 weight 벡터($d_j$)를 그대로 Steering Vector로 활용하는 기법
$$v = d_j \quad \implies \quad h^* = h \pm \alpha d_j$$
SAE는 모델의 Activation 공간을 개별 개념들의 선형 결합으로 분해하게 되는데, 이때 특정 Feature $j$에 해당하는 Decoder 가중치 $d_j$는 <strong>Activation 공간상에서 그 개념이 추가되는 정확한 방향</strong>을 의미한다.
따라서 사전에 대조군 데이터셋을 만들거나 분류기를 학습시킬 필요 없이, <strong>이미 훈련된 SAE에서 원하는 피처와 벡터만 꺼내어 더하거나 빼주는 것만으로 모델 제어가 완료</strong>된다. SAE와 Steering vector가 연동되는 표준형태이다.

---
## 11. Steering Vector의 연산
### 11.1 고정 벡터 더하기/빼기
가장 단순하고 직관적인 방법, 모든 샘플의 Activation h에 똑같은 크기와 방향 벡터 ${\alpha}v$를 일률적으로 더하거나 뺀다.
$$h^* = h \pm \alpha v$$
특정 feature 강화/약화 가능하며 모든 샘플에 대해 동일하게 적용한다.
```text
Sample A: h_A → h_A - αv
Sample B: h_B → h_B - αv
Sample C: h_C → h_C - αv
```
<strong>핵심 한계점</strong>
- <strong>모든 샘플에 동등한 개입</strong> : 해당 속성이 원래 강하게 들어있는 샘플 A, 이미 깨끗해서 해당 속성이 거의 없는 샘플 B나 <strong>똑같이</strong> $-{\alpha}v$<strong>만큼 이동</strong>시킨다.
```
[샘플 A (특징 강함)] ──► h_A ──(- αv)──► h_A* (정상범위로 이동)
[샘플 B (특징 약함)] ──► h_B ──(- αv)──► h_B* (정상범위를 지나쳐 이탈)
```
- <strong>Over-correction(과도한 굴절)</strong> : 속성이 원래 없던 샘플 B는 오히려 자연스러운 Activation 분포 영역 밖으로 밀려나 모델의 기본 기능이나 표현력이 붕괴될 수 있음

### 11.2 Projection 제거
고정 벡터를 더하거나 빼는 방식의 한계를 극복하기 위해, "각 샘플이 실제로 가지고 있는 $v$방향의 성분 양에 비례해서만 제거하자"는 아이디어
$$c = (h - \mu)^\top v$$
단위 벡터  $v$ ($\Vert{}v\Vert{}_2 = 1$)와 기준 중심값 $\mu$에 대해 현재 Activation h가 $v$방향으로 가지는 내적 성분(크기) c를 구한다.
$$h^* = h - \alpha \cdot \underbrace{\left((h - \mu)^\top v\right) v}_{\text{샘플이 실제로 가지는 } v \text{ 성분}}$$
이 크기 c에 비례하여 $v$방향으로 $h$를 깎아낸다.
- $\alpha = 0$ : 전혀 손대지 않음 ($h^* = h$)
- $\alpha = 0.5$ : 해당 방향 성분을 반으로 줄임
- $\alpha = 1$ : 해당 방향 성분을 완전히 직교 상태로 깎아내어 0으로 만듦
```text
             h
            /|
           / |
          /  |  제거할 component
         /   |
        /    ▼
───────●────────────▶ v
      μ
```
고정 subtraction과의 차이
- 고정 : 모든 샘플에 대해 동일한 변화
- Projection : 각 sample이 실제로 가진 v component에 비례하여 변화
이를 subspace단위로 확장을 하게 되면(제거하고 싶은 잡음/편향 속성이 1개가 아니라 여러 개 일때) 직교 기저(Orthonormal Basis) V를 활용하여 <strong>하위 공간 전체를 투영 제거</strong>할 수 있다.
$$h^* = h - \alpha V V^\top (h - \mu)$$
- 장점 : 해당 속성이 없는 샘플은 $c \approx 0$이 되어 거의 바뀌지 않는다. 즉, <strong>원래 노이즈가 있던 샘플만 핀셋으로 골라내어 깎아내므로 안전</strong>하다.

### 11.3 Conditional Steering
분류기(Gate Function or Score)를 두어, <strong>해당 속성이 일정 수준 이상 감지된 경우에만 활성화하여 Steering을 적용하는 방식</strong>
특정 속성의 유무나 강도를 측정하는 스코어 함수 $g(h)$ e.g. sigmoid를 사용한다.$$g(h) = \sigma(w^\top h + b) \quad \in [0, 1]$$
이 스코어를 개입하는 강도에 직접 곱해준다.
$$h^* = h - \alpha \cdot g(h) \cdot v$$
```text
                                  ┌── g(h) 낮음 (0에 가까움) ──► h* ≈ h (개입 안 함)
[입력 h] ──► Concept Score g(h) ──┤
                                  └── g(h) 높음 (1에 가까움) ──► h* = h - αv (개입)
```
- 장점 : projection과 비슷하게 필요할때만 가동되지만, 스코어 함수에 <strong>비선형성</strong>을 추가할 수 있어, <strong>불필요한 개입을 더욱 엄격하게 통제</strong>할 수 있다.

---
## 12. Steering Vector $\alpha$의 의미
$\alpha$는 Steering Vector의 방향으로 얼마나 강하게 밀어붙일 것인가를 결정하는 <strong>거리 계수(Magnitude)</strong>이다.
$$h^* = h + \alpha v$$
즉

- $\alpha$ 가 너무 작음 : 효과가 거의 없음, 모델의 작동 방식이나 출력 행동에 아무런 변화가 나타나지 않는다.
- $\alpha$가 적절함 : target의 행동만 변화
- $\alpha$가 너무 큼 : representation 붕괴, off-target 성능 저하, 비정상적 출력
이다. activation space는 완전히 선형적이지 않기 때문에 한 방향으로<strong> 너무 멀리 이동하면 학습 데이터가 존재하지 않던 영역으로 나갈 수 있다.</strong>
```
training manifold

     ● ● ● ●
   ● ● ● ● ●
     ● ● ●
         │
         │ α가 지나치게 큼
         ▼
                      × h*
```
> [!info]
> <strong>Activation Space의 비선형성</strong>
> 수식상으로는 단순히 선형모델이지만, <strong>Transformer</strong>의 내부공간은 완전한 선형 공간이 아니라 Non Linear Manifold이다.
> - 특정 방향 v로 <strong>조금 움직일</strong>때는 선형적 관계가 유지되어 개념 제어가 잘 작동
> - 일정 거리를 넘어서면 <strong>상위 레이어의 비선형 활성화 함수와 Layer Normalization을 거치면서 변화량이 폭발적으로 왜곡</strong>
> → 즉 선형으로 조금밀었더니, 상위 레이어로 올라갈수록 결과가 비선형적으로 거대하게 일그러지는 현상이 발생

따라서
- <strong>sample 가변성</strong> : 동일한 $\alpha$ = 2.0을 적용하더라도 샘플 A는 깔끔하게 제거되지만, 샘플 B는 표현 공간 밖으로 틩겨 나가 붕괴 될 수 있다.
- <strong>ood Robustness의 취약성</strong> : 입력 데이터의 도메인이나 프롬프트 구조가 약간만 달라져도, 기껏 세팅해둔 steering Vector v와 $\alpha$의 조합이 작동하지 않거나 모델을 파괴할 수 있다.

---
## 13. Steering Vector와 Fine tunning
```
Fine-tuning
입력 → 모델 parameter 자체를 업데이트
            θ → θ*

Steering
입력 → 기존 parameter θ 유지
     → inference 중 activation만 변경
```
### 13.1 FineTunning
모델의 "뇌 구조" 자체를 재구성 : 모델의 가중치/파라미터 $\theta$를 수정
$$\theta^* = \theta - \eta \nabla_\theta \mathcal{L}$$
- 장점
	- <strong>복잡한 Adapation</strong> : 모델 전체 층의 파라미터가 유기적으로 업데이트되므로, 고난도의 새로운 데이터 패턴이나 태스크를 익히는 데 가장 강력
	- 여처 층의 표현을 함께 변경
	- 강한 성능 개선
- 단점
	- <strong>Catastrophic Forgetting</strong> : 새로운 데이터에 최적화되는 과정에서 기존에 가지고 있던 일반적인 지식이나 성능이 손상될 위험이 크다.
	- <strong>비용</strong> : 역전파 과정이 필수적이다.
	- 파라미터 저장필요(되돌리는데 check point가 필요)
	
### 13.2 Steering Vector
추론 시점에 외부에서 주는 "개입"
- 장점
	- <strong>극도로 가벼운 연산</strong> : 역전파 과정 없이, 대조군 입려의 차이나 SAE 피처 벡터를 단순 가산하는 것만으로 완성 가능
	- <strong>실시간 제어</strong> : 추론 시점에 계수만 조절하여 개입의 강도를 자유자재로 켜고 끌 수 있음.
	- 파라미터를 변경하지 않음
- 단점
	- <strong>단일 벡터의 한계</strong> : 복잡하고 정교한 연산 체계를 단순 선형 벡터 하나로 전부 지배하기는 어려움
	- <strong>Entanglement & Breakdown</strong> : 앞서 다루었듯 $\alpha$가 너무 커지면 Data manifold가 무너지는 위험이 존재한다. 
	- hidden activation에 접근 필요
	- layer마다 vector가 다름


---
---
## 14. SAE와 Steering Vector의 관계
$$h \approx b_{\text{dec}} + \sum_{j=1}^{m} z_j d_j$$
SAE는 고차원의 Dense Activation h를 사람이 이해할 수 있는 <strong>Spasre Feature</strong> *$z_j$들과 <strong>그 방향 벡터</strong> $d_j$들의 <strong>선형 결합</strong>으로 분해한다.
- $z_j$ (Feature Activation): 현재 샘플에서 $j$번째 특징(예: 전원 잡음, 환자 연령, 특정 파형 등)이 얼마나 세게 켜졌는지를 나타내는 <strong>크기(Magnitude, 스칼라)</strong> → 세기
- $d_j$ (Decoder Direction): Activation 공간상에서 그 특징이 가리키는 <strong>방향(Direction, 벡터)</strong> → 방향 그 자체
$d_j$ <strong>자체가 바로 Steering Vector</strong> 역할을 하게 되며, $z_j$를 조작함으로써 모델 내부 상태를 정밀하게 조절 할 수 있음
```text
[Dense Activation h] ──► SAE Encoder ──► Sparse Features z (병리학, 연령, 전원잡음...)
                                                  │
                                                  ▼ (원하는 Feature만 조절 z*)
                                                  │
[Steered Activation h*] ◄── SAE Decoder ◄─────── ┘
```

---
## 15. SAE 기반 Steering 방법 1 : Decoder direction 직접 더하기
SAE로 인코딩-디코딩 과정을 전체 다 거치지 않고, 사전(Dictionary)에 학습되어 있는 <strong>Decoder Weight 벡터</strong>$d_j$<strong>를 원래 h에 직접 더하거나 빼는 방식</strong>
- 강화 : $h^* = h + \alpha{d_j}$
- 약화 : $h^* = h - \alpha{d_j}$
##### 장점
- <strong>극도로 간결함</strong> : SAE의 Encoder/Decoder 전체 연산을 통과할 필요 없이 단순 벡터 가산만 수행한다.
- <strong>원래 정보 완벽 보존</strong> : SAE가 h를 복원할 때 발생하는 <strong>Reconstruction Loss(복원 손실)의 영향을 전혀 받지 않으며</strong>, 나머지 원본 신호 정보가 100% 보존된다.
##### 단점
- <strong>현재 샘플 상태 무시</strong> : 해당 샘플에서 Feature j가 원래 켜져 있지 않았던 샘플이라도 고정된 크기만큼 억제/강화됨
- <strong>비직교성 문제</strong> : SAE의 Feature 방향 벡터들($d_j$)는 서로 완벽히 직교하지 않는다. 따라서 $d_j$를 뺄 때 <strong>연관된 다른 정상 Feature 방향까지 함께 깎여 나갈 위험이 있음</strong>

---
## 16. SAE기반 Steering 방법 2: Latent feature 조절
 $$z^* = E(h) \rightarrow \text{Modify } z \rightarrow h^* = D(z^*)$$
Activation $h$를 SAE Encoder $E(\cdot)$에 통과시켜 각 Feature의 실제 활성화 크기인 $z$를 얻은 뒤, $z_j$의 값을 직접 조작하고 다시 Decoder $D(\cdot)$를 통과시켜 $h^*$를 복원하는 방식
즉,
$$z = E(h)$$
SAE를 통해 z를 얻은 후, feature을 조절
- 강화 (Addition): $z_j^* = z_j + \delta$
- 고정 (Clamping): $z_j^* = c$
- 완전 제거 (Ablation): $z_j^* = 0$
- 부분 약화 (Scaling/Attenuation): $z_j^* = (1 - \beta) z_j \quad (0 \le \beta \le 1)$
##### 부분 약화(Saciling)
도메인 차이나, 노이즈 같은 변수를 다룰 때 <strong>완전 제거는 모델의 표현을 다소 과격하게 왜곡</strong>시킬 수 있다. $z_j$의 비율을 조절하는 방식 $(1 - \beta)z_j$는 샘플이 원래 가지고 있던 $z_j$의 크기에 비례해서만 줄어들기 때문에, 앞서 다루었던 <strong>Projection, Conditional</strong>처럼 훨씬 부드럽고 안전하게 개입이 가능해진다.

---
## 17. SAE reconstruction error을 보존하는 방법
단순 교체 $h^* = D(z^*)$의 위험성 : SAE는 원본 Activation h를 <strong>완벽하게 복원</strong>하지 못한다. 즉, 항상 약간의 오차인 <strong>Reconstruction Residual</strong>($r = h - \hat{h}$)이 남는다. 만일 조작된 Latent $z^*$를 디코딩한 값 $D(z^*)$으로 원래의 h를 통째로 바꿔치기 하면, <strong>SAE가 미처 복원하지 못했던 원본 모델의 정상적이고 유용한 정보인 Residual r까지 통쨰로 날아가 버린다.</strong>
따라서 우리가 "SAE가 제대로 복원하지 못한 잔여 오차 r"을 그대로 유지하면서, <strong>SAE공간에서 우리가 조작한 변화량만 원본 h에 반영</strong>하는것이 목표이다.
원본 Residual 정의:$$r = h - D(z)$$Residual을 보존한 개입:$$h^* = D(z^*) + r = D(z^*) + (h - D(z))$$위식의 순서를 재정렬 (최종 수식):$$\boxed{h^* = h + \underbrace{\left(D(z^*) - D(z)\right)}_{\Delta h}}$$
즉, SAE reconstruction으로 원래 h를 완전히 교체하지 않고, <strong>SAE에서 조작한 변화량만 원본 activation에 적용</strong>한다.
```
[원본 Activation h] ───────── (원래 신호 유지) ───────────────┐
         │                                               │
         ▼                                               ▼
     SAE 공간                                     [개입된 Activation]
  z ──► z* 조작                                  h* = h + Δh
         │                                               ▲
         ▼                                               │
  Δh = Decoder(z*) - Decoder(z) (순수 변화량만 추출) ─────────┘
```
따라서 <strong>SAE의 복원 오차가 원본 h에 그대로 남아 보존되므로, SAE의 성능이 완변하지 않더라도 원본 모델의 기본 연산 능력을 해치지 않고 정밀하게 Steering할 수 있게 된다.</strong>

---
## 18. 부분 attenuation(약화)의 정확한 형태
선형 디코더($D(z) = W_{\text{dec}}z + b_{\text{dec}}$)를 사용하는 표준 SAE환경에서, 앞서 도출한 $\Delta h = D(z^*) - D(z)$ 수식을 직접 적용해보자.
제어하고자 하는 Feature 집합 $S$(e.g. 도메인 편향, 측정 장비 노이즈, etc.)에 속한 각 Feature j를 $\beta_j$의 비율만큼 악화시킨다면
$$z_j^* = (1 - \beta_j)z_j \quad \implies \quad z_j^* - z_j = -\beta_j z_j$$
이 변화를 직접 디코더 벡터 $d_j$와 결합하여 최종 개입 수식을 정리하면
$$\boxed{h^* = h - \sum_{j \in S} \beta_j z_j d_j}$$
- 고정 : $h^* = h - \alpha v$ → <strong>모든 샘플에 고정 상수 사용</strong>
- SAE : $h^* = h - \sum_{j \in S} \beta_j z_j d_j$ → <strong>샘플별 실제 활성화 크기</strong> $z_j$<strong>에 비례</strong>하여 적용
```text
Sample A
site feature activation z_j = 4.0
→ 많이 제거

Sample B
site feature activation z_j = 0.5
→ 조금 제거

Sample C
site feature activation z_j = 0.0
→ 제거하지 않음
```

---
## 19. SAE와 일반 Steering의 차이
### 19.1 일반 Contrastive Steering
```text
Positive group activation
        │
        ├── 평균 차이
        ▼
Steering vector v
        │
        ▼
h* = h - αv
```
- 방향 : Positive/Negative 그룹의 <strong>평균 차이 벡텨(v)</strong>계산
- 개입 방식 : 모든 샘플에 동일한 고정 벡터를 일률 가산
- 해석 가능성 : 낮음(벡터가 <strong>여러 개념의 혼합물일 가능성</strong>)
- 장점 : 연산이 단순, 직관적
- 단점 : 정밀제어 어려움, 부작용 존재

### 19.2 SAE feature steering
```text
전체 activation data
        │
        ▼
Unsupervised SAE
        │
        ▼
Sparse feature dictionary
        │
        ▼
Target과 관련된 feature 선택
        │
        ▼
선택 feature만 attenuation
```
- 방향 : SAE를 통한 비지도학습으로 <strong>개별 feature </strong>$d_j$<strong> 중 타겟과 연관된 것 선별</strong>
- 개입 방식 : <strong>샘플별 실제 활성화 크기</strong>($z_j$)<strong>에 비례한 부분 약화/강화</strong>
- 해석 가능성 : 높음(어떤 세부 feature가 조작되었는지 명확히 추적 가능)
- 장점 : 정밀한 타겟 제어 가능
- 단점 : Feature선정 및 가중치 튜닝 과정 필요

---
## 20. Top-down Steering과 Bottom-up SAE
### 20.1 Top-down : 목적 중심
<strong>명확한 목표(e.g. 병원 A와 병원 B데이터 사이의 편향을 제거하고 싶다)</strong>는 목표 부터 정의
```text
목표 행동 정의
   │
   ▼
positive / negative examples
   │
   ▼
contrastive direction
   │
   ▼
steering
```
- truthful vs untruthful
- site A vs site B
- positive vs negative
- refusal vs compliance

### 20.2 Bottom-up : 구조 분석
<strong>실제로 어떤 피처 조각들이 분산되어 있는지 모델 내부의 방대한 Activation을 SAE로 잘게 쪼개어 분석</strong>
```text
모델 activation
   │
   ▼
SAE decomposition
   │
   ▼
feature 1, feature 2, ..., feature M
   │
   ▼
각 feature 의미 분석
```

### 20. SAE
SAE는 결론적으로 bottom-up방식이고, contrasive는 top-down접근 방식이다. 이를 결합하면
```text
Top-down target definition
        │
        ▼
어떤 feature가 target과 관련되는가?
        ▲
        │
Bottom-up SAE dictionary
```
이 두가지를 활용해서 사람이 임의로 만든 거친 Steering vector대신, SAE가 분해한 피처중에서 진짜 타겟과 연관된 알맹이들만 골라서 정밀하게 조작하는 시스템 구축이 가능하다. 
다만, steering strength가 증가할수록 target효과와 일반 능력 보존 사이에 trade off가 발생할 수 있다.
| https://arxiv.org/abs/2501.09929

---
## 21. Contrastive Vector을 SAE로 분해하는 방법
기존 CAA으로 구한 대조 벡터($\mu_+ - \mu_-$)나 타겟 개념을 SAE공간 위로 곱씹어 해석하고 싶을때 사용하는 방법으로 <strong>contrastive벡터(대조 벡터)가 어떤 SAE feature들의 조합인지 분석할 수 있다.</strong>
1. Latent 차이 계산 : Positive 샘플 그룹에서 나온 SAE 활성화 평균과 Negative 샘플 그룹에서 나온 활성화 평균의 차이를 구한다. → 가장 직접적인 방법$$s_j = \mathbb{E}[z_j \mid +] - \mathbb{E}[z_j \mid -]$$ 
2. Top-K Feature Selection : 이 차이 값의 절대값($\vert{}s_j\vert{}$)이 가장 큰 상위 Feature들만 골라 타겟과 직결된 핵심 Feature 집합 $S$로 지정$$S = \text{TopK}_j(\vert{}s_j\vert{})$$
3. SAE 기반 합성 Steering Vector 생성 : 선택된 Feature들의 디코더 방향($d_j$)과 기여도($s_j$)를 엮어 새로운 정밀 벡터를 만든다.$$v_{\text{SAE}} = \sum_{j \in S} s_j d_j$$
```text
Positive / Negative samples
          │
          ▼
Frozen backbone
          │
          ▼
Dense activations h
          │
          ▼
SAE Encoder
          │
          ▼
Sparse activations z
          │
          ▼
Δz_j = mean(z_j | +) - mean(z_j | -)
          │
          ▼
관련 feature 집합 S 선택
          │
          ▼
v_SAE = Σ s_j d_j
          │
          ▼
Steering
```
주의할 점은 SAE decoder의 direction($d_j$)가 일반적으로 서로 직교하지 않는다는 점이다. 따라서 단순히 $d^T_jv$만 계산하면 feature contribution이 중복되거나 왜곡될 수 있다. 가능하면 latent activation difference나 sparse regression을 이용하는 편이 좋다.

---
---

> [!note]
> ### SAE에서 Feature의 추적
> <strong>SAE를 학습할 때는 기본적으로 데이터에서 뽑은 activation만 사용</strong>
> 하지만 SAE feature가 무엇을 의미하는지 알아내려면 다음 단계에서 metadata를 추출해야 함
> 즉, 2 stage로 진행해야함
> ```text
>  1단계: Feature discovery → activation만 사용
>  2단계: Feature grounding → metadata와 연결하여 의미 분석
> ```
> EEG FM SAE연구에서도 각 layer의 training activation만 모아서 TopK SAE를 학습한 뒤, 별도의 단계에서 abnormality, age, sex, medication metadata를 logistic regression, TCAV, enrichment test에 사용했음. 즉, SAE는 비지도학습을 통한 dictionary를 만들고, metadata를 통해 그 의미를 검증하는데 사용
> #### SAE 학습단계
> $h_{i,t}^{(\ell)} \in \mathbb{R}^{d}$
> - i : sample
> - t : token or window
> - l : transformer layer
> - d : embedding dimension
> 이고 SAE는
> ```text
> h₁,₁, h₁,₂, ..., h₁,T
> h₂,₁, h₂,₂, ..., h₂,T
> ...
> ```
> 이것만 보고,  $\mathcal{L}_{\text{SAE}} = \left\| h - \hat{h} \right\|_2^2$ 이를 목적함수로 사용한다. TopK SAE라면 sparsity는 TopK 연산 자체로 제어된다.
> SAE는 h만 보기 때문에 metadata를 직접 보지 않는다. (e.g. site, age, sex, etc.)
> 물론 이 metadata와 관련된 정보가 원래 신호에 존재한다면 SleepFM activation에 들어 있을 수 있다. SAE는 그 activation 구조를 분해할 뿐이다.
> #### 실제 추적
> 윈도우로 분해하더라도 해당 윈도우 혹은 임베딩에 어느 시점, 어느 환자에게서 추출되었는지 추적하는 인덱스 테이블 형식으로 저장해야한다.
> ```text
> PSG record
> │
> ├── patient_id = P103
> ├── site_id    = Site_B
> ├── label      = CI positive
> └── window     = 03:10:00–03:11:00
> │
> ▼
> SleepFM layer 2
> │
> ▼
> activation h
> │
> ▼
> SAE
> │
> ▼
> feature activation z
> ```
> #### Metadata : feature의 의미부여$\Delta_{A,j}$
> ##### 1. 단순 편차 분석
> 특정 병원 A에서 평균 활성화 값과 전체 평균 활성화 값의 차이를 계산
> $$\Delta_{A,j} = \mathbb{E}[z_j \mid \text{site}=A] - \mathbb{E}[z_j]$$
> 이 값이 크면 Feature j가 특정 metatdata와 연관이 깊다라고 1차 추론할 수 있다.
> ##### 2. 교란 변수 통제를 위한 회귀분석
> 단순 편차 분석만으로는 <strong>교란 변수에 의해 활성화되었는지, 혹은 어느 노이즈 때문에 활서와 되었는지</strong>와 같은 것들을 구분할 수 없다. 따라서 여러 조건을 동시에 넣어 선형회귀를 진행한다.
> e.g.
> $$z_j \sim \text{site} + \text{age} + \text{sex} + \text{label} + \text{device}$$
> ##### 3. 혼합 효과 모델
> 한 sample에서 수백개의 window 혹은 token이 나온다면, 동일 환자 내 샘플 간의 상관관계를 통제해야 통계적 왜곡이 생기지 않는다.
> e.g.
> $$z_{i,t,j} = \beta_0 + \beta_{\text{site}} d_i + \beta_y y_i + \beta_{\text{age}} \text{age}_i + \underbrace{u_i}_{\text{환자 개별 Random Effect}} + \epsilon_{i,t}$$
> ##### 결론
> 메타데이터를 붙여서 검증하는 방식은 <strong>우리가 이미 알고 있고 정의해둔 메타데이터 개념에 대해서만 Feature를 밝혀낼 수 있다.</strong> 다만 미지의 개념에 의해 반응하는 Feature라면 정체를 밝혀내지 못하고 미지의 피처로 남을 수 있다.
> | https://arxiv.org/html/2605.13930v3
> #### 📌 Layer간 feature 추적
> Layer1의 SAE feature 317 $\neq$ Layer2의 SAE feature 317 이다. 따라서 Layer간 동일 feature을 추적하려면 다음 중 하나가 필요하다.
> ```text
> 1. Decoder direction cosine similarity
> 2. Activation correlation
> 3. 동일 sample에 대한 co-activation
> 4. Hungarian matching
> 5. Crosscoder
> ```
> Crossencoder는 여러 layer의 activation을 하나의 sparse latent space에서 공동 복원하여 layer간 공유 feature을 찾기 위해 제안되었다.
> | https://transformer-circuits.pub/2024/crosscoders/index.html

> [!note]
> ### Top-activation example의 1위와 2위가 같은 의미인가?
> <strong>서로 다른 sample 여러 개에서 공통적으로 나타나는 패턴을 찾는다</strong>는 의미
> ```
> Feature 317 Top-activating examples
>
>1위: Patient 103, Site B, 01:10–01:11
>     60 Hz noise가 강함
>
>2위: Patient 418, Site B, 04:23–04:24
>     60 Hz noise가 강함
>
>3위: Patient 221, Site C, 02:31–02:32
>     60 Hz noise가 강함
> ```
> 서로 다른 데이터지만, 사람이 설명할때는 모두 "60 Hz noise"라는 같은 패턴으로 보임
> ```text
> [ Sample 1 (환자 P103) ] ──► z_317 = 8.4 (1위) ──┐
>[ Sample 2 (환자 P418) ] ──► z_317 = 7.9 (2위) ──┼──► 관찰: 모두 "60Hz Powerline Noise" 관찰됨
>[ Sample 3 (환자 P221) ] ──► z_317 = 7.1 (3위) ──┘           │
>                                                             ▼
>                                                Feature 317 = "60Hz Noise Feature" 추론
> ```
> 즉 <strong>값이 가장 큰게(TopK) 활성화된 입력 샘플들을 순서대로 나열한 것</strong>이다
> #### 주의점 : 동일 sample의 인접 window
> TopK를 사용할 때 거의 동일한 신호일 수 있다(동일 sample내 인접 window 혹은 token)
> 그러면 feature가 일반적인 패턴을 찾는것이 아니라, 특정 record 하나를 암기한 것일 수 있다.
> 따라서 top example을
> - 한 sample 당 1개
> - 한 record 당 1개
> - 각 데이터소스에서 n개
> - 서로 유사한 winodw는 클러스터링 후 1개
> 권장 e.g.
> ```
> Feature j의 모든 activation 계산
>             │
>             ▼
>Patient별 max activation window 선택
>             │
>             ▼
>상위 patient K명 선택
>             │
>             ▼
> Raw signal + PSD + metadata 비교
> ```
> #### Top examples의 주의점
> ```
> 상위 1–10위   : 60 Hz noise
>상위 11–20위  : 고령 환자의 delta power
>상위 21–30위  : Site B
> ```
> 와 같이 polysemantic할 수 있고, 혹은 feature가 특정 상황에서는 커져야 하지만 다른 SAE feature에 흡수되어 켜지지 않는 feature absorption이 발생할 수 있다. 따라서 top examples가 일관돼 보여도 causal intervention과 false-negative example을 같이 확인해야한다.
> 권장 검사
> - Top activation examples
> - High activation examples
> - Medium activation examples
> - Near-zero examples
> - 예상했지만 활성화되지 않은 counterexamples
> - Feature suppression 후 downstream 변화

> [!note]
> ### Threshold가 0.79라면 α로 성능을 높일 수 있는가?
> <strong>가능할 수 있지만, 고정</strong> $\alpha$<strong>가 단순히 prediction을 전체적으로 이동시키는 것이라면 성능 향상이 아니라 threshold 조정과 동일하다.</strong>
> Steering으로 AUROC까지 높이려면 sample마다 logit변화가 달라야 한다.
> 즉,
> $$s_i^* = s_i + \Delta s_i$$
> $$\Delta s_i \neq \text{constant}$$
> 이고
> - sample adaptive steering : $h_i^* = h_i + \alpha(h_i) v$
> - sample adaptive direction : $h_i^* = h_i + \alpha_i v_i$
> - SAE Activation-based Attenuation : $h_i^* = h_i - \sum_{j \in S} \beta_j z_{i,j} d_j$
> #### Tip
> Steering이 attention polling 혹은 mlp이전에 들어가면 동일한 vector도 sample마다 다른 nonlinear effect를 낼 수 있다.

> [!note]
> ### 하위 layer에서 한 번만 steering 하면 되는가?
> <strong>한 번만 적용할 수는 있다. 실제 일반적인 steering 방식이다.</strong> 하지만 "한 번만 적용하면 상위 layer의 비선형 왜곡을 피할 수 있다"는 해석은 맞지 않다.
> 하위 레이어에서 한번만 개입하더라도, 그 변형된 활성화 값은 <strong>상위의 모든 비선형 레이어들을 연속해서 통과하게 된다.(MLP, Attention, LayerNorm)</strong>
> 레이어 l부터 최종 레이어 L까지의 전체 연산이 변형이 작다고 할때 1차 테일러 근사를 진행하게 되면
> $$h_L^* = F_{\ell \rightarrow L}(h_\ell + \delta_\ell) \approx h_L + J_\ell(h_\ell) \delta_\ell$$
> 나오고 여기서 $J_\ell(h_\ell)$은 상위 레이어 연산들의 야코비안(Jacobian) 행렬이다.
> - 샘플 의존성 : 야코비안은 고정된 상수가 아니라, <strong>입력 샘플</strong>$h_l$<strong>의 위치에 따라 완전히 달라진다.</strong>
> - 결과적 현상 : 따라서 동일한 steering vector을 똑같이 더해줘도 사우이 비선형 레이어들을 지나면서 결과가 크게 달라진다.
> ```text
> ┌──► Sample A: 정상 제어 (Target 방향)
>                      │
>[동일한 δ_ℓ 주입] ────┼──► Sample B: 아무 변화 없음
>                      │
>                      ├──► Sample C: Anti-steerability (반대 방향으로 꺾임)
>                      │
>                      └──► Sample D: Representation 붕괴
> ```
> 또한 비선형 구조때문에 다음 레이어에서 똑같은 양을 더해주면 원상복구 되지 않는다.
> 따라서
> #### 어느 Layer을 선택해야 하는가?
> 이상적인 Layer는
> - Domain/Concept 정보가 충분히 인코딩 되어 있는가?
> - Task정보와 Domain 정보가 서로 얽히지 않고 분리되어 있는가?
> - 해당 Layer로 학습된 SAE의 복원이 안정적이고 DeadFeature가 적은가?
> - 상위 Layer로 올라갈때 개입량이 과도하게 증폭되거나 붕괴되지 않는가?
> e.g. SleepFM 연구에서 : Layer2,3은 안정적이었지만 Layer1은 성능이 무너지는 형상이 발생했다.
> <strong>따라서, 처음부터 초기 layer을 선택하지 말고, final embedding과 중간 layer모두를 비교해야 한다.</strong>

> [!note]
> ### 각 sample마다 domain 정보를 다르게 제거할 수 있는가?
> 고정 steering보다 더 타당한 방식이다.
> 하지만 Site A SAE, Site B SAE, Site C SAE는 권장하지 않는다. 이는 각 SAE가 별개의 좌표계를 학습하기 때문이다.
> 따라서 하나의 Shared SAE를 만드는것이 좋다.
> ```text
> Domain A activations ─┐
> Domain B activations ─┼──▶ Shared SAE
> Domain C activations ─┘
>                           │
>                           ▼
>                   공통 sparse space z
> ```
> 그 다음 metadata를 통해 domain별 feature mask를 만든다.
> #### Soft domain gate
> 도메인 분류기의 분류 확신에 따라 제거 감도를 연속적으로 조절한다. 즉 도메인 A일 확률일수록 A관련 feature을 많이 억제하고, 확률이 낮으면 거의 건드리지 않는다.
> $$z^* = z - \sum_d p(d \mid h) \left[ \beta_d \odot M_d \odot (z - c_{\text{ref}}) \right]$$
> reconstruction residual :  $z$을 Decoder($D$)로 복원하여 원본 정보 loss를 최소화한다.
$$\boldsymbol{h}^ = \boldsymbol{h} + D(\boldsymbol{z}^* - \boldsymbol{z})$$
> ```text
> Sample embedding h
>        │
>        ├────────────────────────────┐
>        │                            │
>        ▼                            ▼
> Shared SAE Encoder             Domain Gate
>        │                            │
>        ▼                            ▼
>        z                     p(A), p(B), p(C)
>        │                            │
>        └──────────────┬─────────────┘
>                       ▼
>        Domain-specific feature attenuation
>                       │
>                       ▼
>                      z*
>                       │
>                       ▼
>               Δh = D(z* - z)
>                       │
>                       ▼
>                  h* = h + Δh
>                       │
>                       ▼
>                  CI classifier
> ```
> 입력 의미에 따라 steering direction이나 strength를 동적으로 절하는 방법은 이미 activation steering 연구에서도 고정 vector의 sample variability를 해결하기 위한 방법으로 제시되고 있다.
> | https://arxiv.org/abs/2410.12299
> #### Subspace Projection(수학적 활용)
> 도메인 subspace $U$가 주어졌을때, 굳이 별도의 Classifier Gate를 두지 않고 <strong>도메인 방향으로의 내적크기</strong>를 이용한는 방식도 있다.
> $$\boldsymbol{h}_i^* = \boldsymbol{h}_i - \alpha U U^\top (\boldsymbol{h}_i - \boldsymbol{\mu})$$
> 이는 원본 임베딩 $h_i$가 도메인 공간 $U$에 많이 겹쳐 있으면(도메인의 성분이 크면) 많이 깎여 나가고, 별로 안 겹쳐 있으면 거의 깎이지 않는다.
> 따라서 이 방식은 hard domain classifier없이도 이미 sample-adaptive하게 된다.
> - `INLP`는 반복적으로 domain classifier을 학습하고 classifier 방향의 null space로 projection하는 방법이다.
> - `LEACE`는 모든 linear classifier가 target concept을 읽을 수 없게 하면서 representation 변화를 최소화하는 closde-form concept erasure 방법이다.
> #### Domain Gate의 문제
> Domain Gate 즉, Soft Gate방식은 학습할 때 <strong>이미 본 In-Domain SIte</strong>(A, B, C)데이터를 교정하는데 매우 강력하다. 즉, 보지 않은 데이터(D)가 들어오면, Gate classifier가 유사한 쪽으로 무리하게 확률을 할당하여 엉뚱하게 feature mask가 씌워질 위험이 있다.
> 📌 <strong>따라서, 일반화가 목적이라면 특정 Site의 Centroid를 지우기 보다, 여러 도메인 전반에 공통으로 존재하는 Nuisance Subspace를 찾아 Projection으로 날리는것이 중요하다.</strong>
> #### Outcom-Domain Confounding
> 도메인 정보를 무작정 지울 때 발생하는 가장 치명적인 위험은 <strong>도메인 특성과 실제 타겟 정답이 얽혀 있는 경우이다.</strong> 즉, 예측에 꼭 필요한 정보까지 같이 삭제되는 경우가 있다.
> ##### 해결책 : Lael-conditional Centroid
> 📌 도메인 평균을 구할 때 전체를 하나로 뭉뚱그리지 말고, <strong>라벨 별로 나누어 도메인 Centroid를 계산한다.</strong>
> $$\boldsymbol{c}_{d,y} = \mathbb{E}[\boldsymbol{z} \mid d, y]$$
> 추론 시점에는 타겟 clf.의 예측 확률을 가중치로 하여 도메인 특성만 정교하게 제거한다.
> $$\boldsymbol{c}_d(h) = \sum_{y} q_y \boldsymbol{c}_{d,y}$$
> `CDAN`은 단순히 marginal domain distribution만 정렬하면 class 구조가 무너질 수 있기 때문에 classifier prediciton을 조건으로 domain alignment를 수행한다.
> | https://arxiv.org/abs/1705.10667

> [!note]
> ### Fine tunning의 Domain영향을 지우는데 Steering을 사용하면 어떨까?
> SAE-guided domain-invariant representation learning 혹은 SAE-guided conditional nuisance attenuation이다.
> 1. SAE의 해석 가능성: 도메인 잡음(Nuisance) 요소만 명확히 분리함
> 2. Metadata 결합: 각 사이트 특성에 맞는 Feature Mask 선택
> 3. 샘플 적응형: 각 샘플의 도메인 성분 세기만큼만 자율적으로 attenuation(감쇄)
> 4. Task 보존: 질병 예측 등 본래 Task 정보는 손실 없이 유지
> 5. 재인코딩 방지: 파인튜닝 중 파운데이션 모델(FM)이 도메인 정보를 꼼꼼히 우회해서 다시 학습하는 현상을 막음
> #### 권장 loss : DANN
> 파인튜닝 모듈의 최적화 손실함수는 <strong>대립적 학습 구조를 띈다.</strong>
> $$\min_{\phi,C} \max_G \quad \mathcal{L}_{\text{task}} - \lambda_d\mathcal{L}_{\text{domain}} + \lambda_{\Delta}\Vert{}h^*-h\Vert{}_2^2 + \lambda_s\Vert{}\beta\Vert{}_1$$
> - $L_{task}$ : CI 성능 유지 또는 향상
> - $L_{doamin}$ : domain classifier가 domain을 맞히기 어렵게 함
> - $\Vert{}h^ - h\Vert{}_2^2$ : reconstruction loss로 원본 임베딩 정보가 너무 크게 훼손되지 않도록 왜곡을 제한
> - $\Vert{}\beta\Vert{}_1$ : 꼭 필요한 최소한의 Feature만 제어하도록 L1을 줌
> #### 위험성
> 그러나 파인튜닝을 하면 SAE가 낡아버릴 수 있다 SAE는 고정된 FM의 출력 분포를 기반으로 학습되기 때문에, FM 전체를 Full Finetunning해버리면 FM의 파라미터가 바뀌면서 출력 분포가 이동한다. 따라서 SAE의 Feature의미가 깨져버려 엉뚱한 신호를 지우는 참사가 발생할 수 있다.
> ```text
> [Fine-tuning 전] SAE Feature 317 = "Site A 센서 잡음"
>        │ (FM 파라미터가 바뀌면서 의미가 달라짐)
>        ▼
>[Fine-tuning 후] SAE Feature 317 = "환자의 실제 생체 신호 (Task 핵심 정보)"
> ```
> 따라서 1. Frozen → SAE, 2. 마지막부분 Unfreeze → SAE, 3.full fine tunning
> ```text
> [Stage 1] Frozen FM & SAE (가장 안전)
>  └─ SleepFM & SAE 고정 ──▶ Classifier, Attenuation Strength(β), Domain Gate만 학습
>
>[Stage 2] Partial Fine-tuning (선택적 해제)
>  └─ SleepFM의 Last Block + Classifier + Controller만 미세 학습 (작은 Learning Rate)
>
>[Stage 3] Full Fine-tuning (실험용 비교군)
>  └─ SAE 재학습/주기적 Refresh 및 Domain Adversarial Loss 필수 병행
> ```
> #### Backbone이 domain을 다시학습
> 고정된 domain feature을 제거하면서 FM을 fintunning하면 FM은 다른 feature에 domain정보를 다시 저장할 수 있다. 즉 site feature 317제거시 → FM이 site 정보를 feature 529에 재인코딩 할 수 있다. 따라서 이를 막기 위해 final representation에 domain adversarial loss를 계속 걸어야한다.
> 즉. SAE feature attenuation + final domain discriminator을 같이 사용하는것이 안전하다.
> #### Domain정보를 0으로 만들면 좋은가?
> 아니다. <strong>도메인 특성과 정답 라벨의 분포가 완전히 독립적이지 않다면, 도메인 차이를 지우려고 할 때 Class 고유의 정보까지 함께 파괴된다.</strong>
> 즉, domain-irvariant representation만 학습한다고 domain adaptation 성공이 보장되지 않으며, label distribution이나 conditional distribution이 다를 경우 source와 target의 공동 오류가 커질 수 있다는 이론적 결과가 보고 되었다.
> | https://proceedings.mlr.press/v97/zhao19a/zhao19a.pdf
> ```text
> [잘못된 목표] 도메인 정보 완전히 0으로 지우기 (X)
>              ▼
>    Class 간의 고유한 유효 분포 구조까지 함께 무너짐
>
>[올바른 목표] Task와 무관한 도메인 잡음(Nuisance)만 억제하기 (O)
>              ▼
>    생리학적/의학적 타겟 정보(Physiological Information)는 안전하게 보존
> ```
> 즉, 도메인 정보를 무작정 "삭제"하는 것이 아니라 "Task에 불필요한 Nuisance 정보만 선택적으로 감쇄"시키는 것이 중요

> [!note]
> ### SAE로 metadata를 맞춘다?
> 비지도학습을 어떻게 지도학습처럼 쓸 수 있을까? → <strong>SAE가 비지도 학습으로 feature을 만들고, 그 feature가 어떤 metadata를 인코딩 하는지 별도의 지도학습 probe로 확인</strong>
> #### feature 식별
> 1. Site relted feature식별 : 비지도 학습으로 뽑아낸 j번째 SAE feature $z_j$가 특정 Site A와 얼마나 결합되어 있는지 <strong>Logistic Regression Probe</strong>난 평균 활성화 차이로 측정 → Site 특화 라벨링 완성
> $$\Delta_{A,j} = \mathbb{E}[z_j \mid \text{Site A}] - \mathbb{E}[z_j \mid \text{other sites}]$$혹은 $$d = softmax(Wsitez+bsite)$$등을 사용
> 2. 안전한 Steering 및 Reconstruction Residual 보존 : 식별된 Site feature 집합 $S_{\text{site}}$에 대해서만 값을 기준점($c_{\text{ref}}$) 방향으로 약화($\beta$)시킨다.$$z_j^* = z_j + \beta_j (c_{\text{ref},j} - z_j)$$ 이때, 전체 임베딩을 SAE의 출력값으로 완전히 덮어씌우면 SAE의 복원 오차때문에 표현력이 손상되기에, 변화량에 Decoder를 곱한 차이값만 원본 임베딩 h에 더해줌(위의 SAE를 활용한 steering vector방식)
> #### 조작 결과
> ```text
> h3에서 site-related feature만 조작
> 			│
> 			▼
>Site 검출 성능             크게 하락
>Age 검출 성능              유지 또는 소폭 변화
>Sex 검출 성능              유지 또는 소폭 변화
>CI label 검출 성능         유지 또는 상승
>Downstream CI 성능         유지 또는 상승
> ```
> #### 기존 probe만 낮아지면 충분할까?
> 아니다, 단지 기존 Probe가 바라보던 방향의 벡터만 깎을 뿐, <strong>다른 차원에 Site 정보가 그대로 남아있을 가능성</strong>을 배제하지 못한다.
> 따라서
> ```text
> 기존 frozen site probe 성능
> → 의도한 방향이 실제로 변했는가?
>
>h*에서 새로운 site probe 재학습
> → 다른 방향을 찾아 site 정보를 복구할 수 있는가?
> ```
> 즉, 새 probe를 다시 학습해도 site를 맞히지 못해야 더 강한 정보 제거 증거가 된다. `LEACE`, `INLP`와 같은 concept-erasure 방법도 특정 probe 하나가 아니라 모든 선형 probe가 target concept을 복구하기 어렵게 만드는 것을 목표로 한다.

> [!note]
> ### Fourier 주파수 분해 후 주파수 Steering Vector 차지
> 









