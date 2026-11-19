---
title: KD(Knowledge Distillation) Basic
date: 2026-09-03 22:50:34 +0900
categories:
  - AI-ML-DL
tags:
  - Knowledge-Distillation
math: true
---
# 1. Knowledge Distillation
<strong>KD(Knowledge Distillation, 지식 증류)</strong>는 성능이 좋은 <strong>Teacher(큰 모델)</strong>이 학습한 정보를 가지고 <strong>Student(더 작고 가벼움)</strong> 모델의 학습에 사용하는 방식이다.

> 단순히 큰 모델의 weight를 작은 모델에 복사한다가 아니다.

<strong>Teacher와 Student의 architecture가 달라도 된다.</strong>

큰 Teacher가 가지고 있는 지식 즉,

- 최종 prediction distribution : class들을 어떻게 판단하는지
- intermediate feature : 입력을 어떻게 표현하는지
- attention : 입력의 어디를 중요하게 보는지
- sample간 관계

등을 작은 Student에 <strong>additional supervision</strong>으로 사용할 수 있다.
##### Teacher가 학습한 판단 방식과 표현을 작은 Student가 따라 배우는 방식

> #### Additional Supervision
> <strong>기존 Ground Truth loss만으로 학습하던 Student에게 Teacher의 prediction이나 feature 같은 추가적인 학습 목표를 제공하는 것</strong>
{: .prompt-info }

## 1.1. 필요성
큰 neural network(NN)은 높은 capacity를 가지만,

- parameter가 큼
- memory 사용량이 큼
- FLOPs가 큼
- inference가 느림

과 같은 문제가 있어, Efficient AI에서는 다음과 같이 문제를 해결한다.   


| 방법                   | 핵심                                   |
| ---------------------- | -------------------------------------- |
| Pruning                | 중요도가 낮은 weight/channel 제거      |
| Quantization           | FP32 → INT8 등 낮은 precision 사용     |
| Knowledge Distillation | 큰 모델의 knowledge를 작은 모델에 전달 |

KD의 특징은 <strong>Student Architecture 자체를 deployment 시점에 더 크게 만들 필요가 없다는 것</strong>이다.
이는

- Train : Teacher ↔︎ Student
- Inference : Student only

따라서 Teacher을 사용함으로 <strong>training cost는 증가</strong>하지만, inference cost 자체는 늘어나지 않는다.

#### Hinton KD의 원래 목적
초기 KD는 단순히 큰 단일 모델을 줄이는 것뿐 아니라, **ensemble 또는 cumbersome model이 가진 generalization ability를 deployment 가능한 single model로 압축하는 것**에서 출발하였다.

## 1.2. Teacher, Student
### Teacher 
일반적으로

- 더 깊음
- 더 넓음
- parameter수가 많음
- representation capacity가 큼
- task accuracy가 높음   
   
### Student
일반적으로

- 작음
- parameter가 작음
- inference가 빠름
- deployment가 쉬움

다만
##### Teacher가 크다고 무조건 좋은 Teacher인 것은 아니다.
Teacher과 Student의 capacity gap이 지나치게 차이나면 Student가 Teacher의 복잡한 function을 잘 따라가지 못할 수 있다.
## 1.3. Knowledge?
단순히 $weight = knowledge$가 아니다. Teacher가 입력 $x$에 대해 만들어내는 $f_T(x)$, 즉 <strong>모델의 행동</strong> 자체에도 knowledge가 있다고 본다.

예를 들어

```
cat         0.70
dog         0.20
automobile  0.01
...
```

로 판단했다면 `cat`에대해서 dog와는 어느정도 비슷하게 보지만, automobile과는 거의 관계가 없다고 본다라는 정보또한 포함된다.

Hinton 교수님의 KD에서는 knowledge → <strong>입력에서 출력으로 가는 learned mapping 자체를 knowledge로 보자고</strong>한다.
## 1.4. Hard Label과 Soft Target
일반적인 분류문제에서의 지도학습에는 <strong>Hard Label</strong>을 사용한다.

```
cat = [1, 0, 0, ...] → 정답은 cat이다.
```

반면 Teacher의 probability distribution을 만든다면 이를 <strong>Soft Target</strong>으로 사용할 수 있다.

- Teacher의 condfidence
- non-target class 사이의 관계
- class similarity structure

등이 포함될 수 있다.

<strong>Soft Target의 엔트로피가 높으면 한 training sample에서 1. 정답 class 2. 두 번째 후보 3.세 번째 후보 4. class간 similarity를 동시에 받을 수 있다.</strong>

즉, KD는 사실상 
##### Teacher가 데이터에서 발견한 regularity를 Student에게 전달
하는 방법으로 볼 수 있다. 이는 Hinton 교수님의 논문에서도 Soft Target이 Regularizer로 작동하는 section6에도 <strong>적은 데이터에서도 overfitting을 크게 완화한것을 보아 Teacher-informed regularization</strong>으로도 볼 수 있다는 의미이다.

## 1.5. Dark Knowledge
hard label만으로는 드러나지 않는 Teacher의 상대적인 class 관계 정보이다.

```
Teacher prediction

truck       0.80
automobile  0.15
airplane    0.04
dog         0.001
```

이면 hard label에서는 truck = 1, others = 0으로 보겠지만, 모델이 어떻게 라벨을 바라보고 있는지 <strong>구조를 확인</strong>할 수 있게 된다.

## 1.6. 전체 KD 구조

```
                       Ground Truth
                            │
                            ↓
                       CE Loss
                            ↑
                            │
Input ───────────────→ Student
 │                         │
 │                         ↓
 │                  Student Logits
 │                         │
 │                         ↓ / T
 │                  Student Soft Output
 │                         ↑
 │                         │
 └──────────────→ Teacher  │
                   │       │
                   ↓       │
              Teacher Logits
                   │
                   ↓ / T
              Soft Target
                   │
                   └──── KD Loss
```

$$L_{total} = \lambda_{CE}L_{CE}+\lambda_{KD}L_{KD}$$
를 최소화하는 문제로 볼 수 있다.

## 1.7. Output dimension
<strong>Capacity</strong> 차이로 Teacher, Student가 존재한다.
<strong>Teacher, Student의 input / output task는 같지만 중간 함수의 capacity가 다르다.</strong>


내부 architecture은 달라도 된다. 그러나 output-based KD에서는 같은 class space를 에측해야한다.
$$P_{Teacher} ↔︎ P_{Student}$$를 비교하기 위함이다.

즉, $$\#\ Parameter_{Teacher} > \#\ P arameter_{Student}$$이지만, Teacher와 Student의 Output dimension은 동일해야한다.
<strong>capacity는 다르지만 task space는 같다.</strong>

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/2c922c4ccc2aa521afaaa8b922044513.png)

## 1.8. BaseRun
### 1. Teacher 먼저 학습
<strong>Teacher는 ordinary supervise learning으로 학습</strong>
$$L_T = CE(z_T, y)$$
KD에서 Teacher가 유용한 정보를 주려면 우선 <strong>Teacher 자체가 task를 충분히 잘 수행할 수 있어야 한다.</strong>

### 2. KD 

##### Same Random Seed

```
Architecture = Same
Initialization = Same
Dataset = Same
Training setting = Same

Difference = KD
```

> #### KD
> <strong>KD는 compression뿐 아니라 regularization 관점도 있다.</strong>
{: .prompt-info }
# 2. Hinton's KD


![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/ba77d6ec8e6277f77ef04000765d743d.png)

Hinton교수님의 Output Distillation
#### Logit
Logit은 classifier의 <strong>softmax 이전 raw score</strong>이다.
#### Softmax
Logits를 probabiltiy distribution으로 변환
$$p_i = \frac{e^{z_i}}{\sum_je^{z_j}}$$
잘 학습된 Teacher는 prediction이 너무 sharp할 수 있어서 Teacher의 일반 softmax를 사용하지 않는다.
#### Temperature
$$p_i(T) = \frac{exp(\frac{z_i}{T})}{\sum_jexp(\frac{z_j}{T})}$$
- $T = 1$ : 일반 softmax
- $T < 1$ : distribution이 더 sharp해진다.
- $T > 1$ : distribution이 soft해진다. 따라서 KD에서는 보통 $T > 1$을 사용한다.

e.g. 직관적으로 Teacher의 logtis들이 T = 4라고 했을때
[8, 4, 2, 1] → [2, 1, 0.5, 0.25]가 되며, 원래 큰 logit 차이가 줄어들게 되므로 작은 probability를 가진 class들의 정보도 더 드러나게 된다.

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/9ed2242e6f8eacde1b45b5e184265a29.png)

T가 너무 커지면 Teacher의 정보가 약해질 수 있다. 이는 $T → \infty$<strong>이면 거의 uniform해진다.</strong>

추가적으로 <strong>Deployment시 Student는 T = 1</strong>을 사용한다.

> #### 높은 Temperature에서 soft-target distillation을 분석
> Temperature scaled된 확률을  각각
> $$
> p_i = \frac{\exp(v_i/T)}{\sum_j \exp(v_j/T)},
> \qquad
> q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
> $$   
> Teacher logits : $v_i$, Student logits : $z_i$   
> $$C = -\sum_i p_i \log q_i$$이고 이때 $z_i$에 대한 cross-entropy의 gradient는   
> $$\frac{\partial C}{\partial z_i} = \frac{1}{T}(q_i - p_i)$$   
> 이고 T가 충분히 높고 logits가 zeromean이라고 가정하면   
> $$\frac{\partial C}{\partial z_i} \approx \frac{1}{N T^2}(z_i - v_i)$$   
> 즉 높은 T limit에서는 사실상   
> $$\frac{1}{2}(z_i - v_i)^2$$   
> 을 최소화 하는것과 비슷하다. 이는$$L_{\mathrm{logit}}=\frac{1}{2}\sum_i (z_i-v_i)^2$$을 student logit $z_i$에 대해 미분하면 $$\frac{\partial L_{\mathrm{logit}}}{\partial z_i}=z_i-v_i$$이고 상수를 제외하면 동일하다.
> 이는 KD가 뜬금없이 soft probability를 쓰는게 아니라. <strong>기존의 logit matching을 더 일반적인 형태로 확장한 것으로 볼 수 있다.</strong>
> logit matching은 단순하고 효과적이지만, 중요한 비식별성이 있는데, 모든 logit에 같은 상수를 더해도 softmax확률은 변하지 않는다는 점이다. 즉 확률 분포를 결정하는건 클래스가 상재적인 차이이고, zero-mean 조건은 이러한 공통 offset을 제거하여 teacher와 student의 logit을 비교 가능하게 만든다.
{: .prompt-info }

#### 이게 왜 Knowledge?
위에서도 설명했다 싶이, 정답뿐만 아니라, <strong>Student가 구조 또한 학습</strong>할 수 있기 때문이다.
추가적으로, <strong>Teacher와 Student모두 같은 Temperature을 사용한다.</strong>
$$p_{T} = softmax(\frac{z_T}{T})$$
$$\log{p_s} = \log{softmax(\frac{z_S}T)}$$
## 2.1. KL Divergence
KD에서는 주로 KL divergence를 통해 계산한다.
$$D_{\mathrm{KL}}(P \parallel Q) = \sum_i P_i \log \frac{P_i}{Q_i}$$   
$$D_{\mathrm{KL}}(P \parallel Q) = \sum_i P_i (\log P_i - \log Q_i)$$   
$$\mathcal{L}_{\mathrm{KD}} = \sum_i p_{T,i} (\log p_{T,i} - \log p_{S,i})$$   
$$\mathcal{L}_{\mathrm{KD}} = \sum_i p_{T,i} \log p_{T,i} - \sum_i p_{T,i} \log p_{S,i} = -H(P_T) + H(P_T, P_S)$$   
$$\nabla_{\theta_S} \mathcal{L}_{\mathrm{KD}} = \nabla_{\theta_S} \mathcal{L}_{\mathrm{CE}}(P_T, P_S) = -\nabla_{\theta_S} \sum_i p_{T,i} \log p_{S,i}$$   
이때   
$$D_{KL}​(P∥Q) \neq D_{KL}​(Q∥P)$$이다. KL은 symmetric distance가 아니다. 따라서 KD에서는 Teacher distribution을 target으로 두고 Student가 이를 따라가게 한다.

### 1. $T^2$의 필요성
Temperature T가 커지면 distribution만 부드러워지는게 아니라, <strong>KD gradient의 scale도 작아진다.</strong> 이를 보정하기 위해 $T^2$을 곱한다. $$L_{KD} = T^2D_{KL}(p_T^{teacher}\ ||\ p_S^{student})$$
$T^2$<strong>는 gradient scale 보정일 뿐이다. 원래대로 되돌리는 것이 아니다.</strong>

### 2. KD의 Loss
$$L = \lambda_{KD}L_{KD} + \lambda_{CE}L_{CE}$$
가 Student의 최종 loss이다.
이때 <strong>Teacher가 항상 옳은 것은 아니기에 Ground Truth CE도 유지</strong>한다.
- <strong>Soft Target</strong> : Teacher의 generalization structure을 전달.
- <strong>Hard Target</strong> : 실제 Ground Truth 방향으로 Studnet를 교정

Hinton 교수님의 논문에서는 Teacher와 다른 방향으로 틀리더라도 <strong>Groung Truth방향으로 틀리는 쪽이 낫다</strong>라고 표현하였다.

## 2.2. Offline KD
Teacher는 이미 학습된 <strong>information provider</strong>이기 때문에, 일반적인 offline KD에서는 <strong>Student만 학습</strong>된다.
##### Teacher
```
Teacher
eval
no_grad
  ↓
teacher logits
```
##### Student
```
Student
train
gradient
optimizer
```

| 방식              | 구조                                   |
| ----------------- | -------------------------------------- |
| Offline KD        | pretrained Teacher → Student           |
| Online KD         | 모델들을 같이 학습하며 knowledge 교환  |
| Self-Distillation | 동일/유사 모델 내부에서 knowledge 전달 |
| Multi-Teacher     | 여러 Teacher가 Student를 지도          |
   
## 2.3. Transfer Set

KD에서 Student가 Teacher의 soft target을 학습하는 데이터는 기존 training set일 수도 있고 별도의 transfer set일 수도 있으며, Teacher가 target을 만들어줄 수 있기 때문에 <strong>unlabeled data도 활용 가능</strong>

# 3. Representation KD
<strong>output이 아니라 hidden representation</strong>을 전달하는 방식이다. 대응대는 stage끼리 맞춘다.

```
Teacher hidden representation
             ↕
Student hidden representation
```

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/cfac4bc374e54024ec8e7b90ecf5dec3.png)

## 3.1. Hidden representation 전달
최종 logits는 매우 압축된 결과이씨 때문

Teacher 내부에서는

```
Input
 ↓
low-level feature
 ↓
higher-level feature
 ↓
hidden representation
 ↓
logits
```

이며 Teacher의 높은 성능이 좋은 internal representation에서 나왔다면
##### 결과뿐만 아니라 representation을 형성하는 방식도 Student가 배워보자
는 접근이 가능하다.

## 3.2. Cosine Similarity
$$\cos\theta = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$$
코사인 유사도를 활용하여 진행한다.

## 3.3. 그대로 matching은 조심해야 한다.
Teacher와 Student의 Architecture가 다르기 때문에, Teacher의 feature coordinate와 Student의 Feature coordinate가 반드시 같은 의미를 가질 이유가 없다. 즉, <strong>두 모델이 동일한 정보를 표현하면서도 내부 basis가 다를 수 있다.</strong> 따라서 단순한 component wise matching이 언제나 최적이라고 볼 수 없다.

## 3.4. Dimension 문제
Teacher와 Student의 Hidden dimension이 동일하지 않기 때문에 바로 cosine similarity를 계산할 수 없다. <strong>따라서 pooling과 같은 방식으로 차원을 맞춰줄 필요가 있다.</strong>

# 4. MSE KD

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/550961ec6f3b452f43d364a31561df07.png)

pooling을 통해서 dimension을 강제로 맞추는 것보다 유연하게

```
Student feature
      ↓
Learnable mapping
      ↓
Teacher feature space
```

## 4.1. Intermediate Regressor
$r(\cdot)$을 regressor이라고 하면
$$r(F_S) \approx F_T$$
이 되도록 학습한다.
<strong>Student representation을 Teacher space로 변환하는 함수 자체를 학습</strong>하는것으로 바꾼다.

> #### Regressor
> <strong>통계의 회귀보다는 feature projection에 더 가깝다.</strong>
> e.g. 
> ```
> Student
> [B,16,8,8]
 >     ↓
>Regressor
>    ↓
>[B,32,8,8]
>
>Teacher
>[B,32,8,8]
> ```
{: .prompt-info }

## 4.2. MSE Feature Distillation
shape를 regressor로 맞춘 뒤
$${L}_{\text{feature}} = \frac{1}{N} \|F_T - r(F_S)\|_2^2$$   
$${L} = \lambda_{\text{feature}} {L}_{\text{feature}} + \lambda_{\text{CE}} {L}_{\text{CE}}$$   
로 만들 수 있다. 최종 loss는 $L$이다.

## 4.3. FitNets와의 연결
FitNets는 중간 representation을 사용한다.

```
Teacher intermediate representation
= Hint

Student intermediate representation
= Guided representation
```

Student Feature가 Teacher의 Hint를 따라가도록 한다. 이는
$$L_{\text{hint}} = \|F_T - r(F_S)\|_2^2$$
이다.

# Check

1. `eval()`은 Teacher의 Dropout/BatchNorm 등을 inference mode로 고정하기 위해 사용한다.
2. `torch.no_grad()`는 Teacher를 학습하지 않으므로 불필요한 gradient 계산과 graph 생성을 막기 위해 사용한다.
3. Temperature $T$가 커지면 distribution이 soft해져 non-target class의 상대적인 정보가 더 드러난다.
4. Hard-label CE는 실제 정답 class를 맞추도록 Student를 학습한다.
5. Soft Target은 Teacher의 class 간 상대적 관계와 confidence인 Dark Knowledge를 전달한다.
6. $T^2$는 Temperature 증가로 작아지는 KD gradient scale을 보정하기 위해 사용한다.
7. Representation KD에서는 hidden dimension이 다를 수 있어 pooling이나 projection으로 dimension alignment가 필요하다.
8. Output KD는 logits/probability, Representation KD는 hidden representation, MSE KD는 regressor로 정렬한 intermediate feature를 전달한다.
9. TAKD는 `Teacher → Student`의 큰 capacity gap을 줄이기 위해 `Teacher → Teacher Assistant → Student`로 knowledge를 단계적으로 전달한다.

##### 실습한 자료

| [PyTorch Knowledge Distillation 실습 자료](https://drive.google.com/file/d/1yJbf37gjslQ9omgy7ymhL82DDWOL2QLc/view?usp=sharing)

##### 실습한 자료 빈칸버전 (PyTorch 공식 튜토리얼 변형)

| [PyTorch Knowledge Distillation 빈칸 실습 자료](https://drive.google.com/file/d/1SeV0mhPb3IpbRFHZGhr52jxxb6UgShM-/view?usp=sharing)

##### 다음 논문 부터는, 간략하게 파악한 정도만 정리하였고 다른 섹션으로 나눠서 자세히 읽은 후 진행할 예정이다.
# 5. Improved Knowledge Distillation via Teacher Assistant

<strong>Teacher가 더 크고 정확하다고 해서 Student에게 항상 더 좋은 Teacher는 아니며, capacity gap이 너무 크면 중간 크기의 Teacher Assistant를 두는 것이 더 좋다.</strong>

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/5de66891dfc1964591af4a4926fd54fc.png)

Student를 2-layer CNN으로 고정하였을때 Teacher을 4, 6, 8, 10개의 레이어들로 점점 크게 만드는 실험이다. Teacher의 accuracy는 계속 증가하지만, Student의 distillation accuarcy는 증가하다가 어느 지점에 최대를 이루며 다시 감소하는 경향을 보인다.

## 5.1. 왜 이런 일이 일어날까?
1. <strong>Better Teacher</strong> : Teacher가 커지면 accuracy가 올라가므로 더 좋은 supervision을 제공한다. → <strong>KD에 긍정적</strong>
2. <strong>Capacity Gap</strong> : Teacher가 너무 복잡해지면 Student capacity로 Teacher behavior을 mimic하기 어려워 진다. → <strong>KD에 부정적</strong>
3. <strong>Teacher Confidence</strong> : 큰 Teacher는 data에 더 confident해져서 soft target이 덜 soft해질 수 있다. 즉 dark knowledge 전달이 약해질 수 있다 → <strong>KD에 부정적</strong>
위의 요소들을 각각 Factor 1~3라고 한다면
##### Factor1 > Factor2 + Factor3가 되다가 어느순간부터 Factor1 < Factor2 + Factor3로 역전되어 성능이 하락한다. → capacity-gap 관점

## 5.2. TAKD

```
Large Teacher
      ↓
Medium Teacher Assistant
      ↓
Small Student
```

와 같이 중간 TA를 둔다.
$$T → TA → S$$
형태로 총 2step으로 distill을 진행한다.
<strong>최종 Student는 original Teacher에게 직접 배우는게 아니라 TA에게 배운다.</strong>
1. Teacher → Student : Huge capacity gap
2. TA → Student : Smaller capacity gap

TA를 사용할 경우 논문 설명으로는

- Factor 2(Capacity gap) 완화
- TA가 덜 confident할 수 있어 Factor3 완화
- 대신 TA accuracy는 Teacher보다 낮다 Factor1이 손해

그럼에도 실험적으로는 2, 3에서 얻는 이득이 1에서의 손해보다 컸다.

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/1b74c185e013bfdf91c34f14bed8bb8f.png)

> #### 비교군
> - <strong>NOKD</strong> : No KD
> - <strong>BLKD</strong> : Baseline KD
{: .prompt-info }

## 5.3. TA의 크기

<strong>network depth의 midpoint가 최적이 아니다.</strong>

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/f904fa76daa6c43993df858d47a5d409.png)

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/e0d29e23e4f4cc5b323b2a48129ff31f.png)

<strong>Teacher와 Student의 standalone accuracy 평균에 가까운 모델이 midpoint보다 최적인 경우가 많았다.</strong>

## 5.4. Path

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/4de6258dac392f544d68853a88e6ad7a.png)

<strong>resource가 충분하다면 여러 TA를 거치는 full distillation path가 좋다.</strong>

## 5.5. VC Theory
모델의 generalization error을 대략 <strong>estimation error + approximation error</strong>로 나누어 보는 통계적 learning theory이며, TAKD에서는 Teacher → Student를 Teacher → TA, TA → Student로 나누면 error upper bound가 작아질 수 있다는 이론적 근거로 사용되었다. 단, 실제 성능 자체가 아니라 <strong>asymptotic justification이다.(upper bound기반)</strong>
## 5.6. Loss Landscape

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/45c340aba37d2187481b841f2aa5bfbb.png)

TAKD는 NOKD, BLKD보다 local minimum 주변에서 더 flat한 loss landscape를 보였으며, 논문에서는 이를 noise에 대한 robustness와 better generalization의 근거 중 하나로 해석하였다.

단, <strong>supporting evidence로 보는 것이 적절하다.</strong>

# 6. Densely Guided Knowledge Distillation using Multiple Teacher Assistants

<strong>TAKD가 여러 TA를 직렬로 거치면서 앞 단계의 오류가 후속 TA로 전달되어 누적될 수 있는 error avalanche 문제를 완화하기 위해, 각 작은 모델을 Teacher와 이전 모든 TA가 동시에 지도하도록 만든다.</strong>

```
T
↓
A1
↓
A2
↓
A3
↓
S

A1 error
   ↓
A2 learns error
   ↓
A3 learns accumulated error
   ↓
Student
```

처럼 잘못된 knowledge를 전달하면 <strong>Error Avalanch Problem</strong>이 기존의 TAKD에서 발생한다.

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/f71407927cd28b2c8ce4fb1302972345.png)

## 6.1. DGKD
<strong>TAKD의 step를 늘리는 것이 오히려 error propagation path도 늘리는 것이 아니냐?</strong>라는 의문에서 시작.

```
Teacher ──────────────┐
   ↓                  │
  A1 ───────────┐     │
   ↓            │     │
  A2 ──────┐    │     │
   ↓       │    │     │
Student ←──┴────┴─────┘
```

와 같은 형태로 학습하고, TA자체를 학습할 때도 동일하게 dense guidance를 적용한다.

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/3363c746688a1028253210782136651a.png)
## 6.2. Dense?
DenseNet에서도 이전 layer들의 feature값을 이후 layer에서 모두 활용하듯, DGKD에서는 이전에 존재하는 <strong>이전에 학습된 모든 상위 Teacher와 TA의 distilled knowledge를 이후 작은 모델에 전달</strong>한다.

## 6.3. Loss
Teacher T, assistant $A_1, A_2$가 있다고 할때
##### A1
$$L_{A_1} = L_{T \rightarrow A_1}$$
##### A2
TAKD일때
$$A_1 \rightarrow A_2$$

DGKD일때
$$L_{A_2} = L_{T \rightarrow A_2} + L_{A_1 \rightarrow A_2}$$
##### Student
$$L_S = L_{T \rightarrow S} + L_{A_1 \rightarrow S} + L_{A_2 \rightarrow S}$$
##### 일반화
$$L_S = (n+1)(1-\lambda) L_{\text{CE}} + \lambda \left( L_{\text{KD}}^{T \rightarrow S} + \sum_{i=1}^{n} L_{\text{KD}}^{A_i \rightarrow S} \right)$$
DGKD는 단순한 앙상블이 아니다. 단순 앙상블은 독립적으로 학습된 여러 모델의 예측을 결합하지만, DGKD는 Teacher에서 파생되어 서로 다른 capacity를 가진 Teacher와 TA들의 knowledge를 함께 사용한다.

<strong>DGKD에서 사용하는 TA들은 Teacher로부터 점진적인 distillation을 통해 학습된 모델들이다. 즉 capacity가 서로 다른 Teacher derived knowldege source를 이용한다.</strong>

## 6.4. Trainer들

<strong>complementary knowledge sources</strong>

Student입장에서
Large Teacher는
- knowledge 풍부
- capacity gap이큼

small TA는
- Student와 capcity 가까움
- mimic 하기 쉬움
- Teacher 보다는 knowledge 제한

```
Large Teacher
→ richer knowledge

Intermediate TA
→ medium complexity knowledge

Small TA
→ easier-to-mimic knowledge
```

## 6.5. Stochastic DGKD
Teacher와 여러 TA가 Student를 동시에 지도하면, <strong>complex teacher group의 logit distribution으로 인해 Student가 overfit할 가능성</strong>이 있다고 제기한다. Student를 학습할 때 mini-batch마다 Teacher와 TA에서 Student로 연결되는 일부 knowledge connection을 확률적으로 제거한다.

<strong>베르누이 variable</strong>을 통해 관리한다.

단, <strong>최종 Student를 가르칠 때만 적용한다.</strong>

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/0cadba88512be2fd46fbcf25e4c84528.png)
## 6.6. Experiment
TAKD와 동일한 세팅을 사용한다.

#### Plain CNN

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/b557011dd51d1cf8c13d6eec4d354c02.png)

#### ResNet

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/90546cd2476dd56bd1bf4ce4d438e9af.png)

#### ImageNet에서의 비교

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/6a65a4aa05d7cd669039611e4d119cc4.png)

#### 앙상블과 비교

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/5c27a91e612fd0d481c1b5e9da5ff60e.png)

#### Error Avalanche Problem

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/a9443e7deeb780c2549cf14e0f639031.png)   
그림 4. 에러 중첩률은 Teacher T10, Student S2, 그리고 세 개의 TA(예: A8, A6, A4)가 있을 때 상위 레벨 모델의 오답과 하위 레벨 모델의 오답이 교차하는 비율을 나타낸다. $(E_i \cap E_j)$는 CIFAR-100 작업에서 i번째 plain CNN 모델의 에러 예시를 의미g한다.
#### t-SNE

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/05990b50829fbb51b1dbfd8267934f65.png)   
그림 5. CIFAR-10 데이터셋에서 ResNet을 사용한 (a) T26 → A20에 대한 KD, (b) A20 → S14에 대한 TAKD, (c) A20 → S14에 대한 본 논문의 DGKD의 t-SNE 시각화 결과. 빨간색 상자 안의 클래스 분포를 살펴보면, (b) TAKD와 (c) 본 논문의 DGKD에서 나타나는 서로 다른 에러 누적률을 확인할 수 있다.

#### TA를 늘리면?

![](/assets/img/posts/20260904_1044_KDKNOWLEDGE/63cdcaf309ac91bd5aeec8e4a3aeec73.png)   

TA를 늘리고 distillation path를 깊게 할수록 DGKD의 성능이 향상된다. 즉 이는 <strong>TAKD의 serial error accumulation 때문에 많은 TA를 활용하지 못하는 것을 개선하였다.</strong>

#### 핵심 결과

- CIFAR-100, `T10 → A8 → A6 → A4 → S2` : TAKD **45.14 → DGKD 48.92**
- CIFAR-10, `T26 → A20 → A14 → S8` : TAKD **88.01 → DGKD 89.02**
- TA를 7개까지 늘렸을 때 : TAKD **44.07**로 감소하지만 DGKD는 **49.56**까지 증가

즉 DGKD의 핵심은 단순 SOTA 성능보다 <strong>distillation path가 깊어질수록 TAKD에서는 error accumulation이 문제가 되지만 DGKD에서는 오히려 여러 TA를 더 잘 활용할 수 있었다는 점</strong>이다.