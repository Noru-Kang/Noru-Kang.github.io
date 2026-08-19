---
title: Paper Follow Up - 26.08.4주차
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
math : true
---
GPT를 이용하여 논문들 서칭 후 follow up을 진행하였습니다.

## 1. 이번 주 주요 이벤트

### Trend : 전체 흐름
<strong>모델 자체의 표현력을 키우는 것보다, 어떤 변이를 없애고 어떤 불확실성을 보존하며 어떤 조건에서 예측을 신뢰할지를 구조적으로 정의하는 연구가 많음</strong>

---
### A : 생체신호 / 의료 시계열
<strong>새로운 backbone 자체보다 실제 배포 시 발생하는 subject/session shift를 어떻게 처리할 것</strong>인가?
- `MRieHy` : <strong>cross-day shift를 Riemannian covariance alignment + deep-feature hypergraph + online test-time adaptation</strong>으로 직접 다룬다. 별도의 target label 없이 현재 세션의 구조를 지속적으로 흡수하는 방향

> ##### cross day shift
> 같은 사람의 신호더라도 <strong>측정 날짜가 바뀌면서 데이터 분포가 달라지는 현상</strong>으로 전극 위치, 피부 상태, 피로도, 장비 상태 등이 바뀌어 train $\neq$ test가 되는 현상
{: .prompt-info }


> ##### Riemannian covariance alignment
> EEG 채널 간 관계를 <strong>cov. matrix로 표현</strong>후 일반적인 유클리디안 공간이 아니라 <strong>SPD행렬의 Riemannian geometry</strong>에서 정렬하여 달라진 covariance의 기준점을 맞춰서 분포 차이를 줄임
{: .prompt-info }

- `ReliaGate` : 정확도 향상 ↔︎ <strong>현재 예측을 사용자에게 노출해도 되는가</strong>를 post-hoc reliability routing 문제로 바꾼다.

> ##### post-hoc reliability routing
> 이미 학습된 예측 모델은 그대로 두고, <strong>모델이 낸 예측을 믿을지 말지를 뒤에서 한 번 더 판단하는 장치</strong>로 confidence, 신호 품질, OOD정도 등을 보고 사용 및 보류를 결정
{: .prompt-info }

- `TransfHAR` : 많은 fine grained label을 요구하지 않고 <strong>coarse unlabeled wrist activity에서 transferable motion prior(손목 센서 데이터의 행동 라벨 말고, 거친 운동 패턴을 대량으로 학습하여 사람의 움직임의 일반적인 특징을 먼저 익힘)를 학습한 뒤 개인별 few-shot adaptation으로</strong>처리
---
### B : 오디오 / 음성
<strong>semantic information과 nuisance factor를 분리하는 representation 설계가 중심</strong>적으로 연구되고 있다.
- `EEG-to-speech` : 같은 음성 자극을 다른 session에서 기록한 EEG를 positive pair로 잡아 <strong>session-invariant representation</strong>을 만들고, 동시에 variational regularization을 사용하여 필요한 내용을 지나치게 제거하지 않도록 한다.

> ##### variational regularization
> representation을 자유롭게 만들지 않고 <strong>특정 확률분포 구조를 따르도록 제약</strong>하는 방법으로, VAE처럼 latent를 가우시안에 가깝게 만들기 위해 <strong>KL divergence를 추가</strong>하는 방법 등, 너무 세션 정보에 맞춰 과적합되는 것을 줄이는 용도로 사용한다.
{: .prompt-info }

- `Speaker-normalized token` : S2U와 T2U를 반복적으로 교대, speech token을 <strong>speaker identity보다 text-predictive information 쪽으로 정제</strong>한다. 
- `ASR` : LLM을 매 inference마다 호출하지 않고, 미리 계산한 next token probability를 캐싱하여 <strong>retrieval/backoff 문제</strong>로 바꾼 연구이다.
- `INSPIRE` : speech retrieval을 단순 의미 검색이 아니라 speaker, style, environmental sound 등을 포함하는 instruction-conditioned retrieval 문제로 확장한다.

---
### C : 관심 분야
<strong>모델의 크기보다 구조적 가정, uncertainty, auditability를 직접 넣는 연구</strong>위주
1. <strong>불확실성과 coverage</strong>
	- `GHCP` : 새로운 subject/site/group에서 <strong>일부 관측값만 확보</strong>했을 때 grouped data에 conformal guarantee를 어떻게 가져갈 것이가.
	- `CNQ` : survival distribution의 여러 quantile을 동시에 예측하면서 quantile crossing을 구조적으로 막는 구조

> ##### grouped data에 conformal guarantee를 어떻게 가져갈 것?
> 일반 conformal predicition은 smaple들이 대체로 불변이라고 가정한다. 그런데 의료 데이터는 환자안의 여러 epoch처럼 <strong>같은 그룹 내부의 sample들이 강하게 연관되어 있다.</strong> 따라서 개별 sampel이 아니라, <strong>subject/site 같은 그룹 구조를 고려하면서 coverage,를 유지하는 방법</strong>이 필요하다.
{: .prompt-info }

> ##### quantile crossing
> q_10 > q_50 예측처럼 <strong>quantile 순서가 뒤집히는 문제</strong>로서.  q_10 <= q_50 <= q_90이어야 하는데, 이론적인것과 다른 현상으로 여러 quantile을 독립적으로 학습하면 이런 현상이 생길 수 있음
{: .prompt-info }

2. <strong>강한 inductive bias를 넣은 시계열</strong>
	- `AsyTO` : 변수 간 attention을 늘리는 대신 <strong>공유 temporal mode + variable-specific gain + periodic prototype</strong>으로 다변량 예측을 factorize한다.
	- `TinyCast` : 적은 파라미터와 계산된 periodictiy를 사용해서 확률적인 zero-shot 예측을 구성한다. 코드, weight도 공개되어 있음
3. <strong>비딥러닝/통계적 ML의 재등장</strong>
	- `Source-Disjoint Tree Ensemble` : raw variable 하난가 여러 tree에 흩어져서 기여하는것을 제한하여 예측 경로 자체를 사람이 추적할 수 있게 설계
	- `Generalised Transportability` : 데이터 소스와 target causal model사이의 abstraction을 사용해 어떤 query가 옮겨질 수 있는지, 정확한 이동이 불가능할때 어느 범위까지 보장할 수 있는지를 다룬다.

> ##### target causal model
> <strong>목표 환경의 인과구조</strong>로서 우리가 예측을 실제로 적용하려는 곳이다. 즉 site A에서 배운 모델을 site B로 옮길때, A의 causal model이 source이고 B가 target causal model로 된다. 즉 데이터 분포가 다른지를 넘어 <strong>원인 → 결과 관계 중 무엇이 유지되고 무엇이 바뀌었는지를 본다.</strong>
{: .prompt-info }

---
### D : 주요 AI / DL / ML / RL
RL에서는 <strong>critic/value function을 버리는 방향만 추구하기보다, 더 좋은 privileged information을 value 쪽에 어떻게 넣을지</strong>가 다시 중요해지는 흐름이다.
- `Le Critique` : task-relevant token-level privileged information을 value function에 제공하면서 policy objective 자체를 왜곡하지 않도록 설계, value quality에 따라 baseline 사용 방식을 조절
- `SCALE` : JEPA/world model latent가 reconstruction에 좋다는 것만으로 planning에 좋은것은 아니라는 문제를 짚고, <strong>실제 state-space distance와 latent-space distance를 calibration한다.</strong>

> ##### JEPA(Joint embedding Predictive Architecture)
> 픽셀이나 raw observation 자체를 복원하기 보다, 현재 상태에서 <strong>미래의 representation/embedding을 예측</strong>하는 방식이다. world model에서 이런 latent representation을 사용하면 실제 환경을 압축된 latent 공간에서 예측 및 planning할 수 있다.
{: .prompt-info }

> ##### 실제 state-space distance와 latent-space distance를 calibration
> 실제 환경에서 두 상태가 가까우면 latent에서도 가깝고, 실제로 멀면 latent에서도 멀도록 만드는 것
{: .prompt-info }

- `Q-based Variational IRL` : inverse RL에서 단일 reward estimate 대신 optimal Q에 대한 variational distribution을 두어<strong> reward posterior와 epistemic uncertainty를 함께 얻는다</strong>.

> ##### Inverse RL
> 일반 RL은 <strong>reward를 알고 최적 행동을 찾는 것</strong>이지만, inverse RL은 <strong>전문가의 행동을 보고 그 사람이 어떤 reward를 알고 최적화했는지를 추론</strong>하는것으로, 사람이 운전하는 것을 보고 안전, 속도, 편안함 등에 어느 정도 가치를 두는지 reward function을 역으로 추정하는 것이다.
{: .prompt-info }

---

### E : 참고할 분야

---
---
## 3. 정독할 논문 정리


---
---
## 4. 상세 논문 정리

### A : 생체신호 / 의료 / 임상 시계열

---
### B : 오디오 / 음성 / 생체의료 오디오

---
### C : 관심 분야

---
### D : 주요 AI / DL / ML / RL


---
---
## 5. 아이디어

### 5.1. GPT 추천

---
### 5.2. 내 아이디어 정리


---
---
## 6. 결론

---