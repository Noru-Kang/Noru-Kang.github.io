---
title: "[2024-1 프로젝트] Time-Series Deep Learning — RNN / LSTM / GRU / Transformer를 활용한 KODEX 200 ETF 주가 예측"
date: 2026-02-24 22:00:40 +0900
categories: [AI-ML-DL, etc.]
tags: [프로젝트, 2024-1, RNN, LSTM, Transformer, GRU, finance, time-series]
---

# Time-Series Deep Learning — RNN / LSTM / GRU / Transformer를 활용한 KODEX 200 ETF 주가 예측

---

## 프로젝트 기본

- **Project Title:** Time-Series Deep Learning — RNN / LSTM / GRU / Transformer를 활용한 KODEX 200 ETF 종가 예측
- **One-line summary:** 한국 대표 ETF인 KODEX 200의 다음 날 종가를 시계열 딥러닝 4개 모델(RNN · LSTM · GRU · Transformer)로 예측하고 성능을 비교한다.
- **Project Type:** DL (Deep Learning) / Data Science
- **My Role / Key Contribution:**
  - FinanceDataReader를 통해 KODEX 200(069500) 전체 상장 기간 OHLCV 데이터를 수집하고 RobustScaler로 전처리를 직접 설계함.
  - RNN · LSTM · GRU · Transformer 4가지 시계열 모델을 동일 파이프라인 위에서 직접 구현하여 R² 기반의 공정한 비교 실험을 수행함.
  - 수업에서 배운 시계열 이론(ADF 정상성 검정, ACF/PACF, Prophet 분해)을 실제 금융 데이터에 적용하며 이론과 실습의 연결을 직접 검증함.
  - 세미나 발표자료(강태영.pdf)를 별도 제작하여 모델 구조/결과/해석을 발표함.

---

## TL;DR

- **Problem:** KODEX 200 ETF는 대표적인 비정상 시계열 자산가격이다. 과거 OHLCV 10일치로 내일 종가를 예측할 수 있을까? 그리고 어떤 시계열 딥러닝 아키텍처가 가장 적합한가?
- **Approach:** OHLCV 데이터를 RobustScaler로 정규화하고, window_size=10의 슬라이딩 윈도우로 입력을 구성한 뒤 RNN · LSTM · GRU · Transformer를 동일 설정(Adam / MSE / EarlyStopping / 60:20:20 분할)으로 학습해 R² Score로 비교한다.
- **Main Result:** 추가 필요: 실행 결과 R² 수치가 코드 출력 셀에 저장되어 있으나 현재 미실행 상태. `r2_y_predict_LSTM`, `r2_y_predict_RNN`, `r2_y_predict_GRU`, `r2_y_predict_Transformer` 변수에 기록됨.
- **Keywords:** `LSTM` `RNN` `GRU` `Transformer` `KODEX200` `ETF` `시계열 예측` `RobustScaler`

---

## Motivation & Background

- **Background:** 주가 예측은 금융 딥러닝의 가장 고전적인 응용 영역 중 하나다. KODEX 200은 코스피 200 지수를 추종하는 국내 최대 규모 ETF로, 한국 주식시장 전체의 흐름을 대표한다. 수업에서 배운 RNN 계열 시계열 모델들을 실제 금융 데이터에 적용함으로써 이론을 실전으로 연결하는 것이 이 프로젝트의 출발점이다.<br>발표자료(슬라이드 7)에서 KODEX 200을 선택한 이유를 세 가지로 명시했다: ① 금융 도메인 지식이 없는 초보자도 접근 가능, ② 자산 규모가 크기 때문에 오로지 숫자(가격)만으로 판단하기 용이, ③ 우리나라를 대표하는 주식들로 구성되어 한국 시장 전체를 반영.
- **Why this problem matters:** 비정상 시계열(ADF 검정 p-value=0.756)인 주가 데이터를 딥러닝으로 예측하는 것은 "모델이 단기 패턴을 얼마나 포착하는가"를 직접 검증할 수 있는 도전적인 과제다. 또한 단일 모델이 아닌 4개 아키텍처를 동등 조건 비교함으로써 각 모델의 시계열 표현 능력 차이를 실증적으로 확인할 수 있다.
- **Gap in existing work:** 단순 종가만 입력으로 사용하는 기존 튜토리얼과 달리, 본 프로젝트는 Open / High / Low / Volume 4개 특징을 함께 입력으로 사용해 다변량 시계열 예측을 시도한다. 또한 비교적 최근에 등장한 Transformer까지 RNN 계열과 동일 조건에서 비교한다.
- **Related work:**
  - Chollet, F., 「케라스 창시자에게 배우는 딥러닝 개정 2판」: 신경망 기초, RobustScaler 개념, GRU 구조 설명의 주요 참고 교재 (슬라이드 11, 12, 22, 27).
  - Prophet (Meta, 2017): 계절성 분해 기반 시계열 예측. 본 프로젝트에서는 탐색적 EDA 목적으로 365일 예측을 시도함.
  - Vaswani et al., "Attention is All You Need" (NeurIPS 2017): Transformer 아키텍처의 원작. 본 프로젝트의 Transformer 블록 구현 기반. 슬라이드 28에서 "GPT4의 기반"으로 소개.
  - `FinanceDataReader` 라이브러리: 한국·미국 주식 데이터, 암호화폐 가격 등을 제공하는 Python API 패키지 (슬라이드 6).

---

## Approach

### ML/DL 관련

- **Model/Architecture:**

  | 모델        | 층 구조                                                                                                          | 입력           | 출력       |
  | ----------- | ---------------------------------------------------------------------------------------------------------------- | -------------- | ---------- |
  | LSTM        | LSTM(20, relu) → Dropout(0.2) → LSTM(20, relu) → Dropout(0.2) → Dense(1)                                         | (batch, 10, 4) | (batch, 1) |
  | RNN         | SimpleRNN(20, relu) → Dropout(0.1) → SimpleRNN(20, relu) → Dropout(0.1) → Dense(1)                               | (batch, 10, 4) | (batch, 1) |
  | GRU         | GRU(20, relu) → Dropout(0.1) → GRU(20, relu) → Dropout(0.1) → Dense(1)                                           | (batch, 10, 4) | (batch, 1) |
  | Transformer | MultiHeadAttention(heads=4, key_dim=64) → Dropout → LayerNorm → FFN(64) → LayerNorm → GlobalAvgPool1D → Dense(1) | (batch, 10, 4) | (batch, 1) |

  **각 모델 선택 근거 (슬라이드 25~28):**
  - **RNN:** 과거의 정보를 사용하여 현재/미래 입력에 대한 성능을 개선하는 딥러닝 구조. 은닉상태와 루프를 통해 과거 정보를 저장. 단, 장기 의존성 학습에 한계(vanishing gradient).
  - **LSTM:** RNN의 장기 종속성 학습 문제를 해결. 부가적인 게이트(forget / input / output gate)로 은닉 셀의 정보를 제어. 시계열 데이터의 양방향 종속성 학습 가능. 전체 시계열 학습에 유용.
  - **GRU:** LSTM을 개선한 모델. 두 개의 게이트(reset / update gate)로 계산 효율 달성. 빠른 학습 시간 및 낮은 계산 복잡성.
  - **Transformer:** 구글에서 개발. seq2seq 구조(인코더·디코더). Attention만으로 구현하여 RNN을 전혀 사용하지 않지만 우수한 성능. GPT-4의 기반 아키텍처.

  - 입력: 과거 **10일**의 [Open, High, Low, Volume] 4차원 특징 벡터
  - 출력: 다음 날 **종가(Close)** 1개 값 (회귀)

- **Loss & Optimization:**
  - Loss: MSE (Mean Squared Error)
  - Optimizer: Adam (기본 설정)

- **Training strategy / key mechanisms:**
  - EarlyStopping: `patience=10`, `restore_best_weights=True`
  - ModelCheckpoint: `weights.h5`에 `val_loss` 기준 최적 가중치 저장
  - Max Epochs: 100 / Batch size: 30
  - Validation data: `(val_x, val_y)` 를 fit 시 실시간 모니터링

- **Inference/Serving path:**
  - 모델 예측 → Scaled 종가 → 스케일 역변환 없이, 실제 마지막 종가 기준 비율 환산으로 내일 예측가(KRW)를 출력:
    ```python
    predLSTM = df.Close[-1] * pred_y[-1] / dfy.Close[-1]
    ```

- **Ablation/Design choices:**
  - LSTM은 Dropout(0.2), RNN/GRU는 Dropout(0.1) — LSTM의 게이트 구조가 더 복잡하므로 보다 강한 정규화를 적용.
  - RobustScaler 선택: 금융 데이터 특유의 급등락(이상치)에 MinMaxScaler보다 강건한 스케일러 선택.
  - window_size=10: 2주 영업일 기준. 단기 패턴 포착에 초점.

### Algorithm/Statistics 관련

해당 없음

### Data Mining/Science 관련

해당 없음

### System/Pipeline 관련

해당 없음

---

## Data & Experiment

- **Dataset type:** 정형 금융 시계열 데이터 (일별 OHLCV)
- **Source:** [FinanceDataReader](https://github.com/FinanceData/FinanceDataReader) — `fdr.DataReader('069500')`로 수집. KRX 상장 데이터 기반.
- **Size:** 추가 필요: 코드 출력 셀 미실행 상태. `df.shape` 셀로 확인 가능. KODEX 200 상장일(2002-10-14) 기준 약 5,000+ 거래일로 추정.
- **Label/Target definition:** `Close` (당일 종가). 슬라이딩 윈도우 기준으로 과거 10일 OHLV → 다음 날 종가 1개를 예측하는 **단기 one-step ahead 회귀** 문제.
- **Preprocessing:**
  1. X (입력): `Open`, `High`, `Low`, `Volume` 4개 컬럼 분리
  2. Y (타깃): `Close` 1개 컬럼 분리
  3. **RobustScaler** 스케일링 (슬라이드 22 근거):
     - 중앙값(median)을 0으로, IQR(사분위수 범위)을 1로 변환
     - 이상치(급등락 이벤트)의 영향이 MinMaxScaler 대비 적음
     - X, Y 각각 독립적으로 `fit_transform` 적용
  4. **슬라이딩 윈도우** 생성: `window_size=10` (2주 영업일 기준), 총 샘플 수 = `len(data) - 10`
  5. ADF 정상성 검정 → p-value=0.756, 비정상 시계열 확인 (모델이 추세/비정상성을 스스로 학습해야 함을 명시)
- **Leakage checks:** 슬라이딩 윈도우 구성 시 타깃 시점(`i + window_size`)을 입력 범위(`i : i+window_size`)에서 철저히 분리하여 미래 누수 방지. 단, 스케일러를 전체 데이터 기준으로 `fit_transform`했으므로 엄밀한 의미의 Train-only fit은 미적용 — 미래 정보 미세 누수 가능성 있음.
- **Split:** 순서 유지(시간축 기준)
  - Train: 60% (`train_size = int(len(data_y) * 0.6)`)
  - Test: 20%
  - Validation: 20% (나머지 전체)
- **Evaluation protocol:** Test 세트에 대해 모델 예측 후 R² Score 계산. 4개 모델 동일 분할로 비교.
- **Metrics:**
  - **R² Score (결정계수):** `sklearn.metrics.r2_score` 사용. 예측값이 실제 분산을 얼마나 설명하는지 측정. 1에 가까울수록 좋고, 0 이하면 평균 예측보다 못함을 의미.
  - (참고) Loss 함수로 MSE 사용, 단 최종 비교는 R²로 수행.
- **Environment:** Google Colab (`!pip install` 패턴, Colab 특유의 출력 mime type으로 확인), Python 3.x
- **Frameworks/Libraries:**
  ```
  tensorflow / keras        # LSTM, RNN, GRU, Transformer, Dense, Dropout
  FinanceDataReader (fdr)   # KODEX 200 데이터 수집 (한국/미국 주식, 암호화폐 지원)
  scikit-learn              # RobustScaler, r2_score
  pandas / numpy            # 데이터 처리
  matplotlib / seaborn      # 시각화 (히트맵, 예측 그래프)
  plotly (==5.11.0)         # 인터랙티브 Line Plot
  kaleido                   # Plotly 정적 이미지 내보내기
  statsmodels               # ADF 정상성 검정, ACF/PACF
  scipy                     # kstest
  prophet                   # 시계열 분해 예측 (EDA 목적)
  pystan                    # Prophet 의존성
  ```
- **Reproducibility:** 추가 필요: 코드 내 random seed(`np.random.seed`, `tf.random.set_seed`) 설정 없음 → 실행마다 결과 소폭 상이할 수 있음.

---

## Results

**R² Score 비교 (Test Set, 노트북 저장 출력값 기준)**

| Model       | R² Score   | 순위  | 비고                                        |
| ----------- | ---------- | ----- | ------------------------------------------- |
| **GRU**     | **0.9839** | 🥇 1위 | 게이트 구조 + 계산 효율의 최적 조합         |
| LSTM        | 0.9690     | 🥈 2위 | 게이트 기반, Dropout 0.2 적용               |
| RNN         | 0.6601     | 🥉 3위 | vanishing gradient 영향                     |
| Transformer | 0.6568     | 4위   | 짧은 시퀀스(10일)에서 Attention 효과 제한적 |

**다음 날 종가 예측값 비교 (RobustScaled 공간 기준)**

> ⚠️ **주의 — 단위 버그:** 코드의 역변환 로직 `df.Close[-1] * pred_y[-1] / dfy.Close[-1]`에서 `df.Close[-1]`와 `dfy.Close[-1]`가 동일한 원본 종가이므로 비율이 1이 되어, 결과값은 실제 KRW가 아닌 **RobustScaled 예측값 그 자체**임. 실제 KRW 역변환을 위해서는 `rs.inverse_transform()` 적용 필요.

| Model       | 출력값 (Scaled) | 실제 KRW 역변환                               |
| ----------- | --------------- | --------------------------------------------- |
| LSTM        | 0.5237          | 추가 필요: `rs.inverse_transform()` 적용 필요 |
| RNN         | 0.2708          | 추가 필요: `rs.inverse_transform()` 적용 필요 |
| GRU         | 0.5057          | 추가 필요: `rs.inverse_transform()` 적용 필요 |
| Transformer | 0.5738          | 추가 필요: `rs.inverse_transform()` 적용 필요 |

- **Statistical significance / confidence:** 단일 실행 결과 비교이며 통계적 유의성 검정(반복 실험/CI)은 미수행.
- **Visualization notes:**
  - 각 모델별 Test Set 기간의 실제 종가(red)와 예측 종가(blue) 비교 Line Plot (`KODEX200 stock price prediction` 제목)
  - 상관관계 히트맵 (Open/High/Low/Close/Volume 간 상관명시, `annot=True`)
  - ACF / PACF 플롯 (자기상관 구조 확인)
  - Prophet 분해 플롯 (추세/주기 성분)
  - Plotly 인터랙티브 Line Plot (Open, High, Low, Close, Volume 각각)

---

## Discussion

- **Key observations:**
  1. ADF 검정(p=0.756) 결과 KODEX 200 종가는 명확한 **비정상 시계열**임이 확인되었다. 추세 제거 없이 딥러닝 모델이 이를 학습하기 때문에, 결과 해석 시 이 점을 고려해야 한다.
  2. **GRU(R²=0.9839)와 LSTM(R²=0.9690)은 매우 높은 R²를 기록**한 반면, RNN(0.6601)과 Transformer(0.6568)는 0.65~0.66 수준에 머물렀다. 게이트 메커니즘의 유무가 가장 큰 성능 분기점임이 확인되었다.
  3. Transformer는 시퀀스 길이 10의 짧은 입력에서 RNN(0.6601)보다 낮은 R²(0.6568)를 기록했다. Multi-Head Attention이 긴 시퀀스에 유리한 구조임을 시사한다.
  4. GRU가 LSTM보다 R² 기준 더 높은 성능을 보인 점이 인상적이다. 파라미터 수가 적음에도 성능이 더 좋아, KODEX 200 단기 시계열에서 GRU의 경량 게이트 구조가 더 효율적임을 보여준다.
  5. 예측가 출력 코드(`df.Close[-1] * pred_y[-1] / dfy.Close[-1]`)에 역변환 버그가 있어 결과가 KRW가 아닌 Scaled 값으로 출력됨. `rs.inverse_transform()` 미적용이 원인.
  > ⚠️ **투자 면책 고지 (슬라이드 54):** 이 프로젝트는 학습/세미나 발표 목적으로 제작되었습니다. "참고만! 투자시 소액만!" — 실제 투자에 직접 활용하지 마세요.
- **Interpretation:** RobustScaler를 통해 급등락 이상치의 영향을 줄인 것은 금융 데이터에 적합한 선택이다. window_size=10(2주치)으로 단기 패턴에 집중한 설계는 장기 추세보다 단기 모멘텀을 학습하는 전략이다. GRU와 LSTM의 높은 R²(0.98/0.97)는 OHLV 10일 시퀀스가 단기 종가 패턴 포착에 충분히 유효한 입력임을 보여준다.
- **Trade-offs:**
  - 단순성 vs 표현력: SimpleRNN은 파라미터 수가 적고 빠르지만 vanishing gradient 문제가 있음 → R² 0.66으로 게이트 모델 대비 큰 차이.
  - 모델 복잡도 vs 데이터 효율: Transformer는 파라미터 수가 많음에도 R² 0.6568로 가장 낮음 — 시퀀스 길이 10이 너무 짧아 Attention이 효과를 발휘하지 못한 것으로 판단.
  - Dropout 비율: LSTM(0.2) vs RNN/GRU(0.1) — 모델별 정규화 강도가 달라 순수 아키텍처 비교의 엄밀성이 다소 떨어짐.
- **Failure cases / surprising results:**
  - 예상과 달리 Transformer가 RNN보다 낮은 R²(0.6568 vs 0.6601)를 기록했다. 일반적으로 Transformer가 더 강력하다고 알려져 있으나, 짧은 시퀀스(10일)와 소규모 데이터에서는 RNN 계열이 더 효율적임을 보여준다.
  - 예측가 역변환 코드에 버그가 있어 KRW 대신 Scaled 값(0.27~0.57)이 출력되었다. 실제 서비스 활용 시 수정 필수.
- **What I learned:**
  1. 시계열 분석의 첫 단계는 **정상성 검정**이며, p-value를 직접 계산하고 해석하는 경험을 쌓았다.
  2. RNN 계열 4개 모델을 동일 조건에서 구현·비교함으로써 각 아키텍처의 구조적 차이(게이트 유무, Attention 메커니즘 유무)가 성능에 어떻게 연결되는지 실증적으로 이해했다.
  3. 금융 데이터 전처리에서 RobustScaler의 선택이 왜 MinMaxScaler보다 적합한지, 스케일러를 어떻게 역변환하는지 이해했다.

---

## Limitations & Future Work

- **Limitations:**
  1. **스케일러 누수 가능성:** `RobustScaler`를 전체 데이터(Train+Val+Test 포함)에 `fit_transform` 적용 → 미래 정보가 스케일링에 미세하게 반영됨. 엄밀하게는 Train 데이터로만 `fit` 후 Val/Test에 `transform` 적용해야 함.
  2. **단일 실행 비교:** Seed 미설정으로 재실행 시 결과가 달라질 수 있어 재현성이 낮음.
  3. **입력 특징 제한:** OHLV 4개만 사용. 거시경제 지표, 뉴스 감성 등 외부 요인 미반영.
  4. **window_size 고정:** 10일을 고정값으로 사용했으나, 최적 window size 탐색 실험은 미수행.
  5. **하이퍼파라미터 미세 탐색:** units=20, head_size=64 등이 단순 고정값이며 Grid Search/Bayesian Optimization 미적용.
  6. **예측가 역변환 버그:** `predLSTM = df.Close[-1] * pred_y[-1] / dfy.Close[-1]` 로직의 분자/분모가 같아 Scaled 값이 그대로 출력됨. `rs.inverse_transform(pred_y)` 로 수정 필요.

- **Future directions:**
  1. Train 데이터만으로 스케일러를 `fit`하는 올바른 파이프라인으로 수정 후 재실험.
  2. `np.random.seed` + `tf.random.set_seed` 고정으로 재현 가능한 실험 환경 구성.
  3. 기술적 지표(RSI, MACD, Bollinger Band 등) 추가 특징 설계.
  4. window_size 탐색 실험 및 다중 스텝 예측(multi-step forecasting) 확장.
  5. 다른 ETF(TIGER 미국S&P500, KODEX 레버리지 등)로 일반화 가능성 검증.

- **If I had more time:**
  - Walk-forward validation(시간축 롤링 검증)으로 시간에 따른 성능 분포 확인.
  - 모델 앙상블(GRU + LSTM) 시도 — R² 0.98 이상 달성 가능성 탐색.
  - Transformer에 더 긴 window_size(30~60일) 적용하여 Attention이 효과를 발휘할 수 있는 조건 재실험.
  - 예측가 역변환 버그 수정 후 실제 KRW 예측값 검증.

---

## Project Structure

```
LSTM 주가 예측/
├── HRD_Finace_deep_learning.ipynb   # 메인 실험 노트북 (엔트리포인트)
│                                     # 데이터 수집 → EDA → 전처리 → 모델링×4 → 비교
└── 해룡당 세미나(2024-1)/            # 세미나 발표자료 폴더
    ├── 강태영.pdf                    # 세미나 발표 PDF (최종본)
    ├── 강태영.pptx                   # 발표 원본 PPTX
    ├── LSTM.png / LSTM.jpeg          # LSTM 구조 설명 이미지
    ├── RNN.png                       # RNN 구조 설명 이미지
    ├── GRU.png                       # GRU 구조 설명 이미지
    ├── Transformer.png               # Transformer 구조 설명 이미지
    ├── 신경망.png / 딥러닝 구조.jpeg  # 딥러닝 개념 설명 이미지
    ├── 시계열 분석.png                # 시계열 이론 이미지
    ├── Finance Data Reader.png        # 데이터 수집 설명 이미지
    ├── 로버스트 스케일링.png           # 전처리 설명 이미지
    ├── 히트맵.png                    # 상관관계 EDA 결과 이미지
    └── PLOt.png / Plot22.png         # 예측 결과 시각화 이미지
```

---

## PDF/Slides Mapping

- **Main slide deck:** `해룡당 세미나(2024-1)/강태영.pptx` → `강태영.pdf` (PDF 변환본) — 2024년 상반기 해룡당 세미나 발표본. **총 55슬라이드, 9개 섹션.**

- **발표 섹션 구성 (9섹션):**

  | 섹션 번호 | 섹션 제목                                                | 슬라이드 번호 | README 섹션 연결                                        |
  | --------- | -------------------------------------------------------- | ------------- | ------------------------------------------------------- |
  | 01        | Data (FINANCE DATA READER / KODEX 200)                   | 5~7           | Data & Experiment → Source                              |
  | 02        | Pre Knowledge (AI·ML·DL / 신경망 / 딥러닝 모델 / 시계열) | 8~13          | Motivation & Background / Approach 배경                 |
  | 03        | EDA (상관 분석 / 시계열 분석)                            | 14~20         | Data & Experiment → 정상성/ADF, Results → Visualization |
  | 04        | Data Scaling & Introduce (RobustScaling / 모델 소개)     | 21~28         | Approach → 전처리 / 모델별 설명                         |
  | 05        | LSTM (Model · Compile · Fit · Predict · Plot)            | 29~33         | Approach → LSTM 결과                                    |
  | 06        | RNN (Model · Compile · Fit · Predict · Plot)             | 34~38         | Approach → RNN 결과                                     |
  | 07        | GRU (Model · Compile · Fit · Predict · Plot)             | 39~43         | Approach → GRU 결과                                     |
  | 08        | Transformer (Model · Compile · Fit · Predict · Plot)     | 44~49         | Approach → Transformer 결과                             |
  | 09        | Conclusion (R² / 예측값 / QnA)                           | 50~55         | Results / Discussion                                    |

- **Slide-to-README mapping (슬라이드 번호 확정):**

  | 슬라이드 | 내용                                                                                 | README 섹션                   |
  | -------- | ------------------------------------------------------------------------------------ | ----------------------------- |
  | 1        | 표지: "Time-Series Deep Learning BY TAE YOUNG"                                       | Project Title                 |
  | 2        | 목차 (9섹션 전체)                                                                    | 전체 구조                     |
  | 3        | 모델 라인업 소개 (LSTM / RNN / GRU / Transformer)                                    | Approach                      |
  | 4        | 결론 섹션 예고 (R squared / Prediction / QnA)                                        | Results                       |
  | 6        | FinanceDataReader 소개 (한국·미국 주식, 암호화폐, GitHub 링크)                       | Data & Experiment → Source    |
  | 7        | KODEX 200 선택 이유 3가지 (초보자 접근 가능 / 자산규모 큼 / 대표 ETF)                | Motivation → Why this problem |
  | 9        | AI / ML / DL 개념 정의                                                               | Background                    |
  | 10~11    | 신경망 구조 (뇌 영감, 수학 모델, 케라스 교재 그림)                                   | Approach 배경                 |
  | 12       | 딥러닝 모델 종류 (케라스 교재 그림)                                                  | Approach 배경                 |
  | 13       | 시계열 데이터 정의 (일정 시간 간격, 주가/날씨 예시)                                  | Data & Experiment             |
  | 15~16    | 상관 분석 (히트맵)                                                                   | Results → Visualization       |
  | 17~20    | 시계열 분석 (ADF, ACF/PACF, Prophet)                                                 | Data & Experiment → 정상성    |
  | 22~23    | RobustScaler 원리 (중앙값=0, IQR=1, 케라스 교재 근거)                                | Approach → 전처리             |
  | 24       | 용어 설명 (샘플/예측/타깃/Loss/units/activation/dropout/optimizer/epochs/batch_size) | Approach → Training strategy  |
  | 25       | LSTM 소개 (게이트 구조, 장기 종속성 해결)                                            | Approach → LSTM               |
  | 26       | RNN 소개 (은닉상태, 루프, 단기 의존성)                                               | Approach → RNN                |
  | 27       | GRU 소개 (LSTM 개선, 두 게이트, 계산 효율, 케라스 교재)                              | Approach → GRU                |
  | 28       | Transformer 소개 (구글 개발, Attention만으로, GPT-4 기반)                            | Approach → Transformer        |
  | 29~33    | LSTM 구현 전체 (모델링 / Compile / Fit / Predict / Plot)                             | Approach → LSTM 결과          |
  | 34~38    | RNN 구현 전체                                                                        | Approach → RNN 결과           |
  | 39~43    | GRU 구현 전체                                                                        | Approach → GRU 결과           |
  | 44~49    | Transformer 구현 전체                                                                | Approach → Transformer 결과   |
  | 51       | R² 비교 결과 — GRU 0.9839 / LSTM 0.9690 / RNN 0.6601 / Transformer 0.6568            | Results → R² Score 표         |
  | 52       | 예측값 비교 — Scaled 출력값 (역변환 버그 있음, 실제 KRW 아님)                        | Results → 예측 종가 표        |
  | 54       | QnA + 면책: "참고만! 투자시 소액만!"                                                 | Discussion → 주의사항         |
  | 55       | 마무리: "THX. HRD FINANCE"                                                           | —                             |

  - R² 수치는 슬라이드 51에 시각화. 노트북 저장 출력값에서 확정: GRU 0.9839 / LSTM 0.9690 / RNN 0.6601 / Transformer 0.6568.
- **Any missing slides / gaps:**
  - 슬라이드 15~20의 EDA 그래프 상관계수 수치 상세 → 히트맵 이미지 직접 확인 필요
  - 예측가 실제 KRW 수치 (역변환 버그로 Scaled 값 출력, `rs.inverse_transform()` 코드 수정 후 재확인 필요)

---

## Citation & License

- **Citation info:** 강태영, "Time-Series Deep Learning — RNN/LSTM/GRU/Transformer를 활용한 KODEX 200 주가 예측", 해룡당 세미나 2024-1, 2024.
- **License:** 추가 필요: 명시된 라이선스 없음.
- **Papers/links:**
  - Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), NeurIPS 2017 — Transformer 구현 기반
  - Hochreiter & Schmidhuber, ["Long Short-Term Memory"](https://www.bioinf.jku.at/publications/older/2604.pdf), Neural Computation 1997 — LSTM 원작
  - Cho et al., ["Learning Phrase Representations using RNN Encoder-Decoder"](https://arxiv.org/abs/1406.1078), EMNLP 2014 — GRU 원작
  - Taylor & Letham, ["Forecasting at Scale (Prophet)"](https://peerj.com/preprints/3190/), PeerJ 2017 — EDA용 시계열 분해
  - Chollet, F., 「케라스 창시자에게 배우는 딥러닝 개정 2판」, 길벗, 2022 — 발표자료 주요 참고 교재 (신경망, RobustScaler, GRU)
  - [FinanceDataReader GitHub](https://github.com/FinanceData/FinanceDataReader) — 데이터 수집 라이브러리
  - [KODEX 200 ETF 상세 정보 (삼성자산운용)](https://www.kodex.com/product_ETF.do?fId=2ETF26)
