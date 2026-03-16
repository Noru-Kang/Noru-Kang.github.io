---
title: Sleep-EDF Expanded
date: 2026-03-15 18:00:00 +0900
categories:
  - etc.
  - DataSet
tags:
  - BIO
  - PSG
math: true
---
# Sleep-EDF Database Expanded

> https://physionet.org/content/sleep-edfx/1.0.0/

<strong>가장 전통적이고 많이 쓰이는 공개 sleep PSG benchmark 중 하나</strong>

---
## ℹ️ Data Info

| 항목         | 내용                                                                                                  |
| ---------- | --------------------------------------------------------------------------------------------------- |
| 샘플 수       | 197 whole-night PSG                                                                                 |
| 하위 구성      | SC 153, ST 44                                                                                       |
| 기록 길이      | SC 약 20시간, ST 약 9시간                                                                                 |
| 주요 채널      | EEG(Fpz-Cz, Pz-Oz), horizontal EOG, chin EMG, event marker, 일부 SC에는 respiration/body temperature 포함 |
| 샘플링 주파수    | EEG/EOG 100 Hz, SC의 EMG envelope 1 Hz, airflow/temp/marker도 1 Hz, ST EEG/EOG/EMG 100 Hz             |
| 라벨         | manual hypnogram                                                                                    |
| scoring 기준 | Rechtschaffen and Kales                                                                             |

---
## 🔔 수집 목적

<strong>automatic sleep staging algorithm이 서로 다른 데이터베이스에서도 generalize되는지 평가</strong>를 위함   
patient phenotype을 최대한 일반적이고 heterogeneous하게 유지하려고 했고, 추가적인 선택 기준을 거의 두지 않음, <strong>실제 의학적 분포를 반영한 데이터에 가까움</strong>

---
## 🗂️ 데이터 특징

#### Sleep Cassette Study(SC)

- 건강한 성인에서 연령이 수면 EEG에 미치는 영향을 연구로서 home recording됨
- 두 번의 day-night period 기록
- 건강한 Caucasian(백인)성인 대상

#### Sleep Telemetry Study
- temazepam(수면제)이 수면에 미치는 영향을 연구로서 hospital recording됨
- mild difficulty falling asleep가 있는 대상
- placebo night와 temazepam night 비교

---
## 📌 Sleep-EDF Expanded의 특징

- EEG/EOG/EMG 중심의 작은 montage
- SC와 ST가 서로 다른 연구 설계에서 왔기 때문에 내부 이질성이 있음
- 일부 채널은 1 Hz라 unified pipeline에서 주의 필요