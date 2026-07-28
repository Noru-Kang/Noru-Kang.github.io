---
title: HMC Sleep Staging Database
date: 2026-03-15 18:00:00 +0900
categories:
  - etc.
  - DataSet
tags:
  - BIO
  - PSG
math: true
---
# Haaglanden Medisch Centrum sleep staging database

> https://physionet.org/content/hmc-sleep-staging/1.1/

실제 임상 PSG를 기반으로 한, <strong>표준적인 sleep staging 중심</strong> 데이터셋

---
## ℹ️ Data Info

| 항목          | 내용                                                     |
| ------------- | -------------------------------------------------------- |
| 샘플 수       | 151 whole-night PSG                                      |
| 대상 구성     | 85 male, 66 female, mean age 53.9 ± 15.4                 |
| 기록 길이     | whole-night PSG                                          |
| 주요 채널     | 4 EEG (F4/M1, C4/M1, O2/M1, C3/M2), 2 EOG, chin EMG, ECG |
| 샘플링 주파수 | 모든 신호 256 Hz                                         |
| 라벨          | hypnogram + lights off/on markers                        |
| scoring 기준  | AASM v2.4 manual scoring                                 |

> #### The AASM Manual for the Scoring of Sleep and Associated Events(AASM v2.4 manual scoring)
> 수면 데이터 라벨 가이드라인   
> 
> <strong>특징</strong>
> - Stage별 <strong>30초 epoch</strong>원칙
> - 수면 분류 : 기존의 R&K 방식(S1~S4)에서 S3와 S4를 통합하여 N3로 정의
> 	- W (Wake): 깨어 있음
> 	- N1 (NREM 1): 얕은 잠 (졸음)
> 	- N2 (NREM 2): 본격적인 수면 (K-complex, Spindle 출현)
> 	- N3 (NREM 3): 깊은 잠 (Slow Wave Sleep)
> 	- R (REM): 꿈 수면
> 
> <strong>이벤트</strong>
> - Micro-arousa : 3초 이상 15초 미만의 뇌파 변화. (CAP의 Phase A와 밀접함)
> - Respiratory : 무호흡(Apnea)은 최소 10초 이상 지속되어야 함.
> - PLM : 다리 움직임이 0.5초~10초 사이여야 함.
{: .prompt-info }


---
## 🔔 수집 목적

<strong>automatic sleep staging algorithm이 서로 다른 데이터베이스에서도 generalize되는지 평가</strong>를 위함   
patient phenotype을 최대한 일반적이고 heterogeneous하게 유지하려고 했고, 추가적인 선택 기준을 거의 두지 않음, <strong>실제 의학적 분포를 반영한 데이터에 가까움</strong>

---
## 🗂️ 데이터 특징

- sleep center 수집 데이터
- in-hospital / ambulatory recording

> #### in-hospital
> 병원의 수면검사실에서 측정   
> - <strong>고품질</strong> : 노이즈가 적고 신호가 매우 깨끗함
> - <strong>First Night Effect</strong> : 병원이라는 낯선 환경 때문 왜곡 가능성이 있음
{: .prompt-info }

> #### ambulatory recording
> 자신의 집에서 측정    
> 근육의 움직임과 관련된 시스템, 뇌가 깨려고 할 때 근육에 긴장이 들어감   
> - <strong>생태적 타당성(Ecological Validity)</strong> : 실제 환자의 '진짜 수면 습관'이 그대로 데이터에 반영
> - <strong>노이즈(Noise)</strong> : 센서 탈락, 뒤척임으로 인한 아티팩트(Artifact), 환경 소음 등이 데이터에 섞일 확률이 높음
{: .prompt-info }

- <strong>256Hz</strong>
- 유효 scoring interval만 남도록 lights off ~ lights on 구간으로 clipping됨
- Stage를 위한 데이터셋이므로, 라벨이 풍부하지 않음
---
## 📌 HMC의 특징

- 현실적인 환자 분포
- Sampling Rate 통일
- sleep stagging에 최적화된 annotation
- 실제 임상
- <strong>Pretrain용으로 좋음</strong>

---
## 📍 Info

- `.edf` : PSG 신호 파일. EEG, EOG, chin EMG, ECG 등
- `SNXXX_sleepscoring.edf` : annotation-only EDF+ 파일. hypnogram annotation과 lights-off/on marker가 저장. EDF+ 표준 텍스트와 polarity rule을 따릅니다. 즉, stage label과 수면 유효 구간을 EDF 계열 포맷 안에서 같이 다룸
- `SNXXX_sleepscoring.txt` : 동일한 scoring 정보를 CSV 형태 텍스트로 제공하는 파일.

> #### Hypnogram
> - X(Time) : 보통 전체 수면 시간(6~8시간).
> - Y(Labels): 수면 단계 (W, N1, N2, N3, REM, etc).
{: .prompt-info }
	