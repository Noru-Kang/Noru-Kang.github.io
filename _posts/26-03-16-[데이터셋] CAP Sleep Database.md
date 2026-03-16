---
title: CAP Sleep Database
date: 2026-03-15 16:00:00 +0900
categories:
  - etc.
  - DataSet
tags:
  - PSG
  - BIO
math: true
---
# 🧢 CAP Sleep Database

> https://physionet.org/content/capslpdb/1.0.0/

NREM 수면의 미세구조(microstructure)인 CAP를 분석하기 위한 병리 중심 PSG 데이터셋

> #### CAP(Cyclic Alternating Pattern)   
> - <strong>Macro</strong> : Stage1~3으로 구분하며, <strong>30초 단위</strong>로 <strong>"이 사람은 깊은잠을 자는가"</strong>에 대해 notation   
> - <strong>Micro</strong> : <strong>30초짜리 stage안의 초 단위의 변화</strong>. e.g. 깊은잠(Stage 3)을 자는 중에도, 뇌가 계속 외부의 자극에 반응하고 있다는 내용을 보여주는 지표   
> ##### CAP   
> NREM 수면 중에 나타나는 <strong>주기적인 뇌파의 리듬</strong>
> - Phase A (Activation) : 뇌파가 빨라지거나 커지는 시점으로 뇌가 깨거나, 깨려고 하는 징후
> - Phase B (Deactivation) : 다시 원래의 상태로 돌아감
> A와 B가 한쌍으로 묶인게 CAP Cycle, 이게 반복되면 CAP Sequence   
>   
> 정상적인 사람도 CAP상태가 나타나지만, <strong>잠이 불편하거나, 병에 걸리면 이 상태가 더 빈번하게 일어남</strong>
{: .prompt-info }

---
## ℹ️ Data Info

| 항목          | 내용                                                                                               |
| ------------- | -------------------------------------------------------------------------------------------------- |
| 샘플 수       | 108 recordings                                                                                     |
| 대상 구성     | 16 healthy + 92 pathological (NFLE 40, RBD 22, PLM 10, insomnia 9, narcolepsy 5, SDB 4, bruxism 2) |
| 기록 길이     | whole-night PSG                                                                                    |
| 주요 채널     | 최소 3 EEG, 2 EOG, chin EMG, bilateral tibial EMG, airflow, abdominal/thoracic effort, SaO2, ECG   |
| 샘플링 주파수 | 페이지에 단일 고정 Hz 미명시 → EDF header 단위 확인 필요                                           |
| 라벨          | sleep stages + CAP phase A                                                                         |
| stage 체계    | W, S1, S2, S3, S4, REM, MT                                                                         |

---
## 🔔 수집 목적

이 데이터셋의 핵심 목적은 <strong>CAP(Cyclic Alternating Pattern)를 정량화</strong>하고, 이것이 수<strong>면 불안정성(sleep instability)과 수면 관련 병리와 어떤 관련이 있는지 평가</strong>

---
## 🗂️ CAP Subtype

Phase Subtype
- A1: 동기화된 EEG event 중심, autonomic / somatomotor 영향이 낮음
- A2: 동기화 + 비동기화가 섞인 mixed pattern, 영향이 중간 정도
- A3: 비동기화된 EEG event가 우세, autonomic / somatomotor 영향이 큼

> #### Autonomic
> 자율신경계   
> 깊은 잠을 잘때 자율신경계는 안정화됨, 하지만 PhaseA가 발생하면 자율신경계가 각성할 수 있음   
> 
> <strong>지표</strong>   
> - 심박수 급상승
> - 호흡 변화
> - 식은땀 등
>ECG, PPG 데이터에서 변동성이 크게 나타난 경우
{: .prompt-info }

> #### Somatomotor
> 체성운동계   
> 근육의 움직임과 관련된 시스템, 뇌가 깨려고 할 때 근육에 긴장이 들어감   
>   
><strong>지표</strong>
>- 턱 근육 긴장도 상승(Chin EMG)
>- 팔다리 움츠림(Tibial EMG)
>- 눈동자 움직임(EOG) 등
>EMG가 채널이 튀는 경우
{: .prompt-info }

### CAP 증가가 보고된 병리 Case

- sleep-disordered breathing
- insomnia
- periodic leg movements
- restless legs syndrome
- REM behavior disorder
- nocturnal frontal lobe epilepsy
- narcolepsy

---
## 📌 CAP의 특징

- tage label도 있으므로 macrostructure와 microstructure를 같이 다룰 수 있음

---
## 📍 Info

- `.edf` : PSG신호 파일(EEG, EOG, chin EMG, bilateral tibial EMG, airflow, 흉부/복부 effort, SaO2, ECG 등 실제 파형)
- `.txt` : annotation 텍스트 파일.scoring 결과는 REMlogic report 형식의 텍스트 파일
- `.edf.st`는 PhysioBank-compatible annotation 파일. .txt와 같은 정보를 annotation 전용 포맷으로 담고 있으며, 시간은 PhysioBank annotation 규칙으로 인코딩되고, 각 annotation의 aux string 안에 event, duration, sleep stage, location, 그리고 가능한 경우 body position까지 포함