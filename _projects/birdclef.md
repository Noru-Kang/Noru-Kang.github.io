---
title: "Flagship 2 - BirdCLEF 2024"
description: Audio classification baseline -> 확장 프로젝트
priority: 2
tags: [audio, DL]
---

## Summary

182종 조류 음성(OGG)을 자동 분류하는 pipeline을 직접 구성한 프로젝트.
OGG를 Mel-Spectrogram(128×384)으로 변환한 뒤, existing pretrained model(EfficientNetV2-B2, ImageNet)을 backbone으로 사용해 분류를 수행했다. 제출 환경이 Kaggle CPU-only, 인터넷 차단, 2시간 런타임 제약이었기 때문에, 5초 프레임 단위 추론과 tf.data 기반 inference pipeline도 함께 구성했다.

PyTorch(학습)에서 Keras/TensorFlow(추론)로 프레임워크 이식을 직접 수행했고, Conformer 이식도 시도했다. 실험에서는 EfficientNetV2-B2(pretrained, 9.2M)가 B3(no pretrain, 12M)보다 더 잘 작동했고, 이를 통해 파라미터 수보다 사전학습이 더 중요할 수 있다는 점을 확인했다.

이 프로젝트는 오디오 신호를 representation으로 다룬 첫 경험이었고, 이후 ECG/EEG로 관심이 확장되는 출발점이 됐다.

Private ROC-AUC: 0.5880 | 643위 / 974팀  
**Stack:** PyTorch, Keras/TensorFlow, torchaudio, librosa, tf.data  
**Role:** 프레임워크 이식, 추론 pipeline 구성, 환경 제약 대응, 보고서 작성

## Link

[Detail Post](/posts/BirdCLEF-2024/)
