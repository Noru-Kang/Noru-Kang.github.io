---
title: "Supporting 1 - EEGAAD"
description: EEG Auditory Attention Decoding
priority: 4
tags: [EEG, DL]
---

## Summary

EEG 기반 Auditory Attention Decoding 과제에서 기존 EEG foundation model을 현재 태스크에 이식하고 finetuning 전략을 조정한 내부 연구 트랙이다. CBraMod pretrained weight를 불러온 뒤, 128 Hz 입력과 200 Hz pretraining 조건 사이의 불일치를 해결하기 위해 TemporalResampler를 직접 설계했고, task-specific classifier head를 다시 구성했다. 학습은 classifier-only cold start 이후 상위 encoder layer만 선택적으로 여는 2-phase finetuning으로 진행했으며, unfreezing 비율(10/30/50)을 비교해 이식 조건을 확인했다. Task 2에서는 audio-only와 audio-visual 조건을 명시적으로 분리해 domain adaptation 문제로 다시 정의하고, CORAL loss, Contrastive loss, Mixup, entropy minimization 적용 방향을 정리했다.

## Link

[Detail Post](/posts/eegaad/)
