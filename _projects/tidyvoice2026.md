---
title: "Supporting 2 - tidyvoice2026"
description: 최적화
priority: 5
tags: [audio, DL, Optimization]
---

## Summary

화자 검증 challenge에서 학습, 추론, 제출까지 이어지는 전체 파이프라인을 정리하고 병목을 줄이는 작업을 맡았다. 원본 코드의 DataLoader worker 고정(`2`)과 forward 내부 fbank 직렬 처리 구간을 각각 `worker 동적 조정 + pin_memory/persistent_workers`, `ThreadPoolExecutor` 병렬화로 바꿨고, 그 결과 학습 시간이 **epoch당 약 25분에서 15분대**로 줄었다. 추가로 S-Norm scoring, cohort 생성, checkpoint/resume, 제출 포맷 검증과 zip 패키징까지 정리해 실험 반복과 제출 안정성을 함께 개선했다.

## Link

[Detail Post](/posts/tidyvoice2026/)
