---
title: "Flagship 1 - Chagas Disease Detection from 12-lead ECG"
description: "12-lead ECG를 통한 Chagas 분류"
priority: 1
tags: ["PhysioNet2025", "CinC", "ECG"]
---

## Summary

12-lead ECG 기반 Chagas disease 탐지를 위해, 임상 기반 handcrafted feature와 deep sequence representation을 결합한 hybrid model을 설계했다. 1D ResNet-BiGRU backbone 위에 RBBB/LAFB 관련 feature(QRS duration, electrical axis, RSR' ratio, wide-S ratio)와 demographic feature를 single-query cross-attention으로 결합했고, source-aware weighted BCE loss를 이용해 label reliability가 다른 데이터 소스(SaMi-Trop, PTB-XL, CODE-15%)를 함께 학습했다. 또한 전처리 파이프라인의 discard 판단 시점을 앞당기고 일부 연산을 단순화해 속도를 개선했다. 추가로 RAG 기반 signal retrieval, graph model, TTA, LoRA 등도 탐색했다. 최종 hidden test score는 0.218이었고, 41개 eligible team 중 17위를 기록했다.

## Link

[Detail Post](/posts/Physionet2025/)
