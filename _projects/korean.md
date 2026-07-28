---
title: "Flagship 3 - 한국어 학습자를 위한 노래 추천 시스템"
description: "한국어 학습을 위한 kpop추천"
priority: 3
tags: [NLP, Data-Science]
---

## Summary

TOPIK 어휘를 기준으로 K-POP 가사의 난이도를 자동 계산하고, 그 결과를 바탕으로 학습자 수준과 장르에 맞는 곡을 추천한 프로젝트다. 레이블이 없는 상황이었기 때문에 모델을 바로 학습시키기보다, `difficulty_level = Σ(TOPIK 급수) / 매칭 어간 수` 형태의 scoring function을 직접 설계했다. 형태소 분석은 MeCab과 Komoran을 함께 사용했고, 계산된 연속 점수는 IQR 기준으로 초급, 중급, 고급으로 나눴다. 추천은 Spotify 특성과 가사 파생변수에서 PCA로 고른 feature(`energy`, `english_ratio`, `korean_ratio`, `dps`)를 사용해 코사인 유사도로 구성했다. 나는 스코어링 알고리즘 설계 및 구현, 전처리, PCA feature selection, 발표를 맡았다.

## Link

[Detail Post](/posts/korean/)
