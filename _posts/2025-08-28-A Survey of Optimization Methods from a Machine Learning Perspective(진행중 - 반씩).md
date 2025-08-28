---
title: "A Survey of Optimization Methods from a Machine Learning Perspective(진행중)"
date: 2025-08-28 13:00:00 +0900
categories: [AI-ML-DL, etc.]
tags: [paper, Computer-Vision, optimizer]
---

# A Survey of Optimization Methods from a Machine Learning Perspective

---

### 🔗 출처

> https://arxiv.org/abs/1906.06821
> 

---

### 📦 추가 자료

…

## 🧩 방법론

> **by …**
> 

<aside>

</aside>

---

# 📌 논문

## 💡 요약

> **by …**
> 

<aside>

</aside>

---

## 📚 정리

### 📌 제목

<aside>

## A Survey of Optimization Methods from a Machine Learning Perspective

</aside>

---

### 🌟 초록

### 번역

원문

Abstract—Machine learning develops rapidly, which has made many theoretical breakthroughs and is widely applied in various fields. Optimization, as an important part of machine learning, has attracted much attention of researchers. With the exponential growth of data amount and the increase of model complexity, optimization methods in machine learning face more and more challenges. A lot of work on solving optimization problems or improving optimization methods in machine learning has been proposed successively. The systematic retrospect and summary of the optimization methods from the perspective of machine learning are of great significance, which can offer guidance for both developments of optimization and machine learning research. In this paper, we first describe the optimization problems in machine learning. Then, we introduce the principles and progresses of commonly used optimization methods. Next, we summarize the applications and developments of optimization methods in some popular machine learning fields. Finally, we explore and give some challenges and open problems for the optimization in machine learning.

Index Terms—Machine learning, optimization method, deep neural network, reinforcement learning, approximate Bayesian inference.

---

번역문

초록—머신러닝(Machine Learning)은 빠르게 발전하며, 많은 이론적 돌파구를 만들었고 다양한 분야에 널리 적용되고 있다. 머신러닝의 중요한 구성 요소인 **최적화(Optimization)** 는 연구자들의 많은 관심을 받아왔다. 데이터 양의 기하급수적 증가와 모델 복잡성의 상승으로 인해, 머신러닝의 최적화 기법은 점점 더 많은 도전에 직면하고 있다. 이러한 문제를 해결하거나 최적화 방법을 개선하려는 많은 연구가 잇달아 제안되어 왔다. 머신러닝 관점에서 최적화 방법을 체계적으로 회고하고 정리하는 것은 최적화와 머신러닝 연구의 발전 모두에 중요한 지침을 제공한다. 본 논문에서는 먼저 머신러닝에서의 최적화 문제를 설명한다. 이후, 널리 사용되는 최적화 기법들의 원리와 발전을 소개한다. 다음으로, 인기 있는 머신러닝 분야에서 최적화 기법의 응용 및 발전을 정리한다. 마지막으로, 머신러닝 최적화에서의 **도전 과제(challenges)와 미해결 문제(open problems)** 를 탐구하고 제시한다.

색인어—머신러닝(Machine learning), 최적화 기법(Optimization method), 심층 신경망(Deep Neural Network), 강화학습(Reinforcement Learning), 근사 베이지안 추론(Approximate Bayesian Inference).

<aside>

# 0. Abstract

- 최적화 기법 서베이 논문
    - 도전 과제
    - 미해결 문제를 제시
</aside>

---

### 💡 결론 & 고찰

### 번역

1. 원문 (요약된 결론 및 고찰 부분)
    
    The development of optimization brings a lot of contributions to the progress of machine learning. However, there are still many challenges and open problems for optimization problems in machine learning.
    
    1. How to improve optimization performance with insufficient data in deep neural networks is a tricky problem…
    2. For sequential models, the samples are often truncated by batches when the sequence is too long, which will cause deviation…
    3. The stochastic variational inference is graceful and practical, and it is probably a good choice to develop methods of applying high-order gradient information…
    4. It may be a great idea to introduce the stochastic technique to the conjugate gradient method to obtain an elegant and powerful optimization algorithm…
    
    The purpose of this paper is to summarize and analyze classical and modern optimization methods from a machine learning perspective. Firstly, we describe the theoretical basis… Then we describe the applications… Finally, we discuss some challenges and open problems…
    

---

1. 번역문
    
    최적화(Optimization)의 발전은 머신러닝의 진보에 많은 기여를 해왔다. 그러나 여전히 머신러닝의 최적화 문제에는 많은 **도전 과제(challenges)와 미해결 문제(open problems)** 가 남아 있다.
    
    1. **데이터 부족 상황에서 심층 신경망(Deep Neural Network, DNN)의 최적화 성능을 향상**시키는 것은 어려운 문제이다. 샘플 수가 충분하지 않으면 고분산(high variance)과 과적합(overfitting)이 발생하기 쉽다. 또한, DNN 학습에서 비볼록(non-convex) 최적화 문제는 전역 최적해(global optimum)가 아닌 국소 최적해(local optimum)에 빠질 위험이 있다.
    2. **순차적 모델(sequential models)** 의 경우, 시퀀스가 너무 길면 일반적으로 미니배치 단위로 잘라서 처리한다. 이때 샘플이 잘리면서 **편향(deviation)** 이 발생할 수 있는데, 이를 어떻게 분석하고 수정할지가 중요하다.
    3. **확률적 변분 추론(Stochastic Variational Inference, SVI)** 은 우아하고 실용적인 접근법이며, 여기에 고차 도함수(high-order gradient) 정보를 적용하는 방법을 개발하는 것이 유망하다.
    4. **확률적 기법(stochastic techniques)을 공액기울기법(Conjugate Gradient Method)** 에 접목하는 것도 우아하고 강력한 최적화 알고리즘을 얻는 방법이 될 수 있다.
    
    이 논문의 목적은 머신러닝 관점에서 **고전적(classical) 및 현대적(modern) 최적화 기법을 요약·분석**하는 데 있다. 우선 이론적 기반을 다루었고, 이어서 다양한 최적화 기법들의 응용을 설명하였으며, 마지막으로 향후 연구를 위한 **도전 과제와 오픈 문제**들을 제시하였다.
    

<aside>

# 6. 결론

- 최적화 문제의 도전과제
    - **데이터 부족 상황에서의 DNN 최적화와 성능 향상 : 샘플 수가 충분하지 않으면 고분산과 과적합이 발생하기 쉽다**. 또한 **non-convex 최적화 문제는 global optimum이아니라 local optimum에 빠질 위험**이 있다.
    - **Sequentail models :** 시퀀스가 너무 길면 **미니 배치 단위로 잘라서 처리하는데 이때 샘플이 잘리면서 deviation(편향)이 발생** 할 수 있다.
    - **Stochastic Variational Inferecne(SVI, 확률적 변분 추론)** : 실용적인 접근법이며, 고차 도함수를 적용하는 방법을 개발하는것이 유망하다.
    - **Stochastic Techniques(확률적 기법), Conjugate Gradient Method(공액기울기법)** : 또한 최적화 알고리즘을 얻는 유망한 방법이다.

<aside>

### 확률적 변분 추론 (Stochastic Variational Inference, SVI)

복잡한 확률 모델에서 **근사적인 추론**을 빠르고 효율적으로 수행하기 위한 알고리즘으로, 분석적으로 계산하기 어려운 사후 분포(posterior dist.)를 다룰 때 유용하다. 대규모 데이터셋에 효과적으로 적용할 수 있다.

- 핵심 아이디어 : 다루기 힘든 실제 **사후분포를 다루기 쉬운 간단한 근사분포로 대체**하는데 **일반적으로 KL-divergence를 최소화하는 방향**을 찾는것임
- **확률적 :** 전체 데이터 중 일부를 즉 미니배치만을 무작위로 샘플링하여 파라미터를 업데이트하는데 이는 SGD랑 비슷하다.
- **고차 도함수 적용의 유망성** : 일반적으로 **SVI는 경사 기반 최적화를 수행**하는데, **헤신거리와 같은 2차 도함수 정보를 활용하게 된다면 손실 함수의 곡률까지 고려**할 수 있어 훨씬 더 빠르고 안정적인 수렴을 기대할 수 있음. 이는 뉴턴 방법이 경사 하강법보다 더 빠르게 최적점에 도달하는것과 동일하다.
</aside>

<aside>

### 확률적 기법 (Stochastic Techniques)

알고리즘 일부 과정에 **무작위성**을 도입하는 모든 방법을 포괄한다. 이는 대규모 문제나 잡음이 많은 데이터를 다룰 때 효과적이다. 결정론적 방법은 일정한 경로로 해를 찾아가는것과 달리, 확률적 방법은 local optima에 빠지는것을 방지하고 global optima를 찾을 가능성을 높여준다.

**주요 예시**:

- **확률적 경사 하강법 (SGD)**: 전체 데이터가 아닌 미니배치를 무작위로 뽑아 경사를 계산하고 모델 파라미터를 업데이트하는 가장 대표적인 확률적 최적화 알고리즘입니다.
- **몬테카를로 방법 (Monte Carlo Methods)**: 무작위 샘플링을 통해 적분이나 확률 분포 계산 등 어려운 수치 계산을 근사적으로 수행하는 기법입니다.
- **시뮬레이티드 어닐링 (Simulated Annealing)**: 담금질 과정에서 영감을 얻은 최적화 방법으로, 확률적으로 현재 해보다 좋지 않은 해를 선택할 수도 있어 지역 최적점을 탈출하는 데 도움을 줍니다.
- **유전 알고리즘 (Genetic Algorithms)**: 생물의 진화 과정을 모방한 탐색 기법으로, 선택, 교배, 돌연변이 등 확률적 연산을 통해 해를 개선해 나갑니다.
</aside>

<aside>

### 공액 기울기법 (Conjugate Gradient Method)

**주로 선형문제나, 이차 형식의 함수를 최적화하는데 사용되는 효율적인 반복 알고리즘임**

- 핵심 아이디어 : 일반적인 경사 하강법은 현재 위치에서 가장 가파른 방향으로만 이동한다. 이 때문에 진행했떤 방향의 탐색 결과를 다시 무효로 하는 지그재그 움직임을 보일 수 있는데, 공액 기울기법은 공액이라는 특별한 성질을 만족하는 방향으로 탐지를 진행하며, **한번 이동하면 그 방향으로는 더 이상 최적화할 필요가 없다는것이 보장**된다.
- 원리
    1. 가장 가파른 경사하강으로 처음 이동
    2. 다음 이동부터는 이전의 기울기 정보를 활용하여, 이전 탐색 방향과 공액 관계에 있는 새로운 탐색 방향을 만든다
    3. 이 과정을 반복한다. 최악은 N번이다.
- 비선형 공액 기울기법도 경사 하강법 보다 훨씬 효율적인 대안으로 사용된다.
</aside>

### ✅ 5줄 요약

- 최적화는 머신러닝 성능 향상의 핵심 동력.
- **데이터 부족 + DNN의 비볼록(non-convex) 구조** → 여전히 큰 난제.
- 순차 모델은 **배치 단위 처리로 인한 편향 문제** 존재.
- **SVI + 고차 정보** 결합은 향후 중요한 연구 방향.
- **Stochastic Conjugate Gradient** 는 우아하면서도 강력한 새로운 최적화법으로 주목.

📌 주요 포인트

| 항목                 | 내용                                                                                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Convex vs Non-convex | • Convex 문제에서는 기존 기법(Gradient Descent, Newton 등)으로 전역 최적해 보장• DNN, RL 등은 대부분 Non-convex → 지역 최적해 위험, saddle point 문제 |
| 데이터셋 문제        | **데이터 부족** 시 DNN은 과적합·분산 증가 → Transfer Learning, Meta Learning 같은 접근 필요                                                           |
| 모델 구조            | Sequential models (RNN, LSTM 등)에서 batch truncation이 최적화 성능에 직접 영향                                                                       |
| 학습 방법            | Variational Inference는 확률적 최적화와 결합해야 확장성 확보Conjugate Gradient에 stochastic 기법 접목은 새로운 연구 방향                              |
| 결과 해석            | 최적화 기법이 ML 발전을 견인했지만, **대규모 비볼록 문제, 고차 정보 활용, 확률적 최적화 결합**이 향후 연구 과제                                       |
</aside>

---

### 📌 서론

### 번역

1. 원문 (Introduction 발췌)

RECENTLY, machine learning has grown at a remarkable rate, attracting a great number of researchers and practitioners. It has become one of the most popular research directions and plays a significant role in many fields, such as machine translation, speech recognition, image recognition, recommendation system, etc. Optimization is one of the core components of machine learning. The essence of most machine learning algorithms is to build an optimization model and learn the parameters in the objective function from the given data. In the era of immense data, the effectiveness and efficiency of the numerical optimization algorithms dramatically influence the popularization and application of the machine learning models. In order to promote the development of machine learning, a series of effective optimization methods were put forward, which have improved the performance and efficiency of machine learning methods.

From the perspective of the gradient information in optimization, popular optimization methods can be divided into three categories: first-order optimization methods, which are represented by the widely used stochastic gradient methods; high-order optimization methods, in which Newton’s method is a typical example; and heuristic derivative-free optimization methods, in which the coordinate descent method is a representative.

As the representative of first-order optimization methods, the stochastic gradient descent method, as well as its variants, has been widely used in recent years and is evolving at a high speed. However, many users pay little attention to the characteristics or application scope of these methods. They often adopt them as black box optimizers, which may limit the functionality of the optimization methods. In this paper, we comprehensively introduce the fundamental optimization methods. Particularly, we systematically explain their advantages and disadvantages, their application scope, and the characteristics of their parameters.

Compared with first-order optimization methods, high-order methods converge at a faster speed in which the curvature information makes the search direction more effective. High-order optimizations attract widespread attention but face more challenges. The difficulty in high-order methods lies in the operation and storage of the inverse matrix of the Hessian matrix. To solve this problem, many variants based on Newton’s method have been developed, most of which try to approximate the Hessian matrix through some techniques.

Derivative-free optimization methods are mainly used in the case that the derivative of the objective function may not exist or be difficult to calculate. There are two main ideas in derivative-free optimization methods. One is adopting a heuristic search based on empirical rules, and the other is fitting the objective function with samples. Derivative-free optimization methods can also work in conjunction with gradient-based methods.

---

1. 번역문

최근 머신러닝(Machine Learning)은 눈에 띄는 속도로 성장하여 많은 연구자와 실무자들을 끌어들이고 있다. 머신러닝은 오늘날 가장 인기 있는 연구 분야 중 하나가 되었으며, 기계번역(Machine Translation), 음성 인식(Speech Recognition), 이미지 인식(Image Recognition), 추천 시스템(Recommendation System) 등 다양한 분야에서 중요한 역할을 수행한다.

머신러닝의 핵심 구성 요소 중 하나가 **최적화(Optimization)** 이다. 대부분의 머신러닝 알고리즘의 본질은 최적화 모델을 세우고, 주어진 데이터로부터 목적함수(Objective Function)의 매개변수(parameter)를 학습하는 것이다. 거대 데이터 시대에는 수치적 최적화 알고리즘의 효과성과 효율성이 머신러닝 모델의 보급과 응용에 직접적으로 큰 영향을 미친다. 이러한 이유로 머신러닝 발전을 촉진하기 위해 다양한 효율적인 최적화 방법들이 제안되어 왔고, 이들은 성능과 효율을 크게 개선시켰다.

최적화에서 사용되는 **기울기(gradient) 정보**의 관점에서 널리 사용되는 방법들은 세 가지로 구분된다.

1. **1차 최적화 기법(First-order optimization methods)**: 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 계열이 대표적이다.
2. **고차 최적화 기법(High-order optimization methods)**: 뉴턴법(Newton’s method)이 전형적인 예이다. 곡률(curvature) 정보를 활용하여 탐색 방향을 더 효과적으로 만든다.
3. **무도함수 최적화 기법(Derivative-free optimization methods)**: 목적 함수의 도함수가 존재하지 않거나 계산하기 어려울 때 사용되며, 좌표 하강법(Coordinate Descent)이 대표적이다.

이 중 1차 최적화 기법을 대표하는 SGD와 그 변형들은 최근 수년간 폭넓게 사용되며 빠르게 발전하고 있다. 그러나 많은 사용자들은 이 기법들의 특성이나 적용 범위에 주의를 기울이지 않고, **블랙박스 최적화 도구(black box optimizers)** 로서 사용하기 때문에, 최적화 방법의 기능적 잠재력을 제한하는 경우가 있다. 따라서 본 논문에서는 기초적인 최적화 기법들을 포괄적으로 소개하고, 각각의 장단점, 적용 가능 범위, 매개변수 특성을 체계적으로 설명한다.

고차 최적화 기법은 1차 기법보다 더 빠른 수렴 속도를 가지지만, Hessian 행렬의 역행렬을 계산·저장해야 하는 어려움이 있다. 이를 해결하기 위해 뉴턴법을 변형한 다양한 방법들이 개발되었으며, 대부분 Hessian을 근사(approximate)하는 방식을 채택한다.

마지막으로, 무도함수 최적화 기법은 목적 함수의 도함수가 존재하지 않거나 계산이 어려울 때 주로 사용된다. 이 방법에는 경험적 규칙에 기반한 **휴리스틱 탐색(heuristic search)** 과 표본(sample)을 이용하여 목적 함수를 근사하는 방법이 있다. 또한 무도함수 최적화 기법은 경사 기반 방법과 함께 사용될 수도 있다.

<aside>

# 1. 서론

- 머신러닝의 핵심 구성 요소중 하나는 최적화이다. 이는 주어진 데이터로부터 목적함수의 매개변수를 학습하는것이다.
- **기울기**
    - **1차 최적화 기법(First-order optimization methods)**: 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 계열이 대표적이다.
    - **고차 최적화 기법(High-order optimization methods)**: 뉴턴법(Newton’s method)이 전형적인 예이다. 곡률(curvature) 정보를 활용하여 탐색 방향을 더 효과적으로 만든다.
    - **무도함수 최적화 기법(Derivative-free optimization methods)**: 목적 함수의 도함수가 존재하지 않거나 계산하기 어려울 때 사용되며, 좌표 하강법(Coordinate Descent)이 대표적이다.
- 이중 SGD와 같은 1차 최적화 기법들은 빠르게 발전되어 왔다. 그러나 사용자들은 이 **기법들의 특성이나 적용 범위에 주의를 기울이지 않고, 블랙박스 최적화 도구로서 사용하기 때문**에 기능적 잠재력을 제한하는 경우가 많다. 그래서 소개하고자 한다.
- **고차 최적화 기법은 1차 기법보다 더 빠른 수렴 속도를 가지지만 헤신 행렬의 역행렬을 저장하고 계싼해야하는 어려움**이 있고 이를 해결하기 위해 다양한 방법들이 개발되었고, 대부분 **해신을 근사하는 방식을 채택**한다.
- 무도함수 최적화 기법은 도함수가 존재하지 않거나 계산이 어려울때 사용되는데 **경험적 규칙에 기반한 휴리스틱 탐색과 표본을 이용하여 목적함수를 근사**한다. 또한 경사기반기법과 같이 사용하기도 한다.

### 요약

- 머신러닝은 본질적으로 **최적화 문제**로 볼 수 있다.
- 최적화는 **1차·고차·무도함수** 세 가지로 분류된다.
- Convex 문제에서는 전역 최적해 보장이 가능하다.
- Non-convex 문제(DNN, RL 등)는 지역 최적점·saddle point로 수렴하기 쉽다.
- SGD 남용은 위험하며, 상황별 최적화 기법 이해와 선택이 필요하다.

### 핵심

| 구분                            | 특징                                                        | 장점                              | 단점                                 | Convex / Non-convex 적용성                                            |
| ------------------------------- | ----------------------------------------------------------- | --------------------------------- | ------------------------------------ | --------------------------------------------------------------------- |
| 1차 기법 (First-order)          | Gradient 기반 (SGD, Mini-batch 등)                          | 계산 단순, 대규모 데이터에 효율적 | 수렴 속도 느림, 학습률 튜닝 필요     | Convex: 전역 최적해 보장 / Non-convex: 지역 최적해, saddle point 문제 |
| 고차 기법 (High-order)          | Hessian 등 곡률(curvature) 정보 활용 (Newton, Quasi-Newton) | 빠른 수렴, 방향성 우수            | Hessian 역행렬 계산·저장 부담        | Convex: 빠른 전역 수렴 / Non-convex: 계산 부담 커서 제한적            |
| 무도함수 기법 (Derivative-free) | Gradient 필요 없음 (Coordinate descent, Heuristic search)   | 미분 불가능 문제 해결, 단순 구현  | 이론적 보장 약함, 전역 최적화 불확실 | Convex/Non-convex 모두 가능하나 Heuristic 의존성 큼                   |
</aside>

---

---

## 🔬 실험과정

### 📚 Machine Learning Formulated as Optimization

### 번역

1. 원문
    
    Almost all machine learning algorithms can be formulated as an optimization problem to find the extremum of an objective function. Building models and constructing reasonable objective functions are the first step in machine learning methods. With the determined objective function, appropriate numerical or analytical optimization methods are usually used to solve the optimization problem.
    
    According to the modeling purpose and the problem to be solved, machine learning algorithms can be divided into supervised learning, semi-supervised learning, unsupervised learning, and reinforcement learning. Particularly, supervised learning is further divided into the classification problem (e.g., sentence classification, image classification, etc.) and regression problem; unsupervised learning is divided into clustering and dimension reduction, among others.
    

---

1. 번역문
    
    거의 모든 머신러닝 알고리즘은 **목적함수(objective function)의 극값(extremum)** 을 찾는 최적화 문제로 공식화(formulated)할 수 있다. 모델을 세우고 합리적인 목적함수를 구성하는 것이 머신러닝 방법의 첫 번째 단계이다. 목적함수가 결정되면, 일반적으로 적절한 수치적(numerical) 혹은 해석적(analytical) 최적화 방법을 사용하여 이 문제를 해결한다.
    
    모델링 목적과 해결하려는 문제에 따라 머신러닝 알고리즘은 크게 **지도학습(supervised learning), 준지도학습(semi-supervised learning), 비지도학습(unsupervised learning), 강화학습(reinforcement learning)** 으로 나눌 수 있다. 특히 지도학습은 다시 **분류(classification)** 와 **회귀(regression)** 문제로 나눌 수 있으며, 비지도학습은 **군집(clustering)** 과 **차원 축소(dimension reduction)** 등으로 세분된다.
    
2. 원문
    
    For supervised learning, the goal is to find an optimal mapping function f(x) to minimize the loss function of the training samples,
    
    minθ (1/N) Σ_{i=1}^N L(yi, f(xi, θ)),   (1)
    
    where N is the number of training samples, θ is the parameter of the mapping function, xi is the feature vector of the ith sample, yi is the corresponding label, and L is the loss function.
    
    There are many kinds of loss functions in supervised learning, such as the square of Euclidean distance, cross-entropy, contrast loss, hinge loss, information gain and so on. For regression problems, the simplest way is using the square of Euclidean distance as the loss function, that is, minimizing square errors on training samples. But the generalization performance of this kind of empirical loss is not necessarily good. Another typical form is structured risk minimization, whose representative method is the support vector machine. On the objective function, regularization items are usually added to alleviate overfitting, e.g., in terms of L2 norm,
    
    minθ (1/N) Σ_{i=1}^N L(yi, f(xi, θ)) + λ ||θ||₂² ,   (2)
    
    where λ is the compromise parameter, which can be determined through cross-validation.
    

---

1. 번역문
    
    **지도학습(Supervised Learning)** 에서 목표는 훈련 샘플의 손실함수(loss function)를 최소화하는 **최적 매핑 함수 f(x)** 를 찾는 것이다.
    
    목적함수는 다음과 같이 정의된다:
    
    minθ (1/N) Σ_{i=1}^N L(yi, f(xi, θ)),   (1)
    
    여기서 N은 훈련 샘플 수, θ는 매핑 함수의 파라미터, xi는 i번째 샘플의 특징 벡터(feature vector), yi는 해당 샘플의 레이블(label), L은 손실함수(loss function)를 의미한다.
    
    지도학습에서 사용되는 손실함수는 다양하다. 예를 들어 **유클리드 거리 제곱(Square of Euclidean distance)**, **교차 엔트로피(Cross-Entropy)**, **대조 손실(Contrast Loss)**, **힌지 손실(Hinge Loss)**, **정보 이득(Information Gain)** 등이 있다.
    
    - 회귀(Regression) 문제의 경우, 가장 단순한 방식은 유클리드 거리 제곱을 손실함수로 사용하는 것이며, 이는 훈련 샘플에 대한 제곱 오차를 최소화하는 것이다. 그러나 이러한 경험적 손실은 일반화 성능이 반드시 좋다고 보장되지 않는다.
    - 또 다른 전형적인 형태는 **구조적 위험 최소화(Structured Risk Minimization, SRM)** 이며, 대표적인 방법은 **서포트 벡터 머신(Support Vector Machine, SVM)** 이다.
    
    또한 과적합(overfitting)을 완화하기 위해 목적함수에 보통 **정규화 항(regularization term)** 을 추가한다. 대표적으로 L2 노름 기반 정규화가 있으며, 다음과 같이 표현된다:
    
    minθ (1/N) Σ_{i=1}^N L(yi, f(xi, θ)) + λ ||θ||₂² ,   (2)
    
    여기서 λ는 균형 파라미터(compromise parameter)로, 일반적으로 교차검증(cross-validation)을 통해 결정된다.
    
    1. 원문
        
        Semi-supervised learning lies between supervised learning and unsupervised learning. In real applications, unlabeled samples are often more than labeled samples. Semi-supervised learning uses a small amount of labeled samples and a large number of unlabeled samples to construct an optimization problem. One type of semi-supervised learning is called semi-supervised support vector machine (S3VM). In S3VM, the constraint of maximizing margin is imposed not only on labeled samples but also on unlabeled samples.
        
        Another type of semi-supervised learning is graph-based semi-supervised learning, which regards all samples as vertices of a graph, and then edges are used to represent the similarity between samples. The assumption is that two similar samples are more likely to have the same label. The optimization objective function is to minimize the difference between labels of adjacent vertices.
        
    
    ---
    
    1. 번역문
        
        **준지도학습(Semi-supervised Learning)** 은 지도학습과 비지도학습의 중간에 위치한다. 실제 응용에서는 **라벨이 없는 샘플(unlabeled samples)** 이 라벨이 있는 샘플보다 더 많은 경우가 흔하다. 준지도학습은 소량의 라벨이 있는 샘플과 다수의 라벨 없는 샘플을 함께 활용하여 최적화 문제를 구성한다.
        
        준지도학습의 한 가지 유형은 **준지도 서포트 벡터 머신(Semi-supervised Support Vector Machine, S3VM)** 이다. S3VM에서는 **마진 최대화(maximizing margin) 제약 조건**을 라벨이 있는 샘플뿐만 아니라 라벨 없는 샘플에도 적용한다.
        
        다른 유형으로는 **그래프 기반 준지도학습(Graph-based Semi-supervised Learning)** 이 있다. 이 방법에서는 모든 샘플을 그래프의 정점(vertex)으로 보고, 간선(edge)은 샘플 간의 유사성을 나타낸다. 기본 가정은 **유사한 두 샘플은 같은 라벨을 가질 가능성이 높다**는 것이다. 최적화 목적함수는 인접 정점의 라벨 간 차이를 최소화하는 것이다.
        
    
    ---
    
    1. 문 (Section II-C 전체)
        
        Clustering algorithms [67], [82], [83], [84] divide a group of samples into multiple clusters ensuring that the differences between the samples in the same cluster are as small as possible, and samples in different clusters are as different as possible. The optimization problem for the k-means clustering algorithm is formulated as minimizing the following loss function:
        
        K
        
        min ∑ ∑ ||x − μk||₂² ,   (2)
        
        Sk=1  x∈Sk
        
        where K is the number of clusters, x is the feature vector of samples, μk is the center of cluster k, and Sk is the sample set of cluster k. The implication of this objective function is to make the sum of variances of all clusters as small as possible.
        
        The dimensionality reduction algorithm ensures that the original information from data is retained as much as possible after projecting them into the low-dimensional space. Principal component analysis (PCA) [85], [86], [87] is a typical algorithm of dimensionality reduction methods. The objective of PCA is formulated to minimize the reconstruction error as
        
        min Σ ||xi − x̂i||₂² ,   (5)
        
        N
        
        i=1
        
        where N represents the number of samples, xi is a D-dimensional vector, x̂i is the reconstruction of xi. zi = {z1i, …, zD′i} is the projection of xi in D′-dimensional coordinates, D ≫ D′. ej is the standard orthogonal basis under D′-dimensional coordinates.
        
        Another common optimization goal in probabilistic models is to find an optimal probability density function of p(x), which maximizes the logarithmic likelihood function (MLE) of the training samples,
        
        max Σ ln p(xi ; θ).   (6)
        
        N
        
        i=1
        
        In the framework of Bayesian methods, some prior distributions are often assumed on parameter θ, which also has the effect of alleviating overfitting.
        
    
    ---
    
    1. 번역문
        
        **클러스터링 알고리즘(Clustering algorithms)** [67], [82], [83], [84] 은 샘플 집합을 여러 개의 클러스터로 나누어, 동일 클러스터 내의 샘플 간 차이가 가능한 한 작고, 서로 다른 클러스터 간 차이가 가능한 한 크도록 한다. k-means 군집화의 최적화 문제는 다음의 손실함수를 최소화하는 것으로 공식화된다:
        
        K
        
        min ∑ ∑ ||x − μk||₂² ,   (2)
        
        k=1  x∈Sk
        
        여기서 K는 클러스터 개수, x는 샘플의 특징 벡터, μk는 k번째 클러스터의 중심, Sk는 클러스터 k에 속하는 샘플 집합이다. 이 목적함수의 의미는 **모든 클러스터의 분산 합을 최소화**하는 것이다.
        
        **차원 축소(dimensionality reduction)** 알고리즘은 데이터를 저차원 공간으로 사영(projection)했을 때 원래 정보를 최대한 유지하는 것을 보장한다. **주성분 분석(Principal Component Analysis, PCA)** [85], [86], [87] 은 전형적인 차원 축소 방법으로, 목적은 샘플의 재구성 오차(reconstruction error)를 최소화하는 것이다.
        
        min Σ ||xi − x̂i||₂² ,   (5)
        
        N
        
        i=1
        
        여기서 N은 샘플 수, xi는 D차원 벡터, x̂i는 xi의 재구성, zi = {z1i, …, zD′i} 는 xi의 D′ 차원 공간으로의 투영 값이며, D ≫ D′ 이다. ej는 D′ 차원 좌표계에서의 표준 직교 기저(orthogonal basis)이다.
        
        **확률적 모델(probabilistic models)** 에서 또 다른 일반적 최적화 목표는 p(x)의 최적 확률 밀도 함수를 찾는 것으로, 이는 학습 샘플의 로그우도(log-likelihood)를 최대화하는 문제이다:
        
        max Σ ln p(xi ; θ).   (6)
        
        N
        
        i=1
        
        베이지안 방법론의 틀에서는 매개변수 θ에 대해 사전분포(prior distribution)를 가정하며, 이는 과적합(overfitting)을 완화하는 효과도 가진다.
        
    
    **D. Optimization Problems in Reinforcement Learning**
    
    Reinforcement learning [42], [88], [89], unlike supervised learning and unsupervised learning, aims to find an optimal strategy function, whose output varies with the environment. For a deterministic strategy, the mapping function from state s to action a is the learning target. For an uncertain strategy, the probability of executing each action is the learning target. In each state, the action is determined by a = π(s), where π(s) is the policy function.
    
    The optimization problem in reinforcement learning can be formulated as maximizing the cumulative return after executing a series of actions which are determined by the policy function,
    
    Vπ(s) = E [ Σ_{k=0}^∞ γ^k r_{t+k} | S_t = s ],   (7)
    
    where Vπ(s) is the value function of state s under policy π, r is the reward, and γ ∈ [0, 1] is the discount factor.
    
    maxπ Vπ(s)
    
    ---
    
    **E. Optimization for Machine Learning**
    
    Overall, the main steps of machine learning are to build a model hypothesis, define the objective function, and solve the maximum or minimum of the objective function to determine the parameters of the model. In these three vital steps, the first two steps are the modeling problems of machine learning, and the third step is to solve the desired model by optimization methods.
    
    ---
    
    1. 번역문
    
    **D. 강화학습(Reinforcement Learning)에서의 최적화 문제**
    
    강화학습 [42], [88], [89] 은 지도학습(supervised learning), 비지도학습(unsupervised learning)과 달리 **환경(environment)** 에 따라 출력이 달라지는 **최적 전략 함수(optimal strategy function)** 를 찾는 것을 목표로 한다.
    
    - 결정적 전략(deterministic strategy)의 경우, 상태(state) s에서 행동(action) a로의 매핑 함수가 학습 대상이다.
    - 불확실한 전략(stochastic strategy)의 경우, 각 행동이 실행될 확률이 학습 대상이다.
    - 각 상태에서 행동은 a = π(s)로 결정되며, π(s)는 정책 함수(policy function)이다.
    
    강화학습의 최적화 문제는 **정책 함수에 의해 결정된 일련의 행동을 수행한 뒤 얻게 되는 누적 보상(cumulative return)을 최대화**하는 것으로 공식화된다.
    
    Vπ(s) = E [ Σ_{k=0}^∞ γ^k r_{t+k} | S_t = s ],   (7)
    
    여기서 Vπ(s)는 정책 π 하에서 상태 s의 가치 함수(value function), r은 보상(reward), γ ∈ [0, 1] 은 할인 인자(discount factor)이다.
    
    최종 목표는 다음과 같다:
    
    maxπ Vπ(s)
    
    **E. 머신러닝에서의 최적화(Optimization for Machine Learning)**
    
    전반적으로 머신러닝의 주요 단계는 세 가지이다.
    
    1. **모델 가설(model hypothesis) 수립**
    2. **목적함수(objective function) 정의**
    3. **목적함수의 최댓값 또는 최솟값을 찾아 모델 파라미터를 결정**
    
    이 세 단계 중 처음 두 단계는 머신러닝의 모델링 문제이며, 세 번째 단계는 **최적화 방법을 통해 원하는 모델을 해결하는 과정**이다.
    

<aside>

# 2. Machine Learning Formulated as Optimization

- 머신러닝 문제 : **목적함수의 극값을 찾는 최적화 문제**
    - 모델을 세우고 적적한 목적함수를 구성하는 것이 중요
    - 일반적으로 목적함수가 결정되면, 수치적 혹은 해석적 방법으로 이를 해결한다.
- **머신러닝 패러다임별 최적화**
    - 지도학습: 손실함수(loss function)를 최소화 (예: MSE, cross-entropy).
    - 준지도학습: 레이블 없는 데이터에 제약조건(constraints) 추가 → 혼합 최적화 문제.
    - 비지도학습: 군집화(K-means → 거리 최소화), 차원축소(PCA → 재구성 오차 최소화) 등 최적화 문제로 변환.
    - 강화학습: 누적 보상(cumulative reward)을 최대화하는 정책(policy) 학습 → 확률적 최적화.

<aside>

### Convex vs Non-convex

- **Convex vs Non-convex**
    - Convex: 선형 회귀, 로지스틱 회귀, SVM과 같은 문제는 Convex 최적화로 수식화 가능. → 전역 최적해(global optimum) 존재.
    - Non-convex: 심층 신경망(DNN), 강화학습(RL) 문제는 대부분 비볼록(non-convex). → 지역 최적해(local optimum) 또는 saddle point에 빠질 위험 존재.
</aside>

### 요약

- 머신러닝은 본질적으로 **목적함수 최적화 문제**로 수식화된다.
- 지도·준지도·비지도·강화학습 각각 최적화 문제의 형태로 정의할 수 있다.
- Convex 문제(로지스틱 회귀, PCA 등)는 전역 최적해 보장.
- Non-convex 문제(DNN, RL 등)는 지역 최적점·saddle point 문제 존재.
- 따라서 ML의 발전은 곧 다양한 **최적화 기법 발전**과 직결된다.

### 정리

| 학습 유형                         | 최적화 수식화                    | Convex / Non-convex 성격                       | 대표 예시                                 |
| --------------------------------- | -------------------------------- | ---------------------------------------------- | ----------------------------------------- |
| 지도학습 (Supervised)             | min (1/N) Σ L(yi, f(xi; θ))      | Convex (로지스틱 회귀, SVM) / Non-convex (DNN) | 분류(Classification), 회귀(Regression)    |
| 준지도학습 (Semi-supervised)      | 라벨 없는 데이터에 제약조건 추가 | Non-convex 혼합 문제 多                        | S3VM, Graph-based SSL                     |
| 비지도학습 (Unsupervised)         | K-means: min Σ                   |                                                | x - μ                                     |
| 강화학습 (Reinforcement Learning) | max E[Σ γ^k r_t+k]               | Non-convex (정책공간 탐색)                     | Q-learning, Policy Gradient, Actor-Critic |

## 2.1. Optimization Problems in Supervised Learning

![image.png](A%20Survey%20of%20Optimization%20Methods%20from%20a%20Machine%20Le%2025ca9b246de180a3838bcfa8d6ca3088/image.png)

- 지도학습에서의 훈련 목표 : 훈련 샘플의 손실함수를 최소화하는 최적 매핑 함수를 찾는것
- **유클리드 거리 제곱(Square of Euclidean distance)**, **교차 엔트로피(Cross-Entropy)**, **대조 손실(Contrast Loss)**, **힌지 손실(Hinge Loss)**, **정보 이득(Information Gain)**
    - 회귀(Regression) 문제의 경우, 가장 단순한 방식은 유클리드 거리 제곱을 손실함수로 사용하는 것이며, 이는 훈련 샘플에 대한 제곱 오차를 최소화하는 것이다. 경험적 손실은 일반화 성능이 반드시 좋다고 보장되지 않는다.
    - 또 다른 전형적인 형태는 **구조적 위험 최소화(Structured Risk Minimization, SRM)** 이며, 대표적인 방법은 **서포트 벡터 머신(Support Vector Machine, SVM)** 이다.
- 과적화 방지를 위해 **정규화항**을 추가하기도 한다. 아래는 L2Norm이다.

![image.png](A%20Survey%20of%20Optimization%20Methods%20from%20a%20Machine%20Le%2025ca9b246de180a3838bcfa8d6ca3088/image%201.png)

### 요약

- 지도학습은 손실함수를 최소화하는 **f(x; θ)** 학습 문제로 정의된다.
- 손실함수 예시: MSE, Cross-entropy, Hinge loss 등.
- Convex 손실함수는 전역 최적해 보장, Non-convex(DNN)는 지역 최적해 문제 존재.
- 정규화 항(L2 norm 등)은 과적합 방지를 위해 자주 추가된다.
- λ 값은 교차검증으로 결정하며, 모델의 복잡도와 일반화 성능을 조절한다.

### 정리

| 요소                         | 설명                                    | Convex 여부               |
| ---------------------------- | --------------------------------------- | ------------------------- |
| 기본 목적함수                | minθ (1/N) Σ L(yi, f(xi; θ))            | 손실함수 종류에 따라 다름 |
| 손실함수 예시                | L2 loss, Cross-entropy, Hinge loss      | 대부분 Convex             |
| Structured Risk Minimization | 일반화 오류 최소화, SVM 대표            | Convex                    |
| 정규화 항                    | λ                                       |                           |
| 심층 신경망                  | 목적함수 비선형·복잡 → 지역 최적점 문제 | Non-convex                |

## 2.2. Optimization Problems in Semi-supervised Learning

![image.png](A%20Survey%20of%20Optimization%20Methods%20from%20a%20Machine%20Le%2025ca9b246de180a3838bcfa8d6ca3088/image%202.png)

다른 유형으로는 **그래프 기반 준지도학습(Graph-based Semi-supervised Learning)** 이 있다. 이 방법에서는 모든 샘플을 그래프의 정점(vertex)으로 보고, 간선(edge)은 샘플 간의 유사성을 나타낸다. 기본 가정은 **유사한 두 샘플은 같은 라벨을 가질 가능성이 높다**는 것이다. 최적화 목적함수는 인접 정점의 라벨 간 차이를 최소화하는 것이다.

- 준지도학습에서의 훈련 목표 : 라벨이 있는 소량의 라벨과 라벨이 없는 다수의 라벨을 활용하여 최적화 문제를 구성
- **준지도 서포트 벡터 머신(Semi-supervised Support Vector Machine, S3VM)**
    - **마진 최대화(maximizing margin) 제약 조건 등으로 라벨과 비라벨에 둘 다 적용한다.**
- **그래프 기반 준지도학습(Graph-based Semi-supervised Learning) : 모든 샘플을 정점으로보고, 유사한 두 샘플은 같은 라벨을 가질 가능성이 높다**는 것 - 라벨 간 차이를 최소화하는 것이다.

<aside>

### SRM

- 경험적 위험 (Empirical Risk) : 모델이 훈련 데이터에서 얼마나 많이 틀리는가
- 구조적 위험 (Structural Risk) : 모델의 복잡도로 인해 발생하는 위험, 과적합이 발생할 위험이 커짐

SRM은 두 가지 위험의 합을 최소화하는 것을 목표로 한다.

- 최종위험 (일반화 오류) ≤ 경험적 위험 + 구조적 위험(모델 복잡도에 따른 패널티)
    - 정규화 : 패널티 항을 추가하는 기법
    - SVM : SRM 원리를 가장 잘 구현한 알고리즘으로 마진을 최대화 하려고함.
</aside>

### 요약

- 준지도학습은 **라벨 부족 문제** 해결을 위해 라벨 없는 데이터까지 최적화에 포함한다.
- S3VM은 라벨 없는 샘플에도 마진 최대화 적용 → **Non-convex 최적화**.
- Graph-based SSL은 그래프 구조를 기반으로 인접 샘플 라벨 차이를 최소화 → **Convex 가능**.
- 준지도학습의 목적함수는 “지도 손실 + 비지도 제약조건”의 혼합.
- 실제 응용에서 일반화 성능 향상을 위해 필수적인 방법론이다.

### 중요

| 방법            | 최적화 공식화                                         | Convex 여부 | 특징                                           |
| --------------- | ----------------------------------------------------- | ----------- | ---------------------------------------------- |
| S3VM            | 라벨 있는 샘플 + 없는 샘플 모두 마진 최대화           | Non-convex  | 라벨 부족 상황에서 강력, 하지만 계산 복잡      |
| Graph-based SSL | 그래프 라플라시안 정규화 → 인접 노드 라벨 차이 최소화 | Convex 가능 | 유사도 기반, Semi-supervised Clustering에 적합 |
| 공통점          | 지도 손실 + 비지도 제약 결합                          | 혼합        | 라벨 없는 데이터 활용으로 일반화 성능 향상     |

## 2.3. Optimization Problems in Unsupervised Learning

- 비지도 학습 목표 : 데이터로부터 **숨겨진 구조**를 발견하는 것을 목표로 함
    - **군집화 :** 대표적인 예, k-means
        
        ![image.png](A%20Survey%20of%20Optimization%20Methods%20from%20a%20Machine%20Le%2025ca9b246de180a3838bcfa8d6ca3088/image%203.png)
        
    - **PCA :** 차원축소 - 재구성 오차를 최소화
        
        ![image.png](A%20Survey%20of%20Optimization%20Methods%20from%20a%20Machine%20Le%2025ca9b246de180a3838bcfa8d6ca3088/image%204.png)
        
    - Bayesian BLE : 확률 밀도 함수를 찾는것으로 우도함수를 최대화하는 문제
1. **Clustering (k-means)**
    - 목적: 각 샘플이 속한 클러스터 중심과의 제곱거리 합 최소화.
    - Convex 여부: **Non-convex** (클러스터 할당이 이산적).
    - 문제: 초기값에 따라 지역 최적해(local optimum)에 머무름.
2. **차원 축소 (PCA)**
    - 목적: 원 데이터와 재구성 데이터 간 오차 최소화.
    - Convex 여부: **Convex** (고유벡터 분해 기반, 전역 최적해 보장).
    - 데이터 분산을 최대 보존하는 저차원 투영을 찾는 수학적으로 안정된 방법.
3. **확률적 모델 (MLE, Bayesian)**
    - 목적: 관측 데이터에 대한 Likelihood 최대화.
    - Convex 여부: **Non-convex** (특히 GMM, VAE, GAN 등).
    - Bayesian 접근: 파라미터에 Prior 분포를 부여 → 과적합 완화.
    - 실제로는 EM 알고리즘, 변분 추론, MCMC 등이 동원됨.

### 요약

- 비지도학습은 데이터의 숨겨진 구조를 찾는 최적화 문제로 정의된다.
- k-means는 클러스터 분산 최소화 → **Non-convex**, 초기화 민감.
- PCA는 재구성 오차 최소화 → **Convex**, 전역 최적해 가능.
- 확률적 모델은 로그우도 최대화 → **Non-convex**, 근사추론 필요.
- Bayesian 접근은 Prior를 추가해 과적합 완화 가능.

### 정리

| 방법              | 최적화 문제              | Convex 여부 | 특징                           |
| ----------------- | ------------------------ | ----------- | ------------------------------ |
| k-means           | min Σ                    |             | x − μk                         |
| PCA               | min Σ                    |             | xi − x̂i                        |
| 확률적 모델 (MLE) | max Σ ln p(xi ; θ)       | Non-convex  | 근사추론 필요, EM/VI/MCMC 활용 |
| Bayesian 확장     | max Posterior with prior | Non-convex  | Prior로 과적합 완화 가능       |

# 2.4. 강화학습

- **환경에 따라 출력이 달라지는 최적 전략함수를 찾는것  → 누적 보상 최대화**
    - 결정 정책 : 상태에서 행동으로 함수를 매핑, 학습 대상
    - 확률적 정책 : 행동이 실행될 확률이 대상

![image.png](A%20Survey%20of%20Optimization%20Methods%20from%20a%20Machine%20Le%2025ca9b246de180a3838bcfa8d6ca3088/image%205.png)

### 요약

- 강화학습은 환경과 상호작용하며 **정책(policy) 최적화**를 수행한다.
- 목적은 상태 가치 함수 Vπ(s)의 기대 누적 보상을 최대화하는 것.
- 단순한 경우 Convex 가능하나, Deep RL은 **Non-convex** 최적화 문제.
- 머신러닝의 세 단계는 모델 가설 설정 → 목적함수 정의 → 최적화 수행.
- 이 중 최적화가 실제 학습을 담당하며, ML 발전의 핵심 동력이다.

### 정리

 

| 구분          | 최적화 문제                        | Convex 여부                   | 특징                               |
| ------------- | ---------------------------------- | ----------------------------- | ---------------------------------- |
| 강화학습 (RL) | maxπ Vπ(s) = E[Σ γ^k r]            | 대부분 Non-convex             | 정책 함수 최적화, 누적 보상 최대화 |
| 지도/비지도   | min/max 목적함수 (손실, 우도 등)   | Convex & Non-convex 혼재      | 데이터 기반 학습                   |
| 머신러닝 전반 | 모델 가설 → 목적함수 정의 → 최적화 | 단계 1·2는 모델링, 3은 최적화 | 최적화가 실질적 학습의 핵심        |
</aside>

### 📚 (예시)

### 번역

<aside>

</aside>

### 📚 (예시)

### 번역

<aside>

</aside>

### 📚 (예시)

### 번역

<aside>

</aside>

### 📚 (예시)

### 번역

<aside>

</aside>

### 📚 (예시)

### 번역

<aside>

</aside>

### 📚 (예시)

### 번역

<aside>

</aside>

---