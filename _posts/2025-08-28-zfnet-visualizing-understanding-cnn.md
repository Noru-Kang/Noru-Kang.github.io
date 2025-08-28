---
title: "(ZFNet) Visualizing and Understanding Convolutional Networks"
date: 2025-08-28 13:00:00 +0900
categories: [AI-ML-DL, etc.]
tags: [paper, Computer-Vision, Diagnostic]
---

# (ZFNet) Visualizing and Understanding Convolutional Networks

---

### 🔗 출처

> https://arxiv.org/abs/1311.2901
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

## Visualizing and Understanding Convolutional Networks

</aside>

---

### 🌟 초록

### 번역

원문 (Abstract)

Large Convolutional Network models have recently demonstrated impressive classification performance on the ImageNet benchmark (Krizhevsky et al., 2012). However there is no clear understanding of why they perform so well, or how they might be improved. In this paper we address both issues. We introduce a novel visualization technique that gives insight into the function of intermediate feature layers and the operation of the classifier. Used in a diagnostic role, these visualizations allow us to find model architectures that outperform Krizhevsky et al. on the ImageNet classification benchmark. We also perform an ablation study to discover the performance contribution from different model layers. We show our ImageNet model generalizes well to other datasets: when the softmax classifier is retrained, it convincingly beats the current state-of-the-art results on Caltech-101 and Caltech-256 datasets.

---

번역문

대규모 합성곱 신경망(Convolutional Network, ConvNet) 모델은 최근 ImageNet 벤치마크(Krizhevsky et al., 2012)에서 인상적인 분류 성능을 보였다. 그러나 왜 그렇게 잘 작동하는지, 혹은 어떻게 개선될 수 있는지에 대해서는 명확한 이해가 부족하다. 본 논문에서는 이 두 가지 문제를 다룬다.

우리는 중간 특징 계층(intermediate feature layers)의 기능과 분류기의 동작을 이해할 수 있도록 새로운 시각화 기법을 제안한다. 진단적 도구로 사용될 때, 이 시각화는 Krizhevsky et al.의 모델을 능가하는 아키텍처를 찾아내는 데 도움을 준다. 또한, 다양한 계층이 성능에 기여하는 정도를 분석하기 위해 ablation study를 수행한다.

마지막으로, 우리의 ImageNet 모델은 다른 데이터셋에도 잘 일반화됨을 보였다. softmax 분류기만 새로 학습시키면, Caltech-101과 Caltech-256 데이터셋에서 기존 최고 성능(state-of-the-art)을 능가하는 결과를 달성한다.

<aside>

# 0. 초록

- 왜 ConvNet이 작동하는지 명확한 이해가 부족하다, 그리고 추가적으로 어떻게 개선될 수 있는지에 대한 이해가 부족하다.
- **진단적 도구** 제안 : 중간 특징 추출(intermediate feature layers)의 기능과 분류기의 동작을 이해 → 기존 모델을 능가하는 아키텍처를 차자내는 데 도움을 준다. 즉 **눈으로 이해**할 수 있다(블랙박스 부분을 화이트박스로)
- **abalation study** 수행
- **ZFNet개발 - 전이능력** 확인

### 정리

| 항목      | 내용                                                        |
| --------- | ----------------------------------------------------------- |
| 데이터셋  | ImageNet 2012, Caltech-101, Caltech-256                     |
| 모델 구조 | AlexNet 기반, stride/filter 개선, softmax 출력              |
| 연구 기여 | Deconvnet 시각화, 구조 최적화, ablation, 전이학습           |
| 평가 결과 | ImageNet에서 AlexNet보다 낮은 오류율, Caltech에서 SOTA 달성 |
</aside>

---

### 💡 결론 & 고찰

### 번역

원문 (Discussion)

We explored large convolutional neural network models, trained for image classification, in a number ways. First, we presented a novel way to visualize the activity within the model. This reveals the features to be far from random, uninterpretable patterns. Rather, they show many intuitively desirable properties such as compositionality, increasing invariance and class discrimination as we ascend the layers. We also showed how these visualization can be used to debug problems with the model to obtain better results, for example improving on Krizhevsky et al.’s (Krizhevsky et al., 2012) impressive ImageNet 2012 result.

We then demonstrated through a series of occlusion experiments that the model, while trained for classification, is highly sensitive to local structure in the image and is not just using broad scene context. An ablation study on the model revealed that having a minimum depth to the network, rather than any individual section, is vital to the model’s performance.

Finally, we showed how the ImageNet trained model can generalize well to other datasets. For Caltech-101 and Caltech-256, the datasets are similar enough that we can beat the best reported results, in the latter case by a significant margin. This result brings into question to utility of benchmarks with small (i.e. < 10^4) training sets. Our convnet model generalized less well to the PASCAL data, perhaps suffering from dataset bias (Torralba & Efros, 2011), although it was still within 3.2% of the best reported result, despite no tuning for the task. For example, our performance might improve if a different loss function was used that permitted multiple objects per image. This would naturally enable the networks to tackle the object detection as well.

번역문 (결론 및 고찰)

우리는 이미지 분류를 위해 학습된 대규모 합성곱 신경망(ConvNet) 모델을 다양한 방식으로 탐구하였다. 먼저, 모델 내부의 활동을 시각화하는 새로운 방법을 제시하였다. 이를 통해 특징(feature)들이 무작위적이거나 해석 불가능한 패턴이 아님이 드러났다. 오히려 계층이 깊어질수록 조합성(compositionality), 불변성(invariance)의 증가, 클래스 구별(class discrimination) 등 직관적으로 바람직한 특성을 보였다. 또한 이러한 시각화 기법이 모델 디버깅에 사용될 수 있음을 보여주었으며, 예를 들어 Krizhevsky et al.(2012)의 ImageNet 2012 성능을 개선할 수 있었다.

그 다음, 일련의 occlusion(가림) 실험을 통해, 분류용으로 학습된 모델이 단순히 장면의 광범위한 맥락(scene context)을 사용하는 것이 아니라 이미지의 **국소적 구조(local structure)**에 매우 민감하게 반응한다는 점을 입증하였다. 모델에 대한 ablation study는 개별 계층이 아니라 네트워크가 가지는 최소한의 깊이(minimum depth)가 성능에 필수적임을 드러냈다.

마지막으로, ImageNet에서 학습된 모델이 다른 데이터셋에서도 잘 일반화됨을 보였다. Caltech-101과 Caltech-256의 경우 데이터셋 특성이 충분히 유사하여 기존 최고 성능(state-of-the-art)을 능가했으며, 특히 Caltech-256에서는 상당한 차이로 성능을 앞섰다. 이 결과는 작은 규모(<10^4)의 학습셋을 갖는 벤치마크의 효용성에 의문을 제기한다. 반면 PASCAL 데이터에서는 상대적으로 일반화가 약했는데, 이는 데이터셋 바이어스(dataset bias) 때문일 수 있다(Torralba & Efros, 2011). 그럼에도 불구하고 별도의 조정 없이도 최고 결과와 3.2% 차이에 불과하였다. 만약 이미지 내 다중 객체를 허용하는 다른 손실 함수를 사용한다면 성능은 더 향상될 수 있으며, 이는 곧 객체 탐지(object detection) 문제까지도 자연스럽게 확장될 수 있을 것이다.

<aside>

# 6. Disccusion

- 시각화하는 새로운 방법을 제시함으로서, feature들이 무작위적이거나 해석 불가능한 패턴이 아님이 들어났음
    - 계층이 깊어질수록 compositionality(조합성), invariance(불변성), class discrimination(클래스 구별) 등 직관적인 특성들이 보였다.
    - 시각화 기법이 모델 디버깅에 사용될 수 있다. AlexNet 성능을 향상
- Occlusion(가림) 실험을 통해, clf.가 scene context(장면의 광범위한 맥락)을 사용하는것이 아니라, 이미지의 local structure(국소적 구조)에 매우 민감하게 반응한다는 것을 입증함.
- Ablation study를 통해 개별 계층이 아니라 네트워크가 가지는 minimum depth(최소한의 깊이)가 성능에 필수적이다.
- 전이 학습의 효용성을 증명했다(일반화 성능), 그러나 PASCAL과 같은 데이터에서 dataset bias때문에, 일반화가 약했으나 그 마저도 낮은 감소였다. → 객체탐지까지 자연스럽게 확장될 수 있다.

### 핵심

- convnet은 해석가능하고, 계층이 깊어질수록 추상화, 불변성이 증가
- 시각화 : 단순 시각 설명도구가 아니라, 모델 디버깅(개선용)에 사용가능
- Occlusion : 모델은 맥락보다, 진짜 객체 구조(local structure)에 집중
- Ablation : 모델은 층의 뉴런수보다, 충분한 깊이가 성능에 핵슴
- 전이학습 특정 데이터셋에서 효과적, 그러나 데이터의 bias를 처리할 수 있다면 다른 데이터셋에서도 효과적일거라 예상
    - ImageNet → Caltech(작은 데이터셋) : 효과적
    - 역은 효과적인지 의문
    - 대규모데이터셋이더라도 모든도메인에 완전히 전이되지 않음
        
        → 손실함수를 손본다면 더 향상되지 않을까?
        

### 5줄 요약

- ConvNet 특징은 무작위가 아니라 점진적으로 추상화되는 의미 있는 표현
- Deconvnet 시각화는 모델 성능 개선에 유용한 진단 도구
- 모델은 국소적 구조를 잘 잡아내며 깊이가 필수적임을 확인
- ImageNet 학습 모델은 Caltech 계열에서 SOTA 달성, 작은 데이터셋 벤치마크의 한계 제기
- PASCAL에서는 일반화가 제한적 → dataset bias, loss function 개선 필요

### 정리

| 항목        | 내용                                                                       |
| ----------- | -------------------------------------------------------------------------- |
| 시각화 발견 | 특징은 추상화·불변성·클래스 구별을 점차 강화                               |
| Occlusion   | 객체 위치에 민감 → 배경 맥락만 이용하지 않음                               |
| Ablation    | 특정 층보다 깊이(depth) 자체가 핵심                                        |
| 전이 성능   | Caltech-101/256에서 SOTA, PASCAL은 dataset bias로 다소 저하                |
| 시사점      | 작은 벤치마크의 유효성 재검토, loss function 개선 시 객체 탐지로 확장 가능 |

</aside>

---

### 🗃️ 데이터

### 번역

**ImageNet 2012**

This dataset consists of 1.3M/50k/100k training/validation/test examples, spread over 1000 categories.

**Caltech-101**

We follow the procedure of (Fei-fei et al., 2006) and randomly select 15 or 30 images per class for training and test on up to 50 images per class reporting the average of the per-class accuracies in Table 4.

**Caltech-256**

We follow the procedure of (Griffin et al., 2006), selecting 15, 30, 45, or 60 training images per class, reporting the average of the per-class accuracies in Table 5.

**PASCAL VOC 2012**

We used the standard training and validation images to train a 20-way softmax on top of the ImageNet-pretrained convnet. This is not ideal, as PASCAL images can contain multiple objects and our model just provides a single exclusive prediction for each image.

---

번역문

**ImageNet 2012**

이 데이터셋은 1000개의 카테고리에 걸쳐 130만 장의 학습 이미지, 5만 장의 검증 이미지, 10만 장의 테스트 이미지로 구성된다.

**Caltech-101**

(Fei-fei et al., 2006)의 절차를 따라 각 클래스당 15장 또는 30장을 무작위로 학습에 사용하고, 최대 50장을 테스트에 사용한다. 결과는 클래스별 정확도의 평균으로 보고한다.

**Caltech-256**

(Griffin et al., 2006)의 절차를 따라 각 클래스당 15, 30, 45, 60장의 이미지를 학습에 사용하고, 클래스별 정확도의 평균을 보고한다.

**PASCAL VOC 2012**

표준 학습 및 검증 이미지를 이용해, ImageNet에서 사전 학습된 ConvNet 위에 20클래스 softmax 분류기를 학습시켰다. 그러나 이 접근은 완벽하지 않은데, PASCAL 이미지는 다중 객체를 포함할 수 있지만 모델은 단일 객체 분류만 수행하기 때문이다.

<aside>

## 데이터셋(생략)

| 데이터셋        | 크기/구성                                        | 특징                      | 활용 목적                 |
| --------------- | ------------------------------------------------ | ------------------------- | ------------------------- |
| ImageNet 2012   | 130만 학습 / 5만 검증 / 10만 테스트, 1000 클래스 | 대규모, 객체 중심         | ConvNet 학습 및 성능 평가 |
| Caltech-101     | 101 클래스, 클래스당 15~30 학습, 최대 50 테스트  | 소규모, 단순 객체         | 전이학습 효과 검증        |
| Caltech-256     | 256 클래스, 클래스당 15~60 학습                  | 클래스 수 많고 다양성 큼  | 전이학습 강건성 평가      |
| PASCAL VOC 2012 | 20 클래스, 장면 내 다중 객체 포함                | 복잡한 장면, multi-object | ConvNet 일반화 한계 확인  |
</aside>

---

### 📌 서론

### 번역

원문 (Introduction 발췌)

Since their introduction by (LeCun et al., 1989) in the early 1990’s, Convolutional Networks (convnets) have demonstrated excellent performance at tasks such as hand-written digit classification and face detection. In the last year, several papers have shown that they can also deliver outstanding performance on more challenging visual classification tasks. (Ciresan et al., 2012) demonstrate state-of-the-art performance on NORB and CIFAR-10 datasets. Most notably, (Krizhevsky et al., 2012) show record beating performance on the ImageNet 2012 classification benchmark, with their convnet model achieving an error rate of 16.4%, compared to the 2nd place result of 26.1%.

Several factors are responsible for this renewed interest in convnet models: (i) the availability of much larger training sets, with millions of labeled examples; (ii) powerful GPU implementations, making the training of very large models practical and (iii) better model regularization strategies, such as Dropout (Hinton et al., 2012).

Despite this encouraging progress, there is still little insight into the internal operation and behavior of these complex models, or how they achieve such good performance. From a scientific standpoint, this is deeply unsatisfactory. Without clear understanding of how and why they work, the development of better models is reduced to trial-and-error. In this paper we introduce a visualization technique that reveals the input stimuli that excite individual feature maps at any layer in the model. It also allows us to observe the evolution of features during training and to diagnose potential problems with the model. The visualization technique we propose uses a multi-layered Deconvolutional Network (deconvnet), as proposed by (Zeiler et al., 2011), to project the feature activations back to the input pixel space. We also perform a sensitivity analysis of the classifier output by occluding portions of the input image, revealing which parts of the scene are important for classification.

Using these tools, we start with the architecture of (Krizhevsky et al., 2012) and explore different architectures, discovering ones that outperform their results on ImageNet. We then explore the generalization ability of the model to other datasets, just retraining the softmax classifier on top. As such, this is a form of supervised pre-training, which contrasts with the unsupervised pre-training methods popularized by (Hinton et al., 2006) and others (Bengio et al., 2007; Vincent et al., 2008). The generalization ability of convnet features is also explored in concurrent work by (Donahue et al., 2013).

---

번역문 (서론)

합성곱 신경망(Convolutional Networks, ConvNets)은 (LeCun et al., 1989)에 의해 1990년대 초 처음 제안된 이후, 손글씨 숫자 분류 및 얼굴 검출과 같은 과제에서 우수한 성능을 보여왔다. 최근 몇 년간, ConvNet은 더 어려운 시각적 분류 과제에서도 탁월한 성능을 발휘한다는 것이 보고되었다. (Ciresan et al., 2012)은 NORB와 CIFAR-10 데이터셋에서 최신(state-of-the-art) 성능을 달성했으며, 특히 (Krizhevsky et al., 2012)은 ImageNet 2012 분류 벤치마크에서 획기적인 기록을 세우며, 오류율 16.4%를 달성하였다(2위 결과는 26.1%).

ConvNet 모델에 대한 관심이 다시 높아진 이유는 여러 가지 요인 때문이다: (i) 수백만 개의 라벨이 달린 대규모 학습 데이터셋의 이용 가능성, (ii) 대규모 모델 학습을 가능케 한 강력한 GPU 구현, (iii) Dropout(Hinton et al., 2012)과 같은 향상된 정규화 기법이다.

그러나 이러한 고무적인 성과에도 불구하고, 이러한 복잡한 모델이 내부적으로 어떻게 동작하는지, 왜 좋은 성능을 내는지에 대해서는 거의 알려지지 않았다. 과학적 관점에서 이는 불만족스럽다. 명확한 이해가 없이는 더 나은 모델 개발이 단순한 시행착오(trial-and-error)에 의존할 수밖에 없다. 본 논문에서는 모델의 어떤 계층에서도 특정 feature map을 자극하는 입력 자극(input stimuli)을 시각화할 수 있는 기법을 제안한다. 또한 훈련 중 특징이 어떻게 진화하는지를 관찰하고, 모델의 잠재적 문제를 진단할 수 있다. 우리가 제안하는 시각화 기법은 (Zeiler et al., 2011)이 제안한 다층 Deconvolutional Network(deconvnet)를 활용하여, feature activation을 다시 입력 픽셀 공간으로 투영한다. 또한 입력 이미지의 일부를 가려서(occlusion) 분류 출력의 민감도(sensitivity)를 분석함으로써, 장면의 어떤 부분이 분류에 중요한지 확인한다.

이러한 도구들을 사용하여 우리는 (Krizhevsky et al., 2012)의 아키텍처를 출발점으로 하여 다양한 구조를 탐색하고, ImageNet에서 더 나은 결과를 내는 아키텍처를 발견하였다. 또한 softmax 분류기만 새로 학습시키는 방식으로 다른 데이터셋에 대한 일반화 능력을 탐구하였다. 이는 (Hinton et al., 2006; Bengio et al., 2007; Vincent et al., 2008) 등이 제안한 비지도 사전학습(unsupervised pre-training)과 달리 **지도 사전학습(supervised pre-training)**에 해당한다. ConvNet 특징의 일반화 능력은 (Donahue et al., 2013)의 동시대 연구에서도 탐구되고 있다.

<aside>

# 1. 서론

- 1990년 초 처음 제안된 CNN은 AlexNet(2021)부터 획기적인 모델로 발전해 왔다. 그 이유는
    - 대규모 학습 데이터셋의 이용 가능성
    - GPU의 구현
    - 정규화 기법(Dropout, etc.)
- 그러나 Blackbox(내부 메커니즘 불명확)이기 때문에, 더 나은 모델 개발이 단순한 시행착오에 의존할 수 밖에 없다. 그래서 어떤 계층에서 feature map을 시각화하는 기법을 제공하고, 이는 어떤 특징이 어떻게 진화하는지 관찰하고 모델을 진단할 수 있다.
- Deconvolutional Network(deconvnet, (Zeiler et al., 2011))을 활용하여, feature activation을 다시 피세 공간으로 투영한다. 또 이미지의 일부를 가려서 분류의 민감도를 분석하여 어떤 부분이 분류에 중요한지 확인한다.
- 이러한 기법들을 활용하여 AlexNet에서 좀 더 발전한 ZFNet을 만들었다.

### ✅ 5줄 요약

- ConvNet은 최근 ImageNet 등에서 성능을 혁신적으로 개선
- 그러나 내부 메커니즘은 여전히 불명확
- 본 논문은 deconvnet 기반 시각화로 이를 분석
- 모델 구조 개선 및 진단 가능성을 제시
- ImageNet 학습 특징이 전이학습에서도 탁월함을 보임

### 📌 정리

| 항목          | 내용                                                                                      |
| ------------- | ----------------------------------------------------------------------------------------- |
| 배경          | ConvNet 성능 급상승 (CIFAR-10, ImageNet 등)                                               |
| 한계          | 내부 동작 원리에 대한 이해 부족                                                           |
| 기여          | 시각화 기법 제안 (deconvnet, occlusion)                                                   |
| 연구 전략     | AlexNet 구조 → 개선 → 시각화 기반 진단 → 전이 성능 확인                                   |
| 사전학습 구분 | 지도 사전학습(supervised pre-training) vs 비지도 사전학습(unsupervised pre-training) 대비 |
</aside>

---

---

## 🔬 실험과정

### 📚 관련 연구

### 번역

원문 (Related Work)

Visualizing features to gain intuition about the network is common practice, but mostly limited to the 1st layer where projections to pixel space are possible. In higher layers this is not the case, and there are limited methods for interpreting activity. (Erhan et al., 2009) find the optimal stimulus for each unit by performing gradient descent in image space to maximize the unit’s activation. This requires a careful initialization and does not give any information about the unit’s invariances. Motivated by the latter’s short-coming, (Le et al., 2010) (extending an idea by (Berkes & Wiskott, 2006)) show how the Hessian of a given unit may be computed numerically around the optimal response, giving some insight into invariances. The problem is that for higher layers, the invariances are extremely complex so are poorly captured by a simple quadratic approximation. Our approach, by contrast, provides a non-parametric view of invariance, showing which patterns from the training set activate the feature map.

(Donahue et al., 2013) show visualizations that identify patches within a dataset that are responsible for strong activations at higher layers in the model. Our visualizations differ in that they are not just crops of input images, but rather top-down projections that reveal structures within each patch that stimulate a particular feature map.

---

번역문 (관련 연구)

네트워크를 직관적으로 이해하기 위해 특징(feature)을 시각화하는 것은 흔한 관행이다. 그러나 대부분은 픽셀 공간으로 직접 투영이 가능한 **1층(1st layer)**에 국한된다. 더 높은 층에서는 이러한 접근이 불가능하며, 활동(activity)을 해석할 수 있는 방법이 제한적이다.

(Erhan et al., 2009)은 각 유닛의 활성화를 최대화하기 위해 이미지 공간에서 경사하강법을 수행하여, 각 유닛에 대한 최적 자극(optimal stimulus)을 찾았다. 그러나 이는 초기화에 민감하며, 유닛의 불변성(invariances)에 관한 정보는 제공하지 못한다. 이러한 한계를 보완하기 위해, (Le et al., 2010)은 (Berkes & Wiskott, 2006)의 아이디어를 확장하여, 주어진 유닛의 최적 반응 근처에서 Hessian을 수치적으로 계산함으로써 불변성에 대한 일부 통찰을 제공했다. 그러나 높은 계층에서는 불변성이 매우 복잡하기 때문에 단순한 이차 근사(quadratic approximation)로는 잘 포착되지 않는다. 반면, 우리의 접근법은 훈련 데이터에서 어떤 패턴이 feature map을 활성화하는지를 보여주는 **비모수적(non-parametric) 관점의 불변성 시각화**를 제공한다.

(Donahue et al., 2013)은 데이터셋 내에서 높은 계층 feature map을 강하게 활성화시키는 이미지 패치를 식별하는 시각화를 제시했다. 우리의 시각화는 단순히 입력 이미지를 잘라내는 것이 아니라, **top-down projection(상향식 투영)**을 통해 특정 feature map을 자극하는 패치 내부의 구조를 드러낸다는 점에서 다르다.

<aside>

## 1.1. 관련 연구

- 대부분은 첫번째 레이어만 직접 시각화한다. 더 깊은 층에서는 이러한 접근이 제한적이다.
- 각 뉴런 유닛의 활성화를 최대화하기 위해서, 이미지 공간에서 각 유닛의 optimal stimulus(최적 자극)을 찾았으나 이는 초기화에 민감하여 유닛의 invariances(불변성)에 대한 정보는 제공하지 못한다. → 이러한 단점을 해결하기 위해 Hessian을 수치적으로 계산하여 일부 통찰을 제공했으나 깊어질수록 통찰을 제공하지 않는다(이차 근사의 단점)
- 이를 해결하기 위해 비모수적 관점의 불변성 시각화를 제공, 이미지를 잘라내는게 아니라 top-down projection을 통해 특정 피쳐맵을 자극하는 패치 내부의 드러낸다

<aside>

**헤세 행렬(Hessian Matrix)**

- 어떤 함수 f(x)의 **Hessian 행렬(Hessian matrix)**은 **이차 도함수(이계 미분)**를 모아놓은 **정방행렬**입니다.
- 예를 들어, f(x₁, x₂, …, xₙ)이 n차원 변수 x를 가진 스칼라 함수라면, Hessian H는 다음과 같이 정의됩니다.

H_ij = ∂²f / ∂xᵢ ∂xⱼ

즉,

H =

[ ∂²f/∂x₁²   ∂²f/∂x₁∂x₂  ... ]

[ ∂²f/∂x₂∂x₁ ∂²f/∂x₂²   ... ]

[   ...           ...     ... ]

- Hessian은 함수의 **곡률(curvature)**을 나타내며, 함수가 특정 지점에서 볼록(convex)한지, 오목(concave)한지, 또는 안장점(saddle point)인지를 판별하는 데 사용됩니다.
- **최적화(Optimization)**: 경사하강법(gradient descent)은 1차 미분(gradient)만 쓰지만, 2차 최적화 기법(Newton's method 등)은 Hessian을 이용해 더 빠르게 수렴합니다.
- **민감도 분석(Sensitivity Analysis)**: 특정 입력 변화가 출력에 어떤 영향을 주는지, 즉 모델의 **국소적 불변성(local invariance)**을 분석할 때 Hessian을 사용합니다.
- 뉴런의 출력이 입력 변화에 따라 얼마나 민감하게 달라지는지(곡률)를 분석
- 곡률이 낮은 방향 → 뉴런이 그 방향의 입력 변화에는 **불변(invariant)**
- 곡률이 높은 방향 → 민감하게 반응 → 중요한 패턴 방향
</aside>

### 핵심

- 시각화 연구는 초기에 시작층이나 초기층에만 국한되어 깊은 층은 해석 불가능
    - **최적 자극 탐색 (Erhan et al., 2009)**: 이미지 공간에서 경사하강법 → 활성화 극대화
        
        → 단점: 초기화 민감, 불변성 정보 없음
        
    - **Hessian 기반 불변성 분석 (Le et al., 2010)**: Hessian 근사로 불변성 파악
        
        → 단점: 고차원 층의 복잡한 불변성을 단순 이차식으로 설명 불가
        
    - **패치 기반 시각화 (Donahue et al., 2013)**: 데이터셋에서 강한 활성화를 일으키는 패치 식별
        
        → 단점: 단순 crop, feature map 내부 구조는 설명하지 못함
        

### ✅ 5줄 요약

- 과거 시각화 연구는 주로 첫 번째 층에 집중
- Erhan et al. (2009): 경사하강법으로 최적 자극 탐색, 불변성 설명 부족
- Le et al. (2010): Hessian 근사로 불변성 분석, 고층에서는 부정확
- Donahue et al. (2013): 데이터셋 패치 기반 시각화, 구조적 해석 제한
- 본 논문: Deconvnet을 통해 **비모수적, 구조적 시각화** 제공 → 고층 feature 해석 가능

### 📌 정리

| 연구                  | 방법                               | 한계                          |
| --------------------- | ---------------------------------- | ----------------------------- |
| Erhan et al. (2009)   | 이미지 공간 경사하강 → 최적 자극   | 초기화 민감, 불변성 설명 불가 |
| Le et al. (2010)      | Hessian 근사 → 불변성 분석         | 고차원 층의 복잡성 반영 못함  |
| Donahue et al. (2013) | 패치 식별 → 활성화 해석            | 단순 crop, 구조 설명 한계     |
| 본 논문               | Deconvnet 기반 top-down projection | 고층 feature 구조적 해석 가능 |
</aside>

### 📚 2. Approach

### 번역

1. 원문 (Approach)
    
    We use standard fully supervised convnet models throughout the paper, as defined by (LeCun et al., 1989) and (Krizhevsky et al., 2012). These models map a color 2D input image xi, via a series of layers, to a probability vector ŷi over the C different classes. Each layer consists of (i) convolution of the previous layer output (or, in the case of the 1st layer, the input image) with a set of learned filters; (ii) passing the responses through a rectified linear function (relu(x) = max(x, 0)); (iii) [optionally] max pooling over local neighborhoods and (iv) [optionally] a local contrast operation that normalizes the responses across feature maps. For more details of these operations, see (Krizhevsky et al., 2012) and (Jarrett et al., 2009). The top few layers of the network are conventional fully-connected networks and the final layer is a softmax classifier. Fig. 3 shows the model used in many of our experiments.
    
    We train these models using a large set of N labeled images {x, y}, where label yi is a discrete variable indicating the true class. A cross-entropy loss function, suitable for image classification, is used to compare ŷi and yi. The parameters of the network (filters in the convolutional layers, weight matrices in the fully-connected layers and biases) are trained by back-propagating the derivative of the loss with respect to the parameters throughout the network, and updating the parameters via stochastic gradient descent. Full details of training are given in Section 3.
    

---

1. 번역문 (접근 방법)
    
    우리는 본 논문 전반에서 (LeCun et al., 1989) 및 (Krizhevsky et al., 2012)에서 정의된 표준적인 **완전 지도 학습(fully supervised) ConvNet 모델**을 사용한다. 이 모델은 2차원 컬러 입력 이미지 xi를 일련의 계층(layer)을 통해 C개의 클래스에 대한 확률 벡터 ŷi로 매핑한다.
    
    각 계층은 다음 연산으로 구성된다:
    
    (i) 이전 계층 출력(혹은 1층의 경우 입력 이미지)을 학습된 필터 집합으로 **합성곱(convolution)**
    
    (ii) 출력 값을 **ReLU(Rectified Linear Unit, relu(x) = max(x, 0))** 비선형 함수에 통과
    
    (iii) [옵션] 지역적 영역에 대한 **최대 풀링(max pooling)**
    
    (iv) [옵션] feature map 전반을 정규화하는 **local contrast normalization**
    
    네트워크의 상위 몇 개 계층은 전통적인 **완전연결층(fully-connected network)**으로 구성되며, 마지막 계층은 **softmax 분류기**이다. 그림 3은 본 논문에서 사용된 모델 구조를 보여준다.
    
    모델 학습에는 N개의 라벨이 달린 이미지 {x, y}가 사용된다. 여기서 yi는 참 클래스 레이블을 나타내는 이산 변수이다. 출력 ŷi와 yi를 비교하기 위해 이미지 분류에 적합한 **교차 엔트로피 손실(cross-entropy loss)**을 사용한다. 네트워크의 파라미터(합성곱 계층의 필터, 완전연결층의 가중치 행렬, 편향)는 손실의 도함수를 네트워크 전체로 **역전파(backpropagation)** 시켜 구하고, 이를 이용해 **확률적 경사하강법(SGD)**으로 갱신한다. 학습에 대한 세부 사항은 Section 3에서 기술한다.
    
2. 원문 (2.1 Visualization with a Deconvnet)
    
    Understanding the operation of a convnet requires interpreting the feature activity in intermediate layers. We present a novel way to map these activities back to the input pixel space, showing what input pattern originally caused a given activation in the feature maps. We perform this mapping with a Deconvolutional Network (deconvnet) (Zeiler et al., 2011). A deconvnet can be thought of as a convnet model that uses the same components (filtering, pooling) but in reverse, so instead of mapping pixels to features does the opposite. In (Zeiler et al., 2011), deconvnets were proposed as a way of performing unsupervised learning. Here, they are not used in any learning capacity, just as a probe of an already trained convnet.
    
    To examine a convnet, a deconvnet is attached to each of its layers, as illustrated in Fig. 1(top), providing a continuous path back to image pixels. To start, an input image is presented to the convnet and features computed throughout the layers. To examine a given convnet activation, we set all other activations in the layer to zero and pass the feature maps as input to the attached deconvnet layer. Then we successively (i) unpool, (ii) rectify and (iii) filter to reconstruct the activity in the layer beneath that gave rise to the chosen activation. This is then repeated until input pixel space is reached.
    
    Unpooling: In the convnet, the max pooling operation is non-invertible, however we can obtain an approximate inverse by recording the locations of the maxima within each pooling region in a set of switch variables. In the deconvnet, the unpooling operation uses these switches to place the reconstructions from the layer above into appropriate locations, preserving the structure of the stimulus. See Fig. 1(bottom) for an illustration of the procedure.
    
    Rectification: The convnet uses relu non-linearities, which rectify the feature maps thus ensuring the feature maps are always positive. To obtain valid feature reconstructions at each layer (which also should be positive), we pass the reconstructed signal through a relu non-linearity.
    
    Filtering: The convnet uses learned filters to convolve the feature maps from the previous layer. To invert this, the deconvnet uses transposed versions of the same filters, but applied to the rectified maps, not the output of the layer beneath. In practice this means flipping each filter vertically and horizontally.
    
    Projecting down from higher layers uses the switch settings generated by the max pooling in the convnet on the way up. As these switch settings are peculiar to a given input image, the reconstruction obtained from a single activation thus resembles a small piece of the original input image, with structures weighted according to their contribution toward to the feature activation. Since the model is trained discriminatively, they implicitly show which parts of the input image are discriminative. Note that these projections are not samples from the model, since there is no generative process involved.
    

---

1. 번역문 (2.1 Deconvnet을 이용한 시각화)
    
    ConvNet의 동작을 이해하려면 중간 계층의 특징 활성(feature activity)을 해석해야 한다. 우리는 이러한 활동을 다시 입력 픽셀 공간으로 매핑하여, 어떤 입력 패턴이 특정 feature map을 활성화시켰는지를 보여주는 새로운 방법을 제시한다. 이 매핑은 **Deconvolutional Network (deconvnet)** (Zeiler et al., 2011)을 이용해 수행된다. Deconvnet은 ConvNet과 동일한 구성 요소(필터링, 풀링)를 사용하지만, 방향이 반대라서 **픽셀→특징**이 아닌 **특징→픽셀**로 역투영한다. (Zeiler et al., 2011)에서는 deconvnet이 비지도 학습용으로 제안되었으나, 여기서는 학습이 아닌 **이미 학습된 ConvNet을 분석(probe)**하는 용도로 사용된다.
    
    분석 과정은 다음과 같다: convnet의 각 계층에 deconvnet을 연결하면(Fig.1 상단), 입력 이미지를 convnet에 넣어 feature를 계산한 뒤, 특정 activation을 선택하고 나머지는 모두 0으로 만든다. 이 feature map을 해당 계층의 deconvnet에 입력으로 넣어 (i) unpool, (ii) rectify, (iii) filter 과정을 거쳐 바로 아래 계층의 activity를 복원한다. 이 과정을 반복하면 최종적으로 입력 픽셀 공간에 도달한다.
    
    - **Unpooling**: ConvNet의 max pooling은 비가역적이다. 이를 보완하기 위해, pooling 시 각 영역의 최댓값 위치를 switch 변수로 기록한다. Deconvnet에서는 이 switch를 사용하여 상위 계층의 복원 결과를 정확한 위치에 배치한다. 이렇게 하면 자극 구조(stimulus structure)가 보존된다.
    - **Rectification**: ConvNet은 ReLU 비선형 함수를 사용하여 feature map을 항상 양수로 만든다. Deconvnet에서도 마찬가지로 ReLU를 적용해 복원된 feature가 양수로 유지되도록 한다.
    - **Filtering**: ConvNet은 학습된 필터로 feature map을 합성곱한다. 이를 역전하기 위해 deconvnet은 **같은 필터의 전치(transposed) 버전**을 사용한다(즉, 필터를 수직·수평으로 뒤집는다).
    
    상위 계층에서 하위 계층으로의 투영 시, ConvNet 학습 과정에서 기록된 pooling switch가 사용된다. 이 switch는 특정 입력 이미지에 종속적이므로, 한 개의 activation으로부터 얻은 복원 결과는 원래 입력 이미지의 일부분과 유사하다. 이는 feature가 입력 이미지의 어떤 구조에 의해 자극되는지를 가중치 형태로 보여준다. 모델은 판별적(discriminative)으로 학습되므로, 이 시각화는 결국 입력 이미지에서 **어떤 부분이 판별에 중요했는지**를 드러낸다. 주의할 점은, 이러한 projection은 **생성 모델의 샘플이 아니라 단순히 역투영된 구조**라는 것이다.
    

<aside>

# 2. Approach

- **지도학습**
- **Layer구조 : Conv → ReLU → (옵션) Max Pooling : Local에 대해 → (옵션) Local Contrast Normalization : feature map 전반을 정규화**
- 네트워크가 깊어지면 소수의 fc layer로 구성, 마지막은 **softmax clf.**
- **아키텍쳐**

![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image.png)

- 손실함수 : **Cross-entropy**
- Optimizer : **SGD(mini-batch)**

### ✅ 5줄 요약

- ConvNet은 입력 이미지를 계층적 변환을 통해 클래스 확률로 매핑
- 계층은 합성곱, ReLU, 풀링, 정규화로 구성
- 상위 계층은 fully-connected + softmax 분류기
- 교차 엔트로피 손실과 backpropagation으로 학습
- 확률적 경사하강법(SGD)으로 파라미터 최적화

### 📌 정리

| 요소          | 설명                              |
| ------------- | --------------------------------- |
| 입력          | 컬러 2D 이미지 (xi)               |
| 출력          | 클래스 확률 벡터 (ŷi)             |
| 계층 구성     | 합성곱 + ReLU + (풀링) + (정규화) |
| 상위 구조     | Fully-connected layers            |
| 최종 분류기   | Softmax                           |
| 손실 함수     | Cross-entropy                     |
| 학습 알고리즘 | Backpropagation + SGD             |

## 2.1 Visualization with a Deconvnet

- **Deconvolutional Network : Conv와 동일한 구성이지만 픽셀 → 특징 투영과 반대로 특징 → 픽셀 투영을한다.**
    - 기존에는 비지도학습용으로 제안되었지만, 여기서는 **이미 학습된 모델을 probe**하는 용도로 사용된다.
- 분석과정
    1. **convnet에 deconvnet을 연결한다.**
    2. **이미지를 convnet에 넣은 후 feature을 계산하고, 특정 활성을 선택하고 나머지는 모두 0으로 만든다.**
    3. **이 feature을 deconvnet에 넣어서 복원한다.**
        1. **unpool**
        2. **rectify**
        3. **filter 과정을 거쳐 바로 아래 계층의 activity를 복원**
- 핵심연산
    - **Unpooling** : max pooling은 비가역적인데, 이를 해결하기 위해 **pooling시 각 영역의 최대값의 위치를 switch변수로 기록**하고, 그 값을 바탕으로 복원 결과를 정확한 위치에 배치
    - **Rectification** : ConvNet처럼 **ReLU**를 사용하여 복원(**복원된 feature가 양수**)
    - **Filtering** : **ConvNet의 학습된 필터를 전치 버전**을 사용한다.(수직, 수평방향)
- 한개의 활성으로부터 얻는 결과는 이미지의 어떤 구조의 일부분과 유사하다. **모델은 판별적으로 학습되므로, 입력이미지에서 어떤부분이 중요했는지를 드러**낸다. 다만 **생성모델의 샘플이 아니라, 단순히 역투영**된다는 점이다. → **생성모델이 아니라 역투영**이다. → 즉 모델이 입력구조의 어떤부분을 가지고 판단했는지 직관적으로 보여준다.
- 분석 구조
    
    ![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%201.png)
    

### ✅ 5줄 요약

- Deconvnet을 이용해 중간층 feature map을 입력 픽셀 공간으로 복원
- 과정: (Unpool → ReLU → Filter) 반복
- pooling switch로 원래 자극 위치 복원
- 결과: 특정 activation이 입력 이미지의 어떤 부분에 의해 유발되었는지 확인 가능
- 이는 생성이 아닌 판별 기반 projection → ConvNet의 판별 근거를 직관적으로 시각화

### 📌 정리

| 단계      | ConvNet (정방향) | Deconvnet (역방향)       |
| --------- | ---------------- | ------------------------ |
| Pooling   | Max pooling      | Unpooling (switch 사용)  |
| ReLU      | ReLU             | ReLU                     |
| Filtering | Learned filter   | Transposed filter (flip) |
| 결과      | Feature map      | 입력 공간 복원           |
</aside>

### 📚 3. Training Detail

### 번역

원문 (3. Training Details)

We now describe the large convnet model that will be visualized in Section 4. The architecture, shown in Fig. 3, is similar to that used by (Krizhevsky et al., 2012) for ImageNet classification. One difference is that the sparse connections used in Krizhevsky’s layers 3,4,5 (due to the model being split across 2 GPUs) are replaced with dense connections in our model.

Other important differences relating to layers 1 and 2 were made following inspection of the visualizations in Fig. 6, as described in Section 4.1.

The model was trained on the ImageNet 2012 training set (1.3 million images, spread over 1000 different classes). Each RGB image was preprocessed by resizing the smallest dimension to 256, cropping the center 256x256 region, subtracting the per-pixel mean (across all images) and then using 10 different sub-crops of size 224x224 (corners + center with(out) horizontal flips). Stochastic gradient descent with a mini-batch size of 128 was used to update the parameters, starting with a learning rate of 10−2, in conjunction with a momentum term of 0.9. We anneal the learning rate throughout training manually when the validation error plateaus. Dropout (Hinton et al., 2012) is used in the fully connected layers (6 and 7) with a rate of 0.5. All weights are initialized to 10−2 and biases are set to 0.

Visualization of the first layer filters during training reveals that a few of them dominate, as shown in Fig. 6(a). To combat this, we renormalize each filter in the convolutional layers whose RMS value exceeds a fixed radius of 10−1 to this fixed radius. This is crucial, especially in the first layer of the model, where the input images are roughly in the [-128,128] range. As in (Krizhevsky et al., 2012), we produce multiple different crops and flips of each training example to boost training set size. We stopped training after 70 epochs, which took around 12 days on a single GTX580 GPU, using an implementation based on (Krizhevsky et al., 2012).

---

번역문 (훈련 세부 사항)

이제 Section 4에서 시각화될 대규모 ConvNet 모델을 설명한다. 아키텍처(Fig. 3)는 ImageNet 분류를 위해 (Krizhevsky et al., 2012)이 사용한 것과 유사하다. 그러나 Krizhevsky 모델에서 2개의 GPU로 분산 학습을 했기 때문에 3,4,5층은 **희소 연결(sparse connections)**을 사용했지만, 본 논문에서는 이를 **밀집 연결(dense connections)**로 대체하였다.

또한 Fig. 6의 시각화 결과(Section 4.1 참고)를 기반으로 1층과 2층에 중요한 수정이 이루어졌다.

모델은 ImageNet 2012 학습셋(130만 장, 1000 클래스)에서 훈련되었다. 각 RGB 이미지는 다음과 같이 전처리되었다:

- 가장 짧은 변을 256으로 리사이즈
- 중앙 256x256 영역을 크롭
- 전체 이미지의 픽셀별 평균을 빼줌 (mean subtraction)
- 224x224 크기의 서브 크롭 10개(모서리 4개 + 중앙, 좌우 반전 포함) 생성

최적화는 확률적 경사하강법(SGD)으로 수행되었으며, mini-batch 크기는 128이었다. 학습률은 10^-2로 시작하고, 모멘텀(momentum) 0.9를 사용하였다. 검증 오류가 plateau에 도달하면 학습률을 수동으로 감소시켰다. 완전 연결층(6,7층)에서는 Dropout(Hinton et al., 2012)을 0.5 비율로 적용하였다. 모든 가중치는 10^-2로 초기화하였고, 편향은 0으로 설정하였다.

훈련 중 1층 필터를 시각화한 결과 일부 필터가 지나치게 지배적임을 확인하였다(Fig. 6(a)). 이를 방지하기 위해 RMS 값이 10^-1을 초과하는 필터는 강제로 재정규화(renormalization)하여 해당 반경으로 맞췄다. 이는 특히 입력 이미지가 [-128,128] 범위에 존재하는 1층에서 중요하다. (Krizhevsky et al., 2012)와 마찬가지로 학습 예제를 증강하기 위해 여러 크롭과 플립을 사용했다. 학습은 70 epoch 동안 수행되었으며, GTX580 GPU 한 장에서 약 12일이 소요되었다. 구현은 Krizhevsky et al.(2012)을 기반으로 했다.

<aside>

# 3. 학습 세부 사항

- AlexNet의 GPU분산 학습을 했기때문에 3, 4, 5층은 **sparse connections(희소 연결 - 우리가 기존에 아는 sparse가 아니라, GPU로 인한 모델 아키텍쳐 분산을 의미)**을 사용했지만, 본 모델에서는 **dense connections**로 대체
    - 또한 1층과 2층에 세부 수정이 이루어짐(시각화 결과를 기반으로)
- AlexNet과 동일하게 증강
- 1층 필터를 시각화 한결과, 일부 필터가 지나치게 지배적이기 때문에 이를 방지하기 위해 **RMS값이 10^-1을 초과하는 필터는 강제로 renormalization(재정규화, 0.1로 만듦, 균형 유지)**한다. 이는 [-128, 128]인 1층에서 중요하다.

### ✅ 5줄 요약

- ImageNet 2012에서 학습 (130만 장, 1000 클래스)
- 전처리: 리사이즈·크롭·평균 제거·데이터 증강
- 학습: SGD, mini-batch=128, learning rate=0.01 시작, momentum=0.9, Dropout=0.5
- 필터 재정규화로 특정 필터 지배 방지
- 70 epoch, GPU 1장, 약 12일 소요

---

### 📌 정리

| 항목      | 설정                                                             |
| --------- | ---------------------------------------------------------------- |
| 데이터    | ImageNet 2012 (1.3M, 1000 클래스)                                |
| 전처리    | 리사이즈 256, 중앙 크롭, 평균 제거, 224x224 서브 크롭 10개, flip |
| 최적화    | SGD, batch=128, lr=0.01, momentum=0.9                            |
| 정규화    | Dropout(0.5), filter RMS clipping(0.1)                           |
| 구조 차이 | AlexNet sparse → dense 연결                                      |
| 학습 시간 | 70 epoch, GTX580 GPU, 12일                                       |
</aside>

### 📚 4. Convnet Visualization

### 번역

1. 원문 (Section 4 발췌)
    
    Using the model described in Section 3, we now use the deconvnet to visualize the feature activations on the ImageNet validation set.
    
    **Feature Visualization:** Fig. 2 shows feature visualizations from our model once training is complete. However, instead of showing the single strongest activation for a given feature map, we show the top 9 activations. Projecting each separately down to pixel space reveals the different structures that excite a given feature map, hence showing its invariance to input deformations. Alongside these visualizations we show the corresponding image patches. … For example, in layer 5, row 1, col 2, the patches appear to have little in common, but the visualizations reveal that this particular feature map focuses on the grass in the background, not the foreground objects.
    
    The projections from each layer show the hierarchical nature of the features in the network. Layer 2 responds to corners and other edge/color conjunctions. Layer 3 has more complex invariances, capturing similar textures. Layer 4 shows significant variation, but is more class-specific (e.g. dog faces, bird legs). Layer 5 shows entire objects with significant pose variation (e.g. keyboards, dogs).
    
    **Feature Evolution during Training:** Fig. 4 visualizes the progression during training of the strongest activation … The lower layers converge within a few epochs, but the upper layers only develop after 40–50 epochs.
    
    **Feature Invariance:** Fig. 5 shows how features change under translation, rotation, and scaling. Small transformations affect the first layer strongly, but higher layers are more stable, showing quasi-linear changes.
    

---

1. 번역문 (ConvNet 시각화)
    
    Section 3에서 설명한 모델을 사용하여, 이제 우리는 deconvnet을 통해 ImageNet 검증셋의 feature activation을 시각화한다.
    
    **특징 시각화 (Feature Visualization):** Fig. 2는 학습이 완료된 모델의 특징 시각화 결과를 보여준다. 각 feature map에서 단일 최강 활성화만 보여주는 대신, 상위 9개의 activation을 표시하였다. 각각을 픽셀 공간으로 투영하면 해당 feature map을 흥분시키는 다양한 구조가 드러나며, 이를 통해 입력 변형(input deformation)에 대한 불변성(invariance)을 확인할 수 있다. 이와 함께 해당되는 원본 이미지 패치도 나란히 보여준다. 예를 들어, 5층(layer 5)의 (row 1, col 2)에서는 원본 패치들이 서로 크게 다르지만, 시각화는 해당 feature map이 **전경 객체가 아니라 배경의 풀(grasstexture)**에 주목한다는 것을 드러낸다.
    
    각 층에서 얻은 projection은 네트워크 특징의 **계층적 성격(hierarchical nature)**을 보여준다.
    
    - 2층(layer 2): 모서리(corner), 색상/엣지 결합 구조에 반응
    - 3층(layer 3): 텍스처(texture)와 같은 복잡한 불변성 패턴 포착
    - 4층(layer 4): 클래스 특이적(class-specific) 패턴 (예: 개 얼굴, 새 다리)
    - 5층(layer 5): 포즈 변화가 큰 전체 객체 (예: 키보드, 개 전체 모습)
    
    **훈련 중 특징의 진화 (Feature Evolution during Training):** Fig. 4는 훈련 과정에서 strongest activation을 시각화한다. 하위 계층은 몇 epoch 내에 수렴하지만, 상위 계층은 40~50 epoch 이후에야 발달한다.
    
    **특징 불변성 (Feature Invariance):** Fig. 5는 특징이 평행 이동(translation), 회전(rotation), 크기 변화(scaling)에 따라 어떻게 달라지는지를 보여준다. 작은 변환은 1층에서 큰 영향을 주지만, 상위 층으로 갈수록 더 안정적(quasi-linear)인 반응을 보인다.
    
2. 원문 (Section 4.1 발췌)
    
    While visualization of a trained model gives insight into its operation, it can also assist with selecting good architectures in the first place. By visualizing the first and second layers of Krizhevsky et al.’s architecture (Fig. 6(b) & (d)), various problems are apparent. The first layer filters are a mix of extremely high and low frequency information, with little coverage of the mid frequencies. Additionally, the 2nd layer visualization shows aliasing artifacts caused by the large stride 4 used in the 1st layer convolutions.
    
    To remedy these problems, we (i) reduced the 1st layer filter size from 11x11 to 7x7 and (ii) made the stride of the convolution 2, rather than 4. This new architecture retains much more information in the 1st and 2nd layer features, as shown in Fig. 6(c) & (e). More importantly, it also improves the classification performance as shown in Section 5.1.
    

---

1. 번역문 (4.1 아키텍처 선택)
    
    훈련된 모델의 시각화는 내부 동작에 대한 통찰을 줄 뿐 아니라, 처음부터 더 좋은 아키텍처를 선택하는 데에도 도움이 된다. Krizhevsky et al.의 아키텍처의 1층과 2층을 시각화한 결과(Fig. 6(b), (d)), 몇 가지 문제가 드러났다.
    
    - **1층 필터**: 매우 고주파(high frequency)와 저주파(low frequency) 정보가 혼합되어 있으며, **중간 주파수(mid frequency) 영역**의 커버리지가 부족하다.
    - **2층 특징**: 1층 합성곱에서 stride=4를 사용한 탓에 **aliasing artifact(샘플링 왜곡)**가 나타난다.
    
    이 문제를 해결하기 위해 저자들은:
    
    1. **1층 필터 크기**를 11x11에서 7x7로 줄이고,
    2. **합성곱 stride**를 4에서 2로 축소하였다.
    
    이 새로운 아키텍처는 Fig. 6(c), (e)에 보이듯이 1층과 2층에서 훨씬 더 많은 정보를 보존한다. 더 중요한 것은, 이러한 변경이 ImageNet 분류 성능을 향상시켰다는 점이다(Section 5.1 참고).
    
2. 원문 (Section 4.2 발췌)
    
    With image classification approaches, a natural question is if the model is truly identifying the location of the object in the image, or just using the surrounding context. Fig. 7 attempts to answer this question by systematically occluding different portions of the input image with a grey square, and monitoring the output of the classifier. The examples clearly show the model is localizing the objects within the scene, as the probability of the correct class drops significantly when the object is occluded.
    
    Fig. 7 also shows visualizations from the strongest feature map of the top convolution layer, in addition to activity in this map (summed over spatial locations) as a function of occluder position. When the occluder covers the image region that appears in the visualization, we see a strong drop in activity in the feature map. This shows that the visualization genuinely corresponds to the image structure that stimulates that feature map, hence validating the other visualizations shown in Fig. 4 and Fig. 2.
    

---

1. 번역문 (4.2 Occlusion Sensitivity)
    
    이미지 분류 접근법에서 자연스러운 의문은, 모델이 실제로 이미지 속 객체의 위치를 인식하는지, 아니면 단순히 주변 맥락(context)만 사용하는지이다. Fig. 7은 입력 이미지의 서로 다른 영역을 회색 정사각형으로 **체계적으로 가려(occlusion)** 보면서 분류기의 출력을 관찰하여 이 질문에 답하고자 한다.
    
    그 결과, 객체가 가려지면 올바른 클래스 확률이 크게 떨어지는 것을 확인할 수 있었다. 이는 모델이 장면 내에서 실제 객체 위치를 찾아내고 있다는 것을 보여준다.
    
    또한 Fig. 7은 최상위 합성곱 계층(top convolutional layer)의 가장 강력한 feature map의 시각화를 함께 보여주는데, 이 feature map의 활성화(공간 전체 합)를 occluder 위치에 따라 측정하였다. 가려진 부분이 시각화에서 나타난 객체 구조와 겹치면 feature map 활성도가 크게 떨어졌다. 이는 시각화가 실제로 해당 feature map을 자극하는 이미지 구조와 정확히 대응하고 있음을 보여주며, Fig. 4와 Fig. 2에서 제시된 다른 시각화 결과를 뒷받침한다.
    
2. 원문 (Section 4.3 발췌)
    
    Deep models differ from many existing recognition approaches in that there is no explicit mechanism for establishing correspondence between specific object parts in different images (e.g. faces have a particular spatial configuration of the eyes and nose). However, an intriguing possibility is that deep models might be implicitly computing them. To explore this, we take 5 randomly drawn dog images with frontal pose and systematically mask out the same part of the face in each image (e.g. all left eyes, see Fig. 8). For each image i, we then compute: εli = xli − x̃li, where xli and x̃li are the feature vectors at layer l for the original and occluded images respectively. We then measure the consistency of this difference vector ε between all related image pairs (i, j): Δl = ∑ H(sign(εli), sign(εlj)), where H is Hamming distance. A lower value indicates greater consistency in the change resulting from the masking operation, hence tighter correspondence between the same object parts in different images.
    
    In Table 1 we compare the Δ score for three parts of the face (left eye, right eye and nose) to random parts of the object, using features from layer l = 5 and l = 7. The lower score for these parts, relative to random object regions, for the layer 5 features show the model does establish some degree of correspondence.
    

---

1. 번역문 (4.3 대응 분석)
    
    기존의 많은 인식 접근법은 서로 다른 이미지에서 **특정 객체 부위 간의 대응(correspondence)**을 명시적으로 설정한다 (예: 얼굴은 눈과 코가 특정 공간적 배열을 가짐). 반면, 딥 모델은 이러한 메커니즘을 명시적으로 포함하지 않는다. 하지만 흥미로운 가능성은, 딥 모델이 암묵적으로 이러한 대응을 학습하고 있을 수 있다는 것이다.
    
    이를 탐구하기 위해, 정면(frontal pose) 상태의 개(dog) 이미지 5장을 무작위로 선택하고, 모든 이미지에서 동일한 얼굴 부위(예: 왼쪽 눈)를 체계적으로 마스킹하였다(Fig. 8 참조). 각 이미지 i에 대해, 원본 이미지와 마스크된 이미지의 layer l 특징 벡터를 각각 xli, x̃li라 하면, 그 차이 εli = xli − x̃li 를 계산한다. 이후, 모든 관련 이미지 쌍 (i, j)에 대해 차이 벡터 ε의 일관성을 해밍 거리(Hamming distance) 기반으로 측정한다:
    
    Δl = ∑ H(sign(εli), sign(εlj))
    
    Δ 값이 낮을수록 동일한 부위를 마스킹했을 때의 변화가 더 일관적이라는 의미이며, 따라서 다른 이미지 간에 해당 부위가 더 강하게 대응되고 있음을 나타낸다.
    
    Table 1은 세 가지 얼굴 부위(왼쪽 눈, 오른쪽 눈, 코)와 무작위 부위를 비교한 결과이다. Layer 5 특징에서는 무작위 부위에 비해 눈과 코에서 Δ 값이 더 낮게 나타났으며, 이는 모델이 일정 수준의 대응성을 형성하고 있음을 보여준다. Layer 7에서는 이 차이가 줄어드는데, 이는 상위층이 주로 **클래스 판별(breed discrimination)**에 집중하기 때문으로 보인다.
    

<aside>

![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%202.png)

# 4. 시각화

- 각 층별 특징 : 그 레이어에서의 복원이 아니라, 역으로 전부다 거친 후 복원, 따라서 깊어질수록 해상도가 높아진다. e.g. 2층의 경우 2 → 1, 5층의 경우 5 → 4 → … → 1 즉 하위계층(초기층)과 상위계층(후반층)을 비교
    - 2층(layer 2): 모서리(corner), 색상/엣지 결합 구조에 반응
    - 3층(layer 3): 텍스처(texture)와 같은 복잡한 불변성 패턴 포착
    - 4층(layer 4): 클래스 특이적(class-specific) 패턴 (예: 개 얼굴, 새 다리)
    - 5층(layer 5): 포즈 변화가 큰 전체 객체 (예: 키보드, 개 전체 모습)
- **입력 변형(input deformation)에 대한 불변성(invariance)을 확인 → 작은 변화는 하위층에서 큰 효과를 주지만 상위층에는 quasi-linear(안정적)인 반응을 보여줌**
    
    ![스크린샷 2025-08-28 오전 12.47.55.jpg]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-08-28_%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB_12.47.55.jpg)
    
- **계층적 성격(hierarchical nature)**
- **하위 계층은 소수의 epochs만에 수렴, 상위는 오래걸림**
    
    ![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%203.png)
    
- **즉 ConvNet은 깊게 학습될수록 추상적인 특징을 학습**

### ✅ 5줄 요약

- Deconvnet으로 각 층의 feature map을 픽셀 공간으로 복원
- 하위층: 저수준 특징(엣지, 코너), 중간층: 텍스처, 상위층: 객체/부위
- 훈련 초기에 하위층은 빠르게 수렴, 상위층은 늦게 발달
- 입력 변형에 대해 상위층은 더 안정적 → 불변성 확보
- ConvNet은 계층적으로 의미 있는 표현을 학습함을 실증

---

### 📌 정리

| 층 (Layer) | 주요 특징          | 시각화 결과               |
| ---------- | ------------------ | ------------------------- |
| Layer 2    | 코너, 엣지+색 결합 | 기본 기하학적 구조 감지   |
| Layer 3    | 텍스처, 반복 패턴  | 메시(mesh), 텍스트 인식   |
| Layer 4    | 클래스 특이적 부위 | 개 얼굴, 새 다리 등       |
| Layer 5    | 전체 객체          | 개, 키보드 등 다양한 포즈 |

## 4.1. Architecture Selection

- AlexNet의 문제점
    - 1층 : 매우 고주파(high frequency)와 저주파(low frequency) 정보가 혼합되어 있으며, **중간 주파수(mid frequency) 영역의  커버가 부족**
    - 2층 : 1층 합성곱에서 stride=4를 사용한 탓에 **aliasing artifact(샘플링 왜곡)**이 발생
- ZFNet 수정사항
    - **1층 필터 크기**를 11x11에서 7x7로 줄이고,
    - **합성곱 stride**를 4에서 2로 축소하였다.

![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%204.png)

### ✅ 5줄 요약

- 시각화를 통해 AlexNet의 1·2층에서 문제점 발견
- 1층: 중간 주파수 부족, 2층: stride=4로 인한 aliasing
- 개선: 11x11 필터 → 7x7, stride=4 → 2
- 결과: 정보 보존 ↑, aliasing ↓
- 성능 또한 개선됨 (Section 5.1)

---

### 📌 정리

| 문제 (AlexNet)                          | 해결책 (Zeiler & Fergus) | 결과                       |
| --------------------------------------- | ------------------------ | -------------------------- |
| 1층: 고·저주파 위주, mid-frequency 부족 | 필터 크기 11x11 → 7x7    | 더 균형 잡힌 필터          |
| 2층: stride=4 → aliasing 발생           | stride=4 → 2             | aliasing 제거, 정보 보존 ↑ |
| 성능                                    | 개선 전 AlexNet          | 개선 후 더 낮은 오류율     |

## 4.2 Occlusion Sensitivity

- 객체의 위치를 인식하는지, 주변 맥락만 사용하는지를 확인해보는 실험으로, 결론적으로 **객체를 지우는 경우 올바른 클래스 확률이 크게 떨어진다**.
- 즉 모델은 **객체 자체에 집중하여,** 객체 탐지의 가능성에 대해 시사

![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%205.png)

### ✅ 5줄 요약

- Occlusion으로 모델이 배경이 아닌 객체 위치에 의존함을 확인
- 객체 부위가 가려지면 올바른 클래스 확률 급락
- top conv layer의 feature map 활성도도 함께 급락
- Deconvnet 시각화가 진짜 자극 구조를 보여준다는 검증
- ConvNet은 장면 context보다 객체 local structure에 민감

---

### 📌 정리

| 질문                                       | 방법                                   | 결과                                      | 의미                                        |
| ------------------------------------------ | -------------------------------------- | ----------------------------------------- | ------------------------------------------- |
| 객체 위치를 보는가, 배경 context를 보는가? | 이미지 영역을 순차적으로 가림          | 객체 부분이 가려지면 확률 급락            | ConvNet은 객체 local structure에 집중       |
| 시각화 신뢰성 검증                         | top conv layer feature map 활성도 관찰 | occluder가 해당 구조를 가리면 활성도 급락 | Deconvnet 시각화 결과가 실제 feature와 일치 |

## 4.3. Correspondence Analysis

- 기존의 모델들은 **특정 객체 부위간의 대응(얼굴 속, 코와 눈)**을 명시적으로 설정하게 되는데, 딥러닝 모델의 경우에는 이러한 메커니즘이 **암묵적으로 대응**한다는 점이다.

✅ 방법

1. 개 얼굴 이미지 5장 선택
2. 동일한 위치(예: 왼쪽 눈)를 가려서 원본과 feature 차이 εli 계산
3. 이미지 쌍(i, j) 간의 차이 벡터 일관성을 해밍 거리로 측정 (Δl)
4. 특정 부위 vs 무작위 부위 비교

✅ 결과

- Layer 5: 눈·코 같은 의미 있는 부위에서 Δ 값이 낮음 → 대응성 확보
- Layer 7: breed 판별에 집중하므로 Δ 값이 무작위 부위와 유사 → 부위 대응 정보 약화

✅ 의미

- ConvNet은 명시적으로 correspondence를 정의하지 않아도, 중간층에서 **객체 부위 간 암묵적 대응**을 학습
- 그러나 깊은 층으로 갈수록 이 정보는 사라지고, **클래스 구분에 더 특화**됨

<aside>

### 해밍 거리

두 벡터가 있을때, 서로 다른 위치의 원소 개수를 새는 거리 측도

값이 작을수록 서로 다른 이미지에서도 같은 부위가 공통된 역할을 하고 있음을 의미

</aside>

![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%206.png)

### ✅ 5줄 요약

- ConvNet이 암묵적으로 객체 부위 대응을 학습하는지 실험
- 동일 부위를 가려 feature 변화량을 비교
- Layer 5: 눈·코에서 변화가 일관적 → 대응성 존재
- Layer 7: breed 판별로 치중 → 대응성 감소
- ConvNet은 중간층에서 correspondence를 형성하지만, 깊은 층에서는 판별적 특징에 집중

---

### 📌 정리

| 층 (Layer) | 부위            | Δ 값 결과       | 의미                                   |
| ---------- | --------------- | --------------- | -------------------------------------- |
| Layer 5    | 눈·코 vs 무작위 | 눈·코 Δ 더 낮음 | 부위 간 대응성 확보                    |
| Layer 7    | 눈·코 vs 무작위 | 유사            | breed 구분에 집중, correspondence 약화 |
</aside>

### 📚 5. Experiments

### 번역

1. 원문 (5.1 ImageNet 2012)
    
    This dataset consists of 1.3M/50k/100k training/validation/test examples, spread over 1000 categories. Table 2 shows our results on this dataset.
    
    Using the exact architecture specified in (Krizhevsky et al., 2012), we attempt to replicate their result on the validation set. We achieve an error rate within 0.1% of their reported value on the ImageNet 2012 validation set.
    
    Next we analyze the performance of our model with the architectural changes outlined in Section 4.1 (7×7 filters in layer 1 and stride 2 convolutions in layers 1 & 2). This model, shown in Fig. 3, significantly outperforms the architecture of (Krizhevsky et al., 2012), beating their single model result by 1.7% (test top-5). When we combine multiple models, we obtain a test error of 14.8%, the best published performance on this dataset (despite only using the 2012 training set).
    
    We note that this error is almost half that of the top non-convnet entry in the ImageNet 2012 classification challenge, which obtained 26.2% error.
    

---

1. 번역문 (5.1 ImageNet 2012)
    
    이 데이터셋은 1000개의 카테고리에 걸쳐 130만 장의 학습 이미지, 5만 장의 검증 이미지, 10만 장의 테스트 이미지로 구성된다. Table 2는 이 데이터셋에서의 실험 결과를 보여준다.
    
    먼저 Krizhevsky et al. (2012)의 아키텍처를 그대로 재현하여 검증셋 결과를 비교했으며, 보고된 값과 0.1% 이내로 일치하는 오류율을 달성하였다.
    
    다음으로, Section 4.1에서 제안된 아키텍처 수정(1층 필터를 11×11에서 7×7로 축소, stride를 4에서 2로 변경)을 적용한 모델을 분석했다. Fig. 3에 나타난 이 새로운 아키텍처는 Krizhevsky 아키텍처보다 성능이 크게 향상되어, 단일 모델 기준 테스트 Top-5 오류율을 1.7% 개선했다.
    
    또한 여러 모델을 앙상블한 경우 테스트 오류율은 14.8%로 감소했으며, 이는 2012 학습셋만 사용한 모델 중 당시 최고 성능이었다. 이 오류율은 ImageNet 2012 대회의 비-ConvNet 방법(26.2%)의 절반 수준에 해당한다.
    
2. 원문 (5.2 Feature Generalization, 논문 본문 그대로)
    
    The experiments above show the importance of the convolutional part of our ImageNet model in obtaining state-of-the-art performance. This is supported by the visualizations of Fig. 2 which show the complex invariances learned in the convolutional layers. We now explore the ability of these feature extraction layers to generalize to other datasets, namely Caltech-101 (Fei-fei et al., 2006), Caltech-256 (Griffin et al., 2006) and PASCAL VOC 2012. To do this, we keep layers 1-7 of our ImageNet-trained model fixed and train a new softmax classifier on top (for the appropriate number of classes) using the training images of the new dataset. Since the softmax contains relatively few parameters, it can be trained quickly from a relatively small number of examples, as is the case for certain datasets.
    
    The classifiers used by our model (a softmax) and other approaches (typically a linear SVM) are of similar complexity, thus the experiments compare our feature representation, learned from ImageNet, with the hand-crafted features used by other methods. It is important to note that both our feature representation and the hand-crafted features are designed using images beyond the Caltech and PASCAL training sets. For example, the hyper-parameters in HOG descriptors were determined through systematic experiments on a pedestrian dataset (Dalal & Triggs, 2005). We also try a second strategy of training a model from scratch, i.e. resetting layers 1-7 to random values and train them, as well as the softmax, on the training images of the dataset.
    
    One complication is that some of the Caltech datasets have some images that are also in the ImageNet training data. Using normalized correlation, we identified these few “overlap” images and removed them from our Imagenet training set and then retrained our Imagenet models, so avoiding the possibility of train/test contamination.
    
    **Caltech-101:** We follow the procedure of (Fei-fei et al., 2006) and randomly select 15 or 30 images per class for training and test on up to 50 images per class reporting the average of the per-class accuracies in Table 4, using 5 train/test folds. Training took 17 minutes for 30 images/class. The pre-trained model beats the best reported result for 30 images/class from (Bo et al., 2013) by 2.2%. The convnet model trained from scratch however does terribly, only achieving 46.5%.
    
    **Caltech-256:** We follow the procedure of (Griffin et al., 2006), selecting 15, 30, 45, or 60 training images per class, reporting the average of the per-class accuracies in Table 5. Our ImageNet-pretrained model beats the current state-of-the-art results obtained by Bo et al. (Bo et al., 2013) by a significant margin: 74.2% vs 55.2% for 60 training images/class. However, as with Caltech-101, the model trained from scratch does poorly. In Fig. 9, we explore the “one-shot learning” (Fei-fei et al., 2006) regime. With our pre-trained model, just 6 Caltech-256 training images are needed to beat the leading method using 10 times as many images. This shows the power of the ImageNet feature extractor.
    
    **PASCAL 2012:** We used the standard training and validation images to train a 20-way softmax on top of the ImageNet-pretrained convnet. This is not ideal, as PASCAL images can contain multiple objects and our model just provides a single exclusive prediction for each image. Table 6 shows the results on the test set. The PASCAL and ImageNet images are quite different in nature, the former being full scenes unlike the latter. This may explain our mean performance being 3.2% lower than the leading (Yan et al., 2012) result, however we do beat them on 5 classes, sometimes by large margins.
    
    ---
    

2. 번역문 (5.2 특징 일반화, 전체)

앞선 실험은 ImageNet 모델의 합성곱 부분이 최첨단 성능을 달성하는 데 중요함을 보여준다. Fig. 2의 시각화는 합성곱 층이 복잡한 불변성을 학습함을 보여주며 이를 뒷받침한다. 이제 우리는 이러한 특징 추출 층이 다른 데이터셋에서도 얼마나 잘 일반화되는지를 탐구한다. 이를 위해 ImageNet에서 학습된 모델의 1~7층을 고정(freeze)하고, 새로운 데이터셋의 학습 이미지를 사용해 상단에 새로운 softmax 분류기만 학습시켰다. softmax는 파라미터 수가 상대적으로 적으므로, 학습 예시가 적은 데이터셋에서도 빠르게 훈련할 수 있다.

우리 모델이 사용하는 softmax 분류기와 기존 접근법이 사용하는 선형 SVM은 복잡성이 유사하다. 따라서 이 실험은 우리가 ImageNet에서 학습한 feature 표현과 기존 수작업 특징(hand-crafted features)을 직접 비교하는 것이다. 중요한 점은, 우리의 feature 표현과 수작업 feature 모두 Caltech과 PASCAL 학습셋 외부의 이미지를 기반으로 설계되었다는 것이다. 예를 들어, HOG descriptor의 하이퍼파라미터는 보행자 데이터셋(Dalal & Triggs, 2005)에서의 체계적 실험으로 결정되었다. 또한 우리는 두 번째 전략으로 “처음부터 학습”도 시도했는데, 이는 1~7층을 무작위로 초기화하고 softmax까지 포함해 전체를 새로 학습시키는 것이다.

한 가지 복잡성은, Caltech 데이터셋 일부 이미지가 ImageNet 학습 데이터에도 포함되어 있다는 점이다. 우리는 정규화된 상관(normalized correlation)을 사용해 이러한 중복 이미지를 확인하고, ImageNet 학습셋에서 제거한 후 모델을 재학습시켜 train/test contamination 문제를 피했다.

**Caltech-101:** (Fei-fei et al., 2006)의 절차를 따라 클래스당 15장 또는 30장을 무작위로 학습에 사용하고, 최대 50장을 테스트에 사용했다. 5개의 학습/테스트 fold에서 클래스별 정확도의 평균을 보고하였다. 클래스당 30장의 경우 학습 시간은 17분이 소요되었다. 사전학습 모델은 (Bo et al., 2013)이 보고한 최고 성능을 2.2% 능가했으며 (86.5% vs 81.4%), 처음부터 학습한 ConvNet은 매우 낮은 성능(46.5%)을 보였다.

**Caltech-256:** (Griffin et al., 2006)의 절차를 따라 클래스당 15, 30, 45, 60장을 학습에 사용했으며, 클래스별 정확도의 평균을 Table 5에 보고하였다. 우리의 ImageNet 사전학습 모델은 기존 최고 성능(Bo et al., 2013)의 55.2%를 크게 초과하여 클래스당 60장일 때 74.2%를 달성했다. 그러나 Caltech-101과 마찬가지로 처음부터 학습한 모델은 성능이 매우 낮았다. Fig. 9에서는 “one-shot learning” (Fei-fei et al., 2006) 시나리오를 탐구했는데, 사전학습 모델은 클래스당 단 6장의 학습 이미지로도 기존 방법보다 높은 성능을 기록했다. 이는 ImageNet feature extractor의 강력함을 보여준다.

**PASCAL 2012:** 표준 학습 및 검증 이미지를 사용하여 ImageNet 사전학습 ConvNet 위에 20-way softmax를 학습시켰다. 이 방식은 이상적이지 않은데, PASCAL 이미지는 다중 객체를 포함할 수 있지만 모델은 단일 독점적 예측만 수행하기 때문이다. Table 6은 테스트셋 결과를 보여준다. PASCAL과 ImageNet 이미지는 성격이 꽤 다른데, 전자는 전체 장면이고 후자는 단일 객체 중심이다. 이로 인해 평균 성능은 최고치(Yan et al., 2012)의 82.2%보다 3.2% 낮은 79.0%였다고 설명된다. 그러나 5개 클래스에서는 오히려 더 좋은 성능을 보였고, 때로는 큰 차이로 앞섰다.

1. 원문 (5.3 Feature Analysis)
    
    We explore how discriminative the features in each layer of our Imagenet-pretrained model are. We do this by varying the number of layers retained from the ImageNet model and place either a linear SVM or softmax classifier on top. Table 7 shows results on Caltech-101 and Caltech-256. For both datasets, a steady improvement can be seen as we ascend the model, with best results being obtained by using all layers. This supports the premise that as the feature hierarchies become deeper, they learn increasingly powerful features.
    
    **Table 7.** Analysis of the discriminative information contained in each layer of feature maps within our ImageNet-pretrained convnet. We train either a linear SVM or softmax on features from different layers (as indicated in brackets) from the convnet. Higher layers generally produce more discriminative features.
    
    ```
    Cal-101 (30/class)   Cal-256 (60/class)
    
    SVM (1)   44.8 ± 0.7    24.6 ± 0.4
    SVM (2)   66.2 ± 0.5    39.6 ± 0.3
    SVM (3)   72.3 ± 0.4    46.0 ± 0.3
    SVM (4)   76.6 ± 0.4    51.3 ± 0.1
    SVM (5)   86.2 ± 0.8    65.6 ± 0.3
    SVM (7)   85.5 ± 0.4    71.7 ± 0.2
    
    Softmax (5)   82.9 ± 0.4    65.7 ± 0.5
    Softmax (7)   85.4 ± 0.4    72.6 ± 0.1
    
    ```
    

---

1. 번역문 (5.3 특징 분석)
    
    우리는 ImageNet 사전학습 모델의 각 계층(feature map)이 얼마나 판별적(discriminative) 정보를 담고 있는지를 탐구한다. 이를 위해 ImageNet 모델에서 유지하는 계층 수를 달리하고, 그 위에 선형 SVM 혹은 softmax 분류기를 얹어 성능을 측정하였다. Table 7은 Caltech-101과 Caltech-256에 대한 결과를 보여준다.
    
    두 데이터셋 모두, 계층이 깊어질수록 성능이 점진적으로 향상되며, 모든 계층을 사용했을 때 최고 성능을 보였다. 이는 **특징 계층(feature hierarchy)이 깊어질수록 점차 강력한 특징을 학습한다**는 전제를 뒷받침한다.
    
    **Table 7.** ImageNet 사전학습 ConvNet의 각 계층 feature map이 담고 있는 판별 정보 분석. ConvNet의 서로 다른 계층(괄호 안)에서 추출한 feature에 대해 선형 SVM 또는 softmax 분류기를 학습하였다. 일반적으로 상위 계층이 더 판별적인 특징을 생성한다.
    
    ```
    Cal-101 (30/class)   Cal-256 (60/class)
    
    SVM (1)   44.8 ± 0.7    24.6 ± 0.4
    SVM (2)   66.2 ± 0.5    39.6 ± 0.3
    SVM (3)   72.3 ± 0.4    46.0 ± 0.3
    SVM (4)   76.6 ± 0.4    51.3 ± 0.1
    SVM (5)   86.2 ± 0.8    65.6 ± 0.3
    SVM (7)   85.5 ± 0.4    71.7 ± 0.2
    
    Softmax (5)   82.9 ± 0.4    65.7 ± 0.5
    Softmax (7)   85.4 ± 0.4    72.6 ± 0.1
    
    ```
    

<aside>

## 5.1. ImageNet 2012

✅ **5줄 요약**

- ImageNet 2012: 학습 130만 / 검증 5만 / 테스트 10만, 1000 클래스
- AlexNet 구조 재현 → 보고된 성능과 동일
- stride 4→2, filter 11×11→7×7 → 성능 향상
- 단일 모델: Top-5 error 1.7% 개선
- 앙상블: 14.8% error → 당시 최고 성능, 비-ConvNet의 절반 수준

---

📌 **정리표 (5.1 ImageNet 2012)**

| 모델                      | Top-5 Error (%) | 비고                       |
| ------------------------- | --------------- | -------------------------- |
| AlexNet (2012)            | 16.4            | Krizhevsky et al.          |
| Zeiler & Fergus (단일)    | 약 14.7–15.0    | stride=2, filter=7×7 적용  |
| Zeiler & Fergus (앙상블)  | 14.8            | 2012 학습셋 기준 최고 성능 |
| 비-ConvNet (Gunji et al.) | 26.2            | 같은 대회 상위 entry       |

![image.png]((ZFNet)%20Visualizing%20and%20Understanding%20Convolutiona%2025ca9b246de18068af2bc450e4cf0526/image%207.png)

## 5.2 Feature Generalization

### ✅ **5줄 요약**

- ImageNet 사전학습 feature는 소규모 데이터셋에서도 강력
- Caltech-101: 86.5%, 기존 최고치보다 +2.2%
- Caltech-256: 74.2%, 기존 최고치보다 +19%
- PASCAL: 평균 79.0%, 최고치 82.2%에 근접, 일부 클래스는 우위
- ConvNet은 범용적 전이 학습 도구임을 입증

---

### 📌 **정리표 (5.2 Feature Generalization)**

| 데이터셋        | 사전학습 모델 성능    | 기존 최고치 | Scratch 학습 | 의미                                  |
| --------------- | --------------------- | ----------- | ------------ | ------------------------------------- |
| Caltech-101     | 86.5%                 | 81.4%       | 46.5%        | 소규모에서도 SOTA                     |
| Caltech-256     | 74.2% (60 imgs/class) | 55.2%       | 38.8%        | 대규모/소규모 모두 압도               |
| PASCAL VOC 2012 | 79.0% (mean)          | 82.2%       | -            | 다중 객체 장면, 일부 클래스는 더 우위 |

## 5.3 Feature Analysis

- **깊어질수록 판별 성능 증가**

### ✅ **5줄 요약**

- ConvNet의 feature 판별력은 깊을수록 향상
- Layer 1: 저성능 (엣지·색 기반)
- Layer 5: 큰 향상 (중간 특징이 매우 강력)
- Layer 7: Caltech-256에서는 최고, Caltech-101에서는 plateau
- ConvNet은 계층적으로 점점 강력한 특징 표현을 학습

---

### 📌 **정리**

| 데이터셋    | Layer 1 | Layer 3 | Layer 5 | Layer 7 | 의미                                        |
| ----------- | ------- | ------- | ------- | ------- | ------------------------------------------- |
| Caltech-101 | 44.8%   | 72.3%   | 86.2%   | 85.5%   | 중간~상위층에서 큰 향상, 최상위층은 plateau |
| Caltech-256 | 24.6%   | 46.0%   | 65.6%   | 71.7%   | 층이 깊을수록 계속 향상, 최상위층이 최강    |
</aside>

---