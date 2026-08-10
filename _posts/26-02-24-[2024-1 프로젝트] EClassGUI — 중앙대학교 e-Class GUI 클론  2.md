---
title: "[2024-1 프로젝트] EClassGUI — 중앙대학교 e-Class GUI 클론 (Java Swing)"
date: 2026-02-24 22:00:20 +0900
categories: [Computer-Science, etc.]
tags: [프로젝트, 2024-1, java]
---

# EClassGUI — 중앙대학교 e-Class GUI 클론 (Java Swing)

---

## 프로젝트 기본

- **Project Title:** EClassGUI — Java Swing 기반 중앙대학교 e-Class 클론
- **One-line summary:** 중앙대 e-Class의 "과제 제출" 흐름을 Java Swing으로 재구현하고, 기존 UI의 불편점(과제 세부 정보 접근성)을 개선한 GUI 데스크톱 애플리케이션
- **Project Type:** System / Pipeline (GUI Application)
- **My Role / Key Contribution:**
  - Java Swing (`JFrame`, `CardLayout`, `GridBagLayout`, `BoxLayout`)을 활용해 로그인 → 대시보드 → 수업 → 과제 및 평가 → 최종과제 제출의 전체 네비게이션 흐름 설계 및 단독 구현
  - 기존 e-Class의 UX 문제점 식별: 과제 세부 정보 확인 시 불필요한 단계 발생 → "과제 및 평가" 패널에 과제 요약을 직접 노출하여 개선
  - `summaryText`를 `public` 멤버로 설계하여 복수 패널에 동일 내용이 일관되게 표시되는 확장 가능한 구조 채택
  - IntelliJ Profiler를 활용한 성능 측정(실행 60초 기준 최대 CPU 사용률 36%) 및 예외 처리 구현
  - **최종 결과: 수업 내 1등**

---

## TL;DR

- **Problem:** 중앙대 e-Class에서 과제를 제출할 때, 과제 세부 내용(마감일·배점 등)을 확인하려면 과제 버튼을 별도로 클릭해야 하는 불편함이 존재한다.
- **Approach:** Java Swing의 `CardLayout`으로 5개의 화면을 전환하며, 과제 요약 정보를 별도 진입 없이 "과제 및 평가" 화면에 직접 노출. `JFileChooser`로 PDF 파일 업로드 기능 구현.
- **Main Result:** e-Class의 로그인 → 과제 제출 전체 흐름을 데스크톱 GUI로 구현하고 UX 개선 반영. 수업 내 1등 수상.
- **Keywords:** Java, Swing, JFrame, CardLayout, GridBagLayout, e-Class 클론, GUI, 과제제출

---

## Motivation & Background

- **Background:** 중앙대학교 e-Class는 Moodle 기반의 LMS(학습 관리 시스템)로, 수강 신청된 과목의 과제 제출·성적 확인·강의 자료 열람 등의 기능을 제공한다. 본 프로젝트는 2024년 1학기 소프트웨어프로젝트 교과목의 기말 프로젝트 과제로 수행되었다.
- **Why this problem matters:** 학생들이 매 학기 반복적으로 사용하는 LMS에서 UX 마찰이 작더라도 누적 시 상당한 불편으로 이어진다. "과제 세부 정보 확인 → 뒤로가기 → 과제 제출" 이라는 불필요한 단계를 줄이는 것이 실질적인 UX 개선 포인트다.
- **Gap in existing work:** 실제 e-Class는 과제 목록 페이지 (`과제 및 평가`)에서 마감일·제출 가능 기간만 표시하고, 과제 상세 내용은 별도 클릭 후 진입해야 확인 가능하다. 이를 한 화면에서 요약 노출하도록 개선하였다.
- **Related work:**
  - 홈페이지 규격 참고: https://emmakwon.kr/ideal-screen-size-for-desktop/ (1280×720 해상도 채택 근거)
  - GridBagLayout 활용: https://blog.naver.com/heoguni/130169571116, https://m.blog.naver.com/cdh1324/20046451663
  - Swing 컴포넌트 사용법: https://m.blog.naver.com/hjyang0/153916843
  - setPreferredSize 활용: https://cadaworld.tistory.com/24

---

## Approach

**(System/Pipeline 블록 채움, 나머지 해당 없음)**

- **System architecture (컴포넌트):**

  | 클래스                                | 역할                                                  |
  | ------------------------------------- | ----------------------------------------------------- |
  | `EClassGUI extends JFrame`            | 최상위 프레임. `CardLayout`을 통해 5개 패널 전환 관리 |
  | `BasicBackgroundPanel extends JPanel` | 로그인 화면 배경 이미지 렌더링용 커스텀 패널          |

  | 패널 (CardLayout 키)                           | 화면                                                     |
  | ---------------------------------------------- | -------------------------------------------------------- |
  | `loginPanel` ("로그인")                        | 아이디/비밀번호 입력 + 로그인 버튼                       |
  | `dashboardPanel` ("대시 보드")                 | 5개 수강 과목 버튼(텍스트+이미지), 할일·최근 피드백 보드 |
  | `softwareProjectPanel` ("소프트웨어 프로젝트") | 수업 홈 (11개 좌측 네비게이션 버튼 + 5개 중앙 정보 보드) |
  | `assignmentEvaluationPanel` ("과제 및 평가")   | 과제 요약 직접 표시 + 최종과제 제출 진입 버튼            |
  | `finalsubmissionPanel` ("최종과제 제출")       | `JFileChooser` (PDF 필터) 기반 파일 업로드               |

- **Data flow:**
  ```
  사용자 입력 (ID/PW) → Login() 검증
      ├── 성공 → cardLayout.show("대시 보드")
      └── 실패 → loginStatusLabel.setText("로그인 실패")

  대시보드 → [소프트웨어프로젝트 버튼/이미지 클릭] → cardLayout.show("소프트웨어 프로젝트")
  소프트웨어프로젝트 → [과제 및 평가 버튼] → cardLayout.show("과제 및 평가")
  과제 및 평가 → summaryText 표시 → [최종과제 제출 버튼] → cardLayout.show("최종과제 제출")
  최종과제 제출 → JFileChooser(PDF 필터) → 파일 선택/취소 처리
  ```
  `summaryText`는 `EClassGUI` 클래스의 `public` 인스턴스 필드로 선언되어 `assignmentboard`(과제 및 평가)와 `finalboard`(최종과제 제출) 두 패널에서 공유된다.

- **Control flow:**
  - 모든 화면 전환은 `CardLayout.show(getContentPane(), 패널명)` 호출로 처리
  - 버튼 이벤트는 익명 `ActionListener` 9개 + 내부 클래스 `OpneActionListener` 1개로 구현
  - "뒤로가기" 버튼은 각 패널에 배치되어 상위 화면으로 복귀하는 단방향 히스토리 구현

- **Deployment/Serving:**
  - IntelliJ IDEA: 프로젝트 열기 → `EClassGUI.java` → `main` 실행
  - CLI 단독 실행:
    ```bash
    # 프로젝트 루트(java_gui 2/)에서 실행
    javac EClassGUI.java
    java EClassGUI
    ```
  - 실행 시 `Image/` 폴더가 작업 디렉토리 기준으로 상대 경로 참조되므로, **반드시 `java_gui 2/` 디렉토리에서 실행**해야 이미지가 정상 로드됨

- **Monitoring/Logging:**
  - IntelliJ Profiler로 성능 측정 수행 (플레임 그래프, 메서드별 CPU 소요 시간 확인)
  - 60초 실행 기준 최대 CPU 사용률: **약 36%**

- **Scaling/Performance:**
  - 단일 사용자(ID="1", PW="1" 하드코딩) 기준 동작
  - `summaryText` public 설계 → 향후 관리자 계정 추가 시 한 곳에서 수정하면 모든 관련 패널에 자동 반영 가능

---

## Data & Experiment

- **Dataset type:** 해당 없음 (정적 GUI 애플리케이션)
- **Source:** 해당 없음
- **Size:** 해당 없음
- **Label/Target definition:** 해당 없음
- **Preprocessing:** 해당 없음
- **Leakage checks:** 해당 없음
- **Split (Train/Val/Test):** 해당 없음
- **Evaluation protocol:** IntelliJ Profiler를 사용한 수동 성능 측정 (약 60초 실행)
- **Metrics:** CPU 사용률(%), 예외 처리 동작 여부 (로그인 실패, 파일 미선택 시 에러 다이얼로그)
- **Environment:**
  - OS: macOS (개발 환경)
  - IDE: IntelliJ IDEA (Eclipse 사용 시 한글 입력 문제 발생으로 대체)
  - JDK: **21** (`project-jdk-name="21"`, `languageLevel="JDK_21"`)
- **Frameworks/Libraries:** Java 표준 라이브러리 (`javax.swing`, `java.awt`) — 외부 의존성 없음
- **Reproducibility:** 별도 설정 불필요. 소스 1개 파일 + `Image/` 폴더만 있으면 동일 결과 재현 가능

---

## Results

| 구분      | 항목                        | 값                                         |
| --------- | --------------------------- | ------------------------------------------ |
| 성능      | 최대 CPU 사용률 (60초 기준) | 약 36%                                     |
| 예외 처리 | 로그인 실패                 | "로그인 실패" 텍스트 표시 + 필드 초기화    |
| 예외 처리 | 파일 미선택 시 과제 제출    | "파일 선택 되지 않음" 에러 다이얼로그 표시 |
| 평가      | 수업 내 순위                | **1등**                                    |

- **Baseline method:** 기존 e-Class 웹 UI (과제 세부 정보 확인 시 별도 페이지 진입 필요)
- **This Work:** "과제 및 평가" 패널에 과제 요약(`summaryText`) 직접 표시 → 진입 단계 1단계 감소
- **Additional results:** 플레임 그래프 및 메서드별 CPU 소요 시간은 보고서 7페이지 수록 (이미지 기반, 추가 필요: 수치 상세)
- **Statistical significance / confidence:** 해당 없음
- **Visualization notes:** 보고서 p4–p5에 로그인·대시보드·소프트웨어 프로젝트·과제 및 평가·최종과제 화면 캡처 수록. p6–p7에 IntelliJ Profiler 플레임 그래프 및 CPU 사용량 그래프 수록.

---

## Discussion

- **Key observations:**
  1. `CardLayout` 단일 프레임 구조는 화면 전환 로직을 단순하게 유지시켜 주지만, 화면 수가 늘어날수록 `EClassGUI` 클래스의 책임이 과도하게 집중된다.
  2. `summaryText`를 인스턴스 필드로 공유함으로써 "단일 수정 → 전체 반영"이 가능한 설계를 자연스럽게 달성하였다.
  3. `GridBagLayout`은 컴포넌트 배치의 유연성은 높으나, 버튼 크기·여백 조절에 많은 제약 조건 코드가 필요해 가독성을 저해한다.
  4. 이미지 경로가 상대 경로(`"Image/..."`)로 하드코딩되어 있어, 실행 디렉토리가 달라지면 이미지 로드가 실패한다.
  5. 인증 정보(ID="1", PW="1")가 소스 코드에 `final` 필드로 직접 선언되어 있어, 확장 시 보안 문제가 발생할 수 있다.
- **Interpretation:** 단일 파일(490줄)로 로그인부터 파일 업로드까지의 전체 GUI 흐름을 구현한 점은 초기 구현에서는 작동하지만, 기능 추가 시 유지보수 비용이 크게 증가한다.
- **Trade-offs:** 단일 파일 구조 → 빠른 초기 구현 가능 vs. 클래스·메서드 수 증가 시 패턴화(MVC 등) 필요
- **Failure cases / surprising results:** `JFileChooser`에서 파일을 선택하지 않고 나갈 경우 에러 다이얼로그가 정상 표시됨. `pack()` 호출이 파일 선택 후 실행되나 실질적 효과 미미.
- **What I learned:**
  1. Java Swing의 `CardLayout`을 활용한 멀티 화면 전환 설계 방식
  2. `GridBagLayout`과 `BoxLayout`의 용도 차이(그리드형 vs. 수직 스택형 배치)
  3. UI 개발 시 사용자 불편 사항을 먼저 식별하고 구조적으로 반영하는 설계 접근법

---

## Limitations & Future Work

- **Limitations:**
  1. 인증 정보가 소스 코드에 하드코딩 (`Id="1"`, `Password="1"`) — 다중 사용자 미지원
  2. `GridBagLayout` 사용으로 버튼 크기·위치 조정이 복잡하여 UI 수정 비용이 높음
  3. "소프트웨어 프로젝트" 과목만 클릭 가능하며, 나머지 4개 과목 버튼은 동작하지 않음
  4. 이미지 경로가 실행 디렉토리 기준 상대 경로로 하드코딩되어 이식성 낮음
  5. 모든 로직이 단일 파일에 존재하여 기능 확장 시 유지보수 어려움

- **Future directions:**
  1. MVC 패턴 도입: 화면(`View`)과 비즈니스 로직(`Controller`/`Model`) 분리
  2. 다중 계정 지원: 학생/교수 역할 분리, 관리자 페이지 구현 (summaryText 원격 수정)
  3. 나머지 4개 과목 페이지 구현
  4. 이미지 경로를 클래스패스 기반(`getClass().getResource(...)`)으로 변경하여 이식성 개선
  5. 데이터 영속성: 과제 제출 내역·성적 등을 파일/DB에 저장하는 기능 추가

- **If I had more time:**
  - 실제 e-Class처럼 다양한 수업 페이지를 사용자 로그인 계정에 따라 동적으로 생성
  - 과제 제출 후 업로드 성공/실패 피드백 UI 개선 (단순 다이얼로그 → 상태 표시줄)
  - 웹 기반으로 포팅하여 브라우저에서 동작 가능하도록 전환

---

## Project Structure

```
java_gui 2/
├── EClassGUI.java                  # 전체 소스 (단일 파일, 490줄)
│                                   #   - EClassGUI extends JFrame  (메인 클래스)
│                                   #   - BasicBackgroundPanel extends JPanel (배경 이미지 패널)
├── Image/
│   ├── 로그인 페이지 - 중앙대.jpg  # 로그인 화면 배경 이미지
│   ├── 기초컴퓨터프로그래밍.png
│   ├── 소프트웨어프로젝트.png
│   ├── 회귀분석.png
│   ├── AI딥.png
│   ├── IT개론.png
│   └── rowImage/                   # 원본 크기 과목 썸네일 이미지 (동일 5종)
├── .idea/                          # IntelliJ IDEA 프로젝트 설정
│   ├── java_gui.iml                # 모듈 정의 (JDK: 상속)
│   └── misc.xml                    # JDK 21, languageLevel=JDK_21
└── out/production/java_gui/        # 컴파일 산출물 (.class 파일)
    ├── EClassGUI.class
    ├── EClassGUI$1.class ~ $9.class  # 익명 ActionListener 클래스
    └── BasicBackgroundPanel.class
```

### 실행 방법

**IntelliJ IDEA:**
1. `java_gui 2/` 폴더를 IntelliJ에서 프로젝트로 열기
2. `EClassGUI.java` → `main` 메서드 실행

**CLI (JDK 21 설치 필요):**
```bash
cd "java_gui 2"
javac EClassGUI.java
java EClassGUI
```
> 이미지 파일(`Image/` 폴더)이 실행 디렉토리 기준으로 참조되므로, 반드시 `java_gui 2/` 내에서 실행할 것.

**로그인 정보:** ID: `1` / PW: `1`  
**탐색 팁:** 파란색 버튼을 따라가면 구현된 핵심 기능 확인 가능

---

## PDF/Slides Mapping

- **Main slide deck(s):** `20204885 강태영 소프트웨어프로젝트 최종보고서3.pdf` (7페이지, 2024년 1학기)

- **Slide-to-README mapping:**

  | 섹션                       | 보고서 페이지         | 내용                                                 |
  | -------------------------- | --------------------- | ---------------------------------------------------- |
  | Problem statement          | p1 (목적 및 내용 A~D) | e-Class "과제 제출" 기능 구현 목적, 불편점 개선 동기 |
  | Method/Architecture        | p2 (주요 개발 내용 F) | EClassGUI 클래스 구조, 패널 목록                     |
  | Method detail              | p3 (메서드 목록)      | 7개 주요 메서드 역할 기술                            |
  | Experiment setup / Results | p4–p5                 | 화면 캡처: 로그인·대시보드·과제 및 평가·최종과제     |
  | Feature detail             | p6 (주요 기능 A~E)    | 각 화면별 동작 상세 기술                             |
  | Performance analysis       | p7 (성능 평가 A~D)    | 플레임 그래프, CPU 사용량(36%), 예외 처리            |
  | Conclusion/Future work     | p7 (확장성과 한계점)  | summaryText 공유 설계의 확장성, GridBagLayout 한계   |

- **Numbers provenance:**
  - CPU 최대 사용률 36% → 보고서 p7 "CPU사용량" 섹션

- **Any missing slides / gaps:**
  - 추가 필요: p4–p5, p7의 프로파일러 그래프·화면 캡처는 이미지 기반으로 수치 상세 추출 불가

---

## Citation & License

- **Citation info:**
  - 작성자: 강태영 (학번: 20204885)
  - 과목: 소프트웨어프로젝트 (2024년 1학기, 중앙대학교)
  - 제출: 기말 프로젝트 최종보고서 (`20204885 강태영 소프트웨어프로젝트 최종보고서3.pdf`)
- **License:** 추가 필요 (별도 라이선스 미지정)
- **Papers/links:**
  - https://emmakwon.kr/ideal-screen-size-for-desktop/
  - https://blog.naver.com/heoguni/130169571116
  - https://m.blog.naver.com/cdh1324/20046451663
  - https://m.blog.naver.com/hjyang0/153916843
  - https://cadaworld.tistory.com/24
