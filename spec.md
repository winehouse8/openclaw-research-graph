# spec.md

## 프로젝트명
**openclaw-research-graph**

## 목적
OpenClaw를 위한 **Neo4j 기반 컨티뉴얼 리서치 스킬**을 구축한다.

이 시스템은 OpenClaw가:
- 특정 주제에 대해 반복적으로 리서치를 수행하고
- 그 결과를 구조화된 지식 그래프로 축적하며
- 시간이 지나 새 정보가 들어오면 기존 답을 갱신하고
- 궁극적으로는 매일 아침 9시처럼 정해진 시간에 자동으로 리서치를 수행할 수 있도록 해야 한다.

이 프로젝트는 단발성 요약기가 아니다.  
장기적으로 축적되고 재사용되는 **리서치 메모리 및 retrieval 시스템**이다.

---

## 핵심 목표
OpenClaw가 다음과 같은 질문을 장기적으로 계속 조사할 수 있어야 한다.

예:
- 16GB 맥미니에서 가장 성능이 좋은 로컬 LLM은 무엇인가?
- OpenClaw의 메모리 기능 향상을 위해 가장 좋은 agentic memory 서비스는 무엇인가?

이 시스템은 다음을 지원해야 한다.
- 초기에는 외부 검색 기반의 cold-start 리서치
- 이후에는 source / thinking 노드가 축적된 그래프 메모리 활용
- 새로운 정보가 생기면 기존 답을 수정하고 업데이트하는 반복 리서치

---

## 필수 기반 요구사항
Neo4j 지식 그래프는 단순 저장소가 아니라,  
**효율적인 지식 그래프 관리 엔진**이어야 한다.

반드시 다음을 지원해야 한다.

### CRUQD
- Create
- Read
- Update
- Query
- Delete

### 중복 감지
- 정확히 같은 source / thinking 감지
- 근접 중복도 보수적으로 감지
- 애매하면 자동 병합하지 않기

### RAG 지원
- 이후 OpenClaw / actor / critic이 활용할 수 있는 retrieval 경로 제공
- source retrieval과 thinking retrieval은 분리
- 단순 final QA용이 아니라 반복 리서치용 RAG여야 함

즉 이 그래프의 핵심 목적은:
**효율적인 CRUQD + 중복 감지 + RAG를 통한 컨티뉴얼 리서치 지원**이다.

---

## 그래프 모델
그래프 구조는 단순하고 보수적으로 유지한다.

### 노드 타입
- `Topic`
- `ResearchObjective`
- `SourceSummary`
- `SourceChunk`
- `ThinkingSummary`
- `ThinkingChunk`

### 관계 타입
- `BELONGS_TO`
- `CHILD_OF`
- `REFERENCES`

### 스코프 규칙
- 모든 `ResearchObjective`는 하나의 `Topic`에 속한다
- 모든 `SourceSummary`는 하나의 `Topic`에 속한다
- 모든 `ThinkingSummary`는 하나의 `Topic`에 속한다
- `SourceSummary`, `ThinkingSummary`는 필요하면 `ResearchObjective`에도 추가로 속할 수 있다
- Chunk는 반드시 parent summary에만 연결된다
- Thinking은 source 및 다른 thinking을 reference할 수 있다
- retrieval 편의를 이유로 의미 없는 direct edge를 남발하지 않는다

---

## Retrieval 규칙
Retrieval은 반드시 **스코프 기반 + 보수적**이어야 한다.

### Source retrieval
- 후보 풀은 반드시 **현재 Topic 하위 source**에서만 만든다
- 그 안에서만 RAG를 수행한다
- 그 뒤 relevance 판단으로 더 줄인다

### Thinking retrieval
- 후보 풀은 반드시 **현재 ResearchObjective 하위 thinking**에서만 만든다
- 그 안에서만 RAG를 수행한다
- 그 뒤 relevance 판단으로 더 줄인다

### Retrieval 흐름
Retrieval의 기본 흐름은 다음과 같다.

**필터링 → RAG → 관련도 판단**

관련 후보 풀이 존재할 때,  
다른 topic/objective까지 넓게 섞어 noisy retrieval을 하지 않는다.

---

## Cold Start 원칙
초기에는 그래프 내부에 사용할 만한 메모리가 없을 수 있다.

이 경우:
- 내부 메모리가 있는 척하지 않는다
- noisy한 graph retrieval을 억지로 하지 않는다
- 외부 검색을 우선한다
- 리서치 과정에서 source / thinking을 새로 축적한다
- 이후 라운드에서 memory-augmented retrieval로 넘어간다

즉 원칙은 단순하다.

- 내부 메모리가 있으면 재사용
- 내부 메모리가 없으면 외부 검색으로 시작
- 새 리서치를 저장한 뒤 다음 루프에서 재사용

---

## Source와 Thinking의 분리
이 프로젝트는 다음 둘을 절대로 섞으면 안 된다.

### Source
- 외부 근거
- 사실
- 문서
- 기사
- 트랜스크립트
- 벤치마크 결과
- 외부에서 나온 주장

### Thinking
- 우리의 해석
- 종합
- 판단
- 가설
- 결론
- 계획
- 반론

이 둘은 하나의 뭉뚱그린 메모리 저장소가 되면 안 된다.

---

## 리서치 루프
시스템은 반복적 리서치 루프를 지원해야 한다.

정상적인 루프는 대략 다음과 같다.

1. 현재 topic / objective 상태를 읽는다
2. cold-start인지, memory-augmented mode인지 판단한다
3. 필요하면 외부 evidence를 수집한다
4. source를 정규화하고 저장한다
5. thinking을 생성/수정한다
6. 다음 단계에서 scoped memory retrieval을 수행한다
7. 현재 답을 갱신한다
8. 필요하면 다시 반복한다

actor/critic 형태의 reasoning은 사용할 수 있지만,  
retrieval layer는 reasoning layer와 분리되어야 한다.

---

## 매일 리서치 목표
이 시스템은 정기 반복 리서치를 지원해야 한다.

목표 동작:
- OpenClaw가 매일 오전 9시에 특정 research objective를 실행
- 새 웹 정보를 수집
- 기존 graph memory를 재사용
- 기존 결론을 업데이트
- 반복 실행할수록 그래프 품질이 올라감

즉 daily research는 이 프로젝트의 중요한 최종 사용 시나리오다.

---

## OpenClaw 통합 목표
최종적으로 이 시스템은 OpenClaw가 직접 사용할 수 있는 research skill이어야 한다.

OpenClaw는 이 시스템으로:
- research objective를 생성하거나 선택하고
- retrieval 함수를 호출하고
- 리서치 루프를 수행하고
- source / thinking 업데이트를 저장하고
- 이후 reasoning에서 그래프를 재활용할 수 있어야 한다

그래프 계층과 reasoning/orchestration 계층은 개념적으로 분리되어야 한다.

---

## 제약
- 설계는 단순하게 유지할 것
- 과설계 금지
- 데이터 처리는 보수적으로 할 것
- 애매한 objective/source 자동 병합 금지
- 똑똑해 보이는 구조보다 명확한 구조를 우선할 것
- retrieval purity를 중요하게 볼 것
- `spec.md`는 프로젝트의 북극성이며 함부로 수정하지 않는다

---

## 성공 기준
다음을 만족하면 이 프로젝트는 성공이다.

- OpenClaw가 특정 주제를 장기적으로 반복 리서치할 수 있다
- Neo4j가 결과를 구조화된 그래프로 저장한다
- 그래프가 효율적인 CRUQD를 지원한다
- 중복 및 근접 중복 처리 경로가 존재한다
- scoped RAG retrieval이 동작한다
- cold-start와 memory-augmented mode가 모두 동작한다
- 새로운 외부 정보가 들어오면 기존 결론을 갱신할 수 있다
- 매일 오전 9시 자동 리서치로 확장 가능한 구조가 마련된다
