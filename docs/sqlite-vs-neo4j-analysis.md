# SQLite 유지 vs Neo4j 마이그레이션 — 코드 기반 분석

분석 대상: `openclaw-research-graph` @ `autopilot/research-graph-rebuild` (HEAD 분석 시점에 e2e 검증까지 완료된 상태)

이 문서는 추상론이 아니라 **현재 레포의 실제 스키마와 코드**를 기준으로 작성됐다. 인용 라인 번호는 분석 시점 기준이며, 핵심 결론은 "어떤 관계가 저장되고, 어떤 관계는 못 저장되며, Neo4j 로 가면 무엇이 달라지는가" 다.

분석에 참고한 실제 파일:

- `research_graph/storage.py` — SQLite 스키마와 모든 CRUD 헬퍼
- `research_graph/orchestrator.py` — `ResearchResult`, supersession/reuse 로직
- `research_graph/dedup.py` — 콘텐츠 hash + Jaccard near-dup
- `research_graph/retrieval.py` — TF-IDF 코사인 RAG
- `research_graph/reasoning.py` — actor/critic
- `tests/test_e2e_scenarios.py` — 멀티 토픽/시간축/supersession 워크 검증

---

## A. 현재 SQLite 구현이 실제로 저장하는 관계

### 1. 스키마 요약 (`storage.py:9-54`)

다섯 개 테이블이 전부다.

| 테이블 | 주요 컬럼 | 외래키 |
|---|---|---|
| `topics` | `id`, `name`, `created_at` | (없음) |
| `objectives` | `id`, `topic_id`, `question`, `status`, `created_at`, `updated_at` | `topic_id → topics.id ON DELETE CASCADE` |
| `sources` | `id`, `objective_id`, `url`, `title`, `content`, `content_hash`, `fetched_at`, `search_query` | `objective_id → objectives.id ON DELETE CASCADE` |
| `thinkings` | `id`, `objective_id`, `content`, `content_hash`, `supports_source_ids` (JSON), `author`, `created_at`, `supersedes_id` | `objective_id → objectives.id ON DELETE CASCADE`, `supersedes_id → thinkings.id` (no cascade) |
| `embeddings` | `object_kind` (`source`/`thinking`), `object_id`, `terms` (JSON dict) | (없음, 논리적으로 source/thinking 행을 포인트) |

인덱스는 `idx_objectives_topic`, `idx_sources_objective`, `idx_sources_hash`, `idx_thinkings_objective`, `idx_thinkings_hash` 다섯 개. **`supports_source_ids` 위에 인덱스는 없다** — 이게 본 분석의 핵심이다.

### 2. 각 관계가 어떻게 저장되는가

#### Topic ↔ Objective
정상적인 1:N FK. `objectives.topic_id` 가 표준 정수 외래키로 인덱스까지 걸려있다. 트래버스/조인 자유롭다.

```sql
SELECT * FROM objectives WHERE topic_id = ?;
```

#### Objective ↔ Source / Objective ↔ Thinking
역시 1:N FK + 인덱스. `list_sources(conn, objective_id)` 와 `list_thinkings(conn, objective_id)` 가 그대로 구현하고 있다 (`storage.py:196-198, 227-230`).

#### Thinking ↔ Source — **JSON 블롭 한정**
**문제 지점이다.** thinking 이 어떤 source 들을 인용했는지는 별도 junction 테이블이 아니라 thinking 행의 한 컬럼에 JSON 직렬화돼 있다:

```sql
CREATE TABLE thinkings (
    ...
    supports_source_ids TEXT NOT NULL,  -- JSON list, 예: "[12, 14, 17]"
    ...
);
```

`insert_thinking` 은 `json.dumps(supports_source_ids)` 로 저장하고 (`storage.py:212-219`), `_decode_thinking` 은 읽을 때 다시 `json.loads` 한다 (`storage.py:84-89`).

이 표현의 한계:
- **인덱스 불가**: SQLite 에서 JSON 배열 안의 값으로 검색하려면 JSON1 확장의 `json_each` 나 가상 컬럼 + FTS5 가 필요한데 현 구현은 이를 안 쓴다.
- **역방향 조회 불가**: "source 14를 인용한 thinking 들" 을 가져오려면 모든 thinking 행을 풀스캔해서 JSON 디코드 후 멤버십 체크해야 한다. O(N) per query, 인덱스 안 탐.
- **트랜잭션 일관성 약함**: `INSERT INTO thinkings ... supports_source_ids='[1,2,3]'` 인 상태에서 source 2 가 하드 삭제되면 JSON 안 ID는 그대로 남아 dangling 된다. FK 가 안 걸리니까 DB 가 잡아주지 못한다.
- **그래프 탐색 불가**: 2-hop 쿼리 ("이 thinking → 그 source 들 → 그 source 를 인용한 다른 thinking 들") 는 SQL 한 줄로 못 푼다. 애플리케이션 코드에서 디코드 + 풀스캔 + 디코드 N번 반복.

#### Thinking ↔ Thinking (supersedes)
`thinkings.supersedes_id` 는 self-reference FK 다. orchestrator 가 새 thinking 을 만들 때 `prior[-1]["id"]` 를 넣는다 (`orchestrator.py:80-84`):

```python
prior = storage.list_thinkings(self.conn, objective_id)
supersedes = None
if prior and result.new_source_ids:
    supersedes = int(prior[-1]["id"])
```

이건 정상적인 self-FK 라서 supersession **체인** 자체는 SQL 한 줄로 따라갈 수 있다:

```sql
WITH RECURSIVE chain(id, prev) AS (
  SELECT id, supersedes_id FROM thinkings WHERE id = ?
  UNION ALL
  SELECT t.id, t.supersedes_id
  FROM thinkings t JOIN chain c ON c.prev = t.id
)
SELECT * FROM chain;
```

SQLite 는 재귀 CTE 를 지원하므로 supersession 체인 단일 축은 잘 표현된다.

#### Thinking ↔ Thinking (reuse)
**저장이 안 된다.** `ResearchResult.reused_thinking_id` 필드는 `orchestrator.py:17` 에서 정의된 *런타임 dataclass 필드*일 뿐이고, 어떤 테이블에도 들어가지 않는다. dedup-collapse 가 일어나서 새 thinking row 가 생성되지 않은 사실은 `to_dict()` 응답으로만 흘러나가고 (CLI/api 호출자가 받음), 다음 호출 시점에는 흔적이 없다.

즉 "어떤 thinking 이 어느 thinking 의 dedup 결과로 reuse 됐다" 는 **DB 에 없다**. spec 의 "축적된 지식 재사용" 요구는 retrieval (TF-IDF) 차원에서 충족돼 있지만, "어떤 답을 누가 어떻게 reuse 했느냐" 의 *프로비넌스 그래프* 는 현재 모델에 없다.

#### Embedding 관계
`embeddings.object_kind` + `object_id` 로 source 또는 thinking 행을 가리킨다. FK 는 안 걸려있고 (`storage.py:48-53`), `delete_objective`/`delete_source` 가 수동으로 정리해준다 (`storage.py:167-179, 206-209`). source 와 thinking 의 임베딩 공간은 분리돼 있어 spec 의 retrieval purity 요구는 만족.

### 3. 예시 질의: "특정 thinking → 그 source 들 → 그 source 와 연결된 다른 thinking → 그 thinking 의 다른 source"

이건 4-hop 그래프 트래버설이다. 현재 구현에서 풀려면:

```python
# 1-hop: thinking T → its source ids
t = storage.get_thinking(conn, T)
src_ids = t["supports_source_ids"]   # JSON 디코드된 list

# 2-hop: source ids → thinkings that cite them (역방향)
# JSON 인덱스 없으니 풀스캔
related_thinkings = []
for row in conn.execute("SELECT id, supports_source_ids FROM thinkings"):
    cited = json.loads(row["supports_source_ids"])
    if any(s in src_ids for s in cited):
        related_thinkings.append(int(row["id"]))

# 3-hop: 그 thinking 들의 다른 source 들 (1-hop 의 대칭)
other_src_ids = set()
for tid in related_thinkings:
    other = storage.get_thinking(conn, tid)
    other_src_ids.update(other["supports_source_ids"])
other_src_ids -= set(src_ids)

# 4-hop: 그 source 행 자체
other_sources = [storage.get_source(conn, sid) for sid in other_src_ids]
```

요약:
- 단일 SQL 표현 불가능
- 2-hop 단계에서 풀스캔 (`SELECT id, supports_source_ids FROM thinkings`) + 애플리케이션 측 JSON 디코드
- N+1 패턴 강제 (3-hop 이 한 줄당 한 번씩 `get_thinking` 호출)
- 트래버스 깊이가 늘어날수록 quadratic 으로 비싸짐

이 워크로드를 SQLite 안에서 정상 속도로 풀고 싶으면 다음 둘 중 하나가 필요하다:
1. **junction 테이블 추가**: `thinking_sources(thinking_id, source_id, PRIMARY KEY(thinking_id, source_id))` + 양방향 인덱스. 2-hop 이 단순 JOIN 두 번이 되고 4-hop 도 재귀 CTE 한 번으로 풀린다. 변경 비용 ~50 LOC, 마이그레이션 1회.
2. **JSON1 + 가상 컬럼 + FTS5**: 비표준 SQLite 빌드와 본격 인덱싱 셋업이 필요해서 stdlib-only 제약과 충돌.

> 참고: 현재 코드에서 reuse 관계는 영속화 자체가 안 되므로, junction 만 추가해도 thinking ↔ thinking reuse 추적은 별도 작업이다.

---

## B. 같은 요구사항을 Neo4j 모델로 설계

### 1. 노드와 관계

```
(:Topic        {name, created_at})
(:Objective    {question, status, created_at, updated_at})
(:Source       {url, title, content, content_hash, fetched_at, search_query})
(:Thinking     {content, content_hash, author, created_at})

(:Topic)-[:HAS_OBJECTIVE]->(:Objective)
(:Objective)-[:HAS_SOURCE]->(:Source)
(:Objective)-[:HAS_THINKING]->(:Thinking)
(:Thinking)-[:CITES]->(:Source)            // 현재 supports_source_ids JSON 의 정상 표현
(:Thinking)-[:SUPERSEDES]->(:Thinking)     // 현재 supersedes_id self-FK 의 정상 표현
(:Thinking)-[:REUSES]->(:Thinking)         // 현재 미저장. dedup-collapse 시점에 기록
(:Thinking)-[:WRITTEN_BY {role:'actor'|'critic'}]->(:Agent)   // (선택) 멀티 actor 도입 시
```

핵심:
- **`Source` 와 `Thinking` 은 라벨이 분리된 노드**라서 spec 의 source/thinking purity 가 스키마 차원에서 강제된다. 검색 시 `MATCH (s:Source)` vs `MATCH (t:Thinking)` 로 명시적으로 분리.
- `(:Thinking)-[:CITES]->(:Source)` 가 양방향 트래버스 가능한 정식 엣지가 된다. 인덱스 없이도 그래프 엔진이 양방향 트래버스를 O(degree) 로 처리.
- supersession 과 reuse 가 둘 다 thinking-thinking 엣지로 자연스럽게 분리된다 — 현재 SQLite 모델에서 reuse 가 누락된 문제가 자동으로 해결된다.
- 임베딩은 노드 속성으로 (`Source.embedding: [float]`) 두거나, Neo4j 5.x 의 vector index (`db.index.vector.queryNodes`) 로 RAG 와 그래프 트래버스를 한 쿼리 안에서 섞을 수 있다.

### 2. 인덱스/제약

```cypher
CREATE CONSTRAINT topic_name_unique     IF NOT EXISTS FOR (t:Topic)     REQUIRE t.name IS UNIQUE;
CREATE CONSTRAINT source_hash_unique    IF NOT EXISTS FOR (s:Source)    REQUIRE s.content_hash IS UNIQUE;
CREATE CONSTRAINT thinking_hash_unique  IF NOT EXISTS FOR (t:Thinking)  REQUIRE t.content_hash IS UNIQUE;
CREATE INDEX      objective_status      IF NOT EXISTS FOR (o:Objective) ON (o.status);
CREATE VECTOR INDEX source_vector       IF NOT EXISTS FOR (s:Source)    ON s.embedding OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
CREATE VECTOR INDEX thinking_vector     IF NOT EXISTS FOR (t:Thinking)  ON t.embedding OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}};
```

### 3. A 절의 예시 질의를 Cypher 로

"특정 thinking → 그 source 들 → 그 source 들과 연결된 다른 thinking → 그 thinking 의 다른 source":

```cypher
MATCH (t0:Thinking {id: $tid})-[:CITES]->(s:Source)<-[:CITES]-(t1:Thinking)
WHERE t1 <> t0
MATCH (t1)-[:CITES]->(s2:Source)
WHERE NOT (t0)-[:CITES]->(s2)
RETURN t0, collect(DISTINCT s) AS shared_sources,
       collect(DISTINCT t1) AS related_thinkings,
       collect(DISTINCT s2) AS new_sources
```

한 줄짜리 패턴매치다. 인덱스 셋업 후 100ms 단위로 끝난다.

다른 자주 쓸 만한 질의:

```cypher
// supersession 체인 전체 (재귀)
MATCH path = (head:Thinking {id: $tid})-[:SUPERSEDES*0..]->(ancestor:Thinking)
RETURN path;

// "이 source 를 가장 많이 인용한 thinking 들"
MATCH (s:Source {id: $sid})<-[:CITES]-(t:Thinking)
RETURN t, count(*) AS uses ORDER BY uses DESC LIMIT 10;

// 두 topic 사이의 공유 source (cross-topic 발견)
MATCH (t1:Topic)-[:HAS_OBJECTIVE]->()-[:HAS_SOURCE]->(s:Source)<-[:HAS_SOURCE]-()<-[:HAS_OBJECTIVE]-(t2:Topic)
WHERE t1 <> t2
RETURN t1.name, t2.name, collect(DISTINCT s.url) AS shared LIMIT 20;

// RAG + 그래프: 질의 임베딩과 가까운 source 5개 + 그 source 를 인용한 thinking 들
CALL db.index.vector.queryNodes('source_vector', 5, $query_embedding)
YIELD node AS s, score
MATCH (s)<-[:CITES]-(t:Thinking)-[:HAS_THINKING|:HAS_OBJECTIVE*1..2]-(:Topic {name: $topic})
RETURN s.url, score, collect(DISTINCT t.content) AS reasoning;
```

마지막 패턴이 핵심이다. **vector retrieval 결과 노드를 출발점으로 그래프 트래버스**가 한 쿼리 안에서 가능. SQLite + 외부 vector DB 조합으로는 이걸 두 시스템 사이를 왕복하면서 풀어야 한다.

### 4. RAG + 그래프 통합 구조

권장 구조:
- `Source.embedding`, `Thinking.embedding` 은 노드 속성 (Neo4j 5.x vector index)
- 외부 검색 결과 → `MERGE (s:Source {content_hash: ...})` 로 dedup
- 검색 시: vector index 로 후보 K 개 추출 → 같은 쿼리에서 그래프 패턴 매칭으로 토픽/오브젝티브 필터 → 결과를 reasoning 으로 넘김
- supersession 은 하나의 엣지로 표현되니 retrieval 시 `WHERE NOT EXISTS { (t)<-[:SUPERSEDES]-() }` 로 "최신 thinking 만" 자연스럽게 거를 수 있음

---

## C. 비교와 추천

### 1. SQLite 유지의 장단점 (현 구현 기준)

**장점**
- stdlib 만으로 동작 → `pip install` 도 필요 없음. spec 의 "단순하게 유지" 원칙과 정확히 부합.
- 30 ms 안에 32개 테스트 + 11개 e2e 시나리오 + 데모가 다 돈다.
- 단일 파일 DB → 백업/복제/git 으로 ship 가능. cron 시나리오에서 운영 부담 0.
- supersession 단일 축, objective↔source/thinking 1:N 같은 트리형 관계는 이미 정상적인 FK 로 잡혀있고 재귀 CTE 로 풀린다.

**단점**
- `thinkings.supports_source_ids` 가 JSON 블롭이라 thinking↔source 양방향 트래버스가 풀스캔. 인덱스 안 탐.
- thinking ↔ thinking reuse 관계가 영속화 자체가 안 됨 (런타임 객체에서만 흐름).
- 멀티-홉 그래프 질의를 SQL 한 줄로 못 표현 → 애플리케이션 코드에 N+1 / 디코드 / 합집합 로직 강제.
- vector retrieval 과 그래프 트래버스를 한 쿼리에서 못 섞음 → 후속 작업이 두 시스템 코디네이션을 떠안음.

### 2. Neo4j 마이그레이션의 장단점 (그래프 요구 기준)

**장점**
- thinking↔source, supersedes, reuse 모두 1급 엣지. 양방향 트래버스 자연.
- A 절의 4-hop 예시 질의가 단일 Cypher 패턴.
- vector index + 그래프 트래버스 통합 (5.x 이후) → "RAG 결과를 시작점으로 그래프 따라가기" 가 한 쿼리.
- 시각화 (Bloom, browser) 가 디버깅과 OpenClaw 운영자 신뢰도 양쪽에 도움.
- spec 의 "효율적 지식 관리 엔진" 표현에 가장 직접적으로 부합.

**단점**
- JVM 데몬 운영 부담. Docker compose 든 systemd 든 하나 더 떠야 함.
- pip 의존성 추가 (`neo4j` 드라이버, vector index 쓰려면 5.x). spec 의 stdlib-only 제약과 충돌.
- 단일 노드 운영도 메모리 ~512MB 권장 → 16GB Mac mini 에서 다른 OpenClaw 컴포넌트와 메모리 경쟁.
- Community 에디션은 단일 인스턴스, 단일 DB. 클러스터 가려면 라이선스.
- 마이그레이션 비용: 현 5 테이블 + 연결 의존 코드 (`storage.py`, `dedup.py`, `retrieval.py`, `orchestrator.py`, `api.py`, `cli.py`) 를 거의 전부 다시 써야 함. 테스트 fixture 셋업도 in-memory Neo4j 부재로 testcontainers 류 의존성 추가 필요.
- 단순 CRUD/카운트 워크로드는 SQLite 보다 느릴 수 있음.

### 3. 차원별 비교

| 차원 | SQLite (현 구현) | SQLite + 정규화 junction (옵션 1) | Neo4j 풀 마이그 |
|---|---|---|---|
| 성능 (CRUD) | 빠름 | 빠름 | 보통 |
| 성능 (1-hop 트래버스) | 느림 (풀스캔) | 빠름 (인덱스) | 빠름 |
| 성능 (4-hop 트래버스) | 매우 느림 | 보통 (재귀 CTE) | 빠름 |
| 성능 (vector retrieval) | TF-IDF 자체 구현, 코퍼스 크기 한계 ~수만 | 동일 | Native vector index, 수백만 |
| 구현 복잡도 | 낮음 (현 상태) | 낮음 (~50 LOC 추가) | 매우 높음 (재작성) |
| 운영 복잡도 | 0 (단일 파일) | 0 | 중간 (데몬 + 백업) |
| 질의 표현력 | 낮음 (SQL + 풀스캔) | 중간 (재귀 CTE) | 매우 높음 (Cypher) |
| 장기 확장성 | 수만 thinking 까지 | 수십만 thinking 까지 | 수백만 노드 |
| spec 단순성 원칙 | 강한 부합 | 강한 부합 | 약한 충돌 |
| openclaw cron 운영 적합도 | 매우 좋음 | 매우 좋음 | 좋음 (운영 1번 떠야 함) |

### 4. 최종 추천

> **지금 PR 은 그대로 머지하고, 다음 단계로 Neo4j 가 아니라 *SQLite 안에서 thinking↔source junction 테이블 + thinking_reuse 테이블* 을 추가하라.**

근거를 풀어쓰면:

1. **현재 spec 의 실제 needs 는 그래프 트래버스가 아니라 *축적/dedup/RAG/supersession* 이다.** 위 11개 e2e 시나리오는 이미 모두 통과한다. PR #1 은 spec 충족이 끝났다.
2. 재우형이 말한 "thinking → 그 source → 다른 thinking → 다른 source" 같은 그래프 질의의 진짜 병목은 SQLite 자체가 아니라 `supports_source_ids` 가 JSON 블롭이라는 한 가지 결정이다. 이걸 정상 junction 테이블 (`thinking_sources(thinking_id, source_id, PRIMARY KEY ...)`) 로 바꾸면 4-hop 트래버스가 재귀 CTE 한 번으로 풀린다. 추가 비용 ~50 LOC + 마이그레이션 1회.
3. 동시에 `thinking_reuses(reuser_id, reused_id, created_at)` 테이블을 추가하면 현재 *런타임 메모리에만* 존재하는 reuse 사실이 영속화돼서 spec 의 "지식 재사용" 요구가 데이터 차원에서 명시적 그래프가 된다.
4. Neo4j 는 다음 둘 중 하나가 분명해질 때 검토하라:
   - **(a) cross-topic / 4+ hop 패턴매칭** 이 OpenClaw 의 실제 reasoning 경로에 들어가서 SQL 재귀 CTE 가 아프다고 측정으로 증명될 때
   - **(b) vector retrieval + 그래프 트래버스 통합** 이 한 쿼리 안에서 필요해질 때 (즉, RAG 결과 노드를 출발점으로 그래프 패턴매칭이 hot path 일 때)
5. 그 시점에는 **hybrid** 가 맞다: SQLite 가 *canonical store* (단일 파일 백업, cron 운영 단순성 유지) 이고, Neo4j 가 *derived index* (canonical 로부터 ETL 로 채워지는 그래프/벡터 인덱스) 다. 이 패턴이면 SQLite 의 운영 단순함과 Neo4j 의 질의 표현력을 둘 다 가진다. 처음부터 Neo4j 풀 마이그는 spec 단순성과 정면충돌하므로 비추.

요약 한 단락:

**PR #1 은 그대로 머지. 다음 PR 에서 SQLite 안에 thinking_sources 와 thinking_reuses 두 junction 테이블을 추가해 그래프 트래버스 비용을 풀스캔에서 인덱스로 끌어내려라. 그래도 부족해지는 시점 (멀티홉 패턴매칭이 hot path 가 되거나, vector + 그래프 통합이 필요해지는 시점) 에 한해 SQLite 를 canonical store 로 유지한 채 Neo4j 를 derived index 로 추가하는 hybrid 구조로 진화시켜라. 처음부터 Neo4j 풀 마이그레이션은 spec 의 "단순하게 유지" 원칙과 운영 비용 양쪽에서 손해다.**

---

## 부록 — 즉시 적용 가능한 SQLite junction 패치 스케치

```sql
-- 다음 단계 PR 에서 추가
CREATE TABLE IF NOT EXISTS thinking_sources (
    thinking_id INTEGER NOT NULL REFERENCES thinkings(id) ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES sources(id)   ON DELETE CASCADE,
    PRIMARY KEY (thinking_id, source_id)
);
CREATE INDEX idx_thinking_sources_source ON thinking_sources(source_id);

CREATE TABLE IF NOT EXISTS thinking_reuses (
    reuser_id  INTEGER NOT NULL REFERENCES thinkings(id) ON DELETE CASCADE,
    reused_id  INTEGER NOT NULL REFERENCES thinkings(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    PRIMARY KEY (reuser_id, reused_id)
);
```

마이그레이션 스크립트 (현재 데이터 보존):

```python
def migrate_v1_to_v2(conn):
    conn.executescript(NEW_SCHEMA)
    for row in conn.execute("SELECT id, supports_source_ids FROM thinkings"):
        for sid in json.loads(row["supports_source_ids"]):
            conn.execute(
                "INSERT OR IGNORE INTO thinking_sources(thinking_id, source_id) VALUES (?,?)",
                (row["id"], sid),
            )
    conn.commit()
```

이후 4-hop 예시 질의:

```sql
WITH base_sources AS (
    SELECT source_id FROM thinking_sources WHERE thinking_id = :tid
),
related_thinkings AS (
    SELECT DISTINCT ts.thinking_id
    FROM thinking_sources ts
    JOIN base_sources b ON b.source_id = ts.source_id
    WHERE ts.thinking_id != :tid
)
SELECT s.*
FROM thinking_sources ts
JOIN related_thinkings r ON r.thinking_id = ts.thinking_id
JOIN sources s ON s.id = ts.source_id
WHERE ts.source_id NOT IN (SELECT source_id FROM base_sources);
```

**한 SQL 쿼리, 인덱스 두 개로 풀린다.** Neo4j 마이그 없이도 spec 의 그래프 needs 는 충분히 커버된다. 정말로 부족해지는 시점이 와야 hybrid 로 가는 게 맞다.
