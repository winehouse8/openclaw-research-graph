CREATE CONSTRAINT topic_id_unique IF NOT EXISTS
FOR (t:Topic)
REQUIRE t.topic_id IS UNIQUE;

CREATE CONSTRAINT research_objective_id_unique IF NOT EXISTS
FOR (ro:ResearchObjective)
REQUIRE ro.objective_id IS UNIQUE;

CREATE CONSTRAINT source_node_id_unique IF NOT EXISTS
FOR (s:SourceSummary)
REQUIRE s.node_id IS UNIQUE;

CREATE CONSTRAINT source_chunk_node_id_unique IF NOT EXISTS
FOR (c:SourceChunk)
REQUIRE c.node_id IS UNIQUE;

CREATE CONSTRAINT thinking_node_id_unique IF NOT EXISTS
FOR (th:ThinkingSummary)
REQUIRE th.node_id IS UNIQUE;

CREATE CONSTRAINT thinking_chunk_node_id_unique IF NOT EXISTS
FOR (tc:ThinkingChunk)
REQUIRE tc.node_id IS UNIQUE;

CREATE INDEX topic_name_idx IF NOT EXISTS
FOR (t:Topic)
ON (t.name);

CREATE INDEX objective_name_idx IF NOT EXISTS
FOR (ro:ResearchObjective)
ON (ro.name);

CREATE INDEX objective_status_idx IF NOT EXISTS
FOR (ro:ResearchObjective)
ON (ro.status);

CREATE INDEX source_url_idx IF NOT EXISTS
FOR (s:SourceSummary)
ON (s.canonical_url);

CREATE INDEX source_kind_idx IF NOT EXISTS
FOR (s:SourceSummary)
ON (s.source_kind);

CREATE INDEX thinking_kind_idx IF NOT EXISTS
FOR (th:ThinkingSummary)
ON (th.thinking_kind);

CREATE INDEX source_created_at_idx IF NOT EXISTS
FOR (s:SourceSummary)
ON (s.created_at);

CREATE INDEX thinking_created_at_idx IF NOT EXISTS
FOR (th:ThinkingSummary)
ON (th.created_at);

CREATE INDEX thinking_updated_at_idx IF NOT EXISTS
FOR (th:ThinkingSummary)
ON (th.updated_at);
