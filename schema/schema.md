# Neo4j research graph schema

## Node labels
- Topic
- ResearchObjective
- SourceSummary
- SourceChunk
- ThinkingSummary
- ThinkingChunk

## Relationships
- BELONGS_TO
- CHILD_OF
- REFERENCES

## Parent rules
- `ResearchObjective` may belong to `Topic`
- `SourceSummary` must belong to `Topic`
- `ThinkingSummary` must belong to `Topic`
- `SourceSummary` may additionally belong to `ResearchObjective`
- `ThinkingSummary` may additionally belong to `ResearchObjective`
- `SourceChunk` may belong only to `SourceSummary` through `CHILD_OF`
- `ThinkingChunk` may belong only to `ThinkingSummary` through `CHILD_OF`

## Time format
All timestamps use ISO 8601 with timezone.
Example: `2026-04-08T14:04:00+09:00`

## Embedding policy
- `SourceSummary.embedding` uses `summary` only
- `ThinkingSummary.embedding` uses `summary` only
- `SourceChunk.embedding` uses `text` only
- `ThinkingChunk.embedding` uses `text` only
- metadata is never embedded

## ResearchObjective purpose
`Topic` is the broad long-lived area.
`ResearchObjective` is the concrete research goal inside a topic.
Scheduled research can start by reading active `ResearchObjective.objective` values.
