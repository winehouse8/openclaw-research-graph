CREATE VECTOR INDEX source_summary_embedding_idx IF NOT EXISTS
FOR (s:SourceSummary)
ON (s.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX thinking_summary_embedding_idx IF NOT EXISTS
FOR (th:ThinkingSummary)
ON (th.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX source_chunk_embedding_idx IF NOT EXISTS
FOR (c:SourceChunk)
ON (c.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};

CREATE VECTOR INDEX thinking_chunk_embedding_idx IF NOT EXISTS
FOR (tc:ThinkingChunk)
ON (tc.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1024,
  `vector.similarity_function`: 'cosine'
}};
