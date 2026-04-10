from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunking import chunk_text
from .db import session_scope
from .embed import embed_text
from .utils import now_iso, slugify, stable_id


def _node_to_dict(node, include_embedding: bool = True) -> Dict[str, Any]:
    data = dict(node)
    data['labels'] = sorted(list(node.labels))
    if not include_embedding:
        data.pop('embedding', None)
    return data


def _trim_text(value: Optional[str], limit: int = 240) -> str:
    text = (value or '').strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + '...'


def _compact_node_view(node: Dict[str, Any], fallback_labels: Optional[List[str]] = None) -> Dict[str, Any]:
    labels = node.get('labels') or fallback_labels or []
    return {
        'node_id': node.get('node_id') or node.get('objective_id') or node.get('topic_id'),
        'labels': labels,
        'title': node.get('title') or node.get('name'),
        'summary_preview': _trim_text(node.get('summary') or node.get('text') or node.get('objective')),
        'canonical_url': node.get('canonical_url'),
        'source_kind': node.get('source_kind'),
        'updated_at': node.get('updated_at'),
    }


def _slim_hit_view(hit: Dict[str, Any]) -> Dict[str, Any]:
    node = hit.get('node_preview') or _compact_node_view(hit.get('node') or {})
    return {
        'score': hit.get('score'),
        'node_id': node.get('node_id'),
        'labels': node.get('labels', []),
        'title': node.get('title'),
        'summary_preview': node.get('summary_preview'),
        'match_reasons': hit.get('match_reasons', []),
        'one_hop_preview': hit.get('one_hop_preview', {}),
    }


def _infer_open_questions(
    objective: Dict[str, Any],
    thinking_summaries: List[Dict[str, Any]],
    source_summaries: List[Dict[str, Any]],
) -> List[str]:
    questions: List[str] = []
    objective_text = (objective.get('objective') or objective.get('name') or '').strip()
    if objective_text:
        questions.append(f"What evidence would materially change the current answer to: {objective_text}?")
    if source_summaries:
        questions.append('Which source most strongly supports the current conclusion, and which source most strongly limits confidence?')
    if thinking_summaries:
        questions.append('What changed in the latest thinking summary compared with the baseline view?')
    if not questions:
        questions.append('What evidence is still missing for this objective?')
    return questions[:3]


def _build_review_summary(
    objective: Dict[str, Any],
    thinking_summaries: List[Dict[str, Any]],
    source_summaries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    latest_thinking = thinking_summaries[0] if thinking_summaries else {}
    baseline_thinking = thinking_summaries[1] if len(thinking_summaries) > 1 else latest_thinking
    strongest_source = source_summaries[0] if source_summaries else {}
    return {
        'objective_name': objective.get('name'),
        'current_conclusion_preview': _trim_text(latest_thinking.get('summary') or latest_thinking.get('text') or objective.get('objective')),
        'what_changed_preview': _trim_text(latest_thinking.get('summary')) if latest_thinking else None,
        'baseline_preview': _trim_text(baseline_thinking.get('summary')) if baseline_thinking else None,
        'strongest_source_preview': _compact_node_view(strongest_source) if strongest_source else None,
        'confidence_note': 'Provisional, because the current objective is supported by a small set of source summaries and lightweight reflection nodes.',
    }


def _fetch_chunk_previews(
    session,
    label: str,
    parent_label: str,
    parent_ids: List[str],
    limit: int,
    include_embedding: bool,
) -> List[Dict[str, Any]]:
    if not parent_ids:
        return []

    query = f'''
    MATCH (c:{label})-[:CHILD_OF]->(p:{parent_label})
    WHERE p.node_id IN $parent_ids
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(t:Topic)
    OPTIONAL MATCH (p)-[:BELONGS_TO]->(ro:ResearchObjective)
    RETURN c, p, t, ro
    ORDER BY c.updated_at DESC
    LIMIT $limit
    '''
    previews: List[Dict[str, Any]] = []
    for rec in session.run(query, {'parent_ids': parent_ids, 'limit': limit}):
        chunk = _node_to_dict(rec['c'], include_embedding=include_embedding)
        parent = _node_to_dict(rec['p'], include_embedding=False)
        previews.append({
            'node': chunk,
            'node_preview': _compact_node_view(chunk),
            'parent_summary': _compact_node_view(parent),
            'parent_type': parent_label,
            'topic': _compact_node_view(_node_to_dict(rec['t'], include_embedding=False)) if rec['t'] is not None else None,
            'objective': _compact_node_view(_node_to_dict(rec['ro'], include_embedding=False)) if rec['ro'] is not None else None,
        })
    return previews


def _records_to_dicts(records, key: str, include_embedding: bool = True) -> List[Dict[str, Any]]:
    out = []
    for record in records:
        node = record.get(key)
        if node is not None:
            out.append(_node_to_dict(node, include_embedding=include_embedding))
    return out


def find_or_create_research_objective(
    objective_text: str,
    topic_hint: Optional[str] = None,
    name_hint: Optional[str] = None,
    success_criteria: Optional[str] = None,
) -> Dict[str, Any]:
    text = (objective_text or '').strip()
    if not text:
        return {'status': 'error', 'message': 'objective_text is required'}

    with session_scope() as session:
        candidates_query = '''
        MATCH (ro:ResearchObjective)
        OPTIONAL MATCH (ro)-[:BELONGS_TO]->(t:Topic)
        WHERE $topic_hint IS NULL OR t.topic_id = $topic_hint OR toLower(t.name) CONTAINS toLower($topic_hint)
        RETURN ro, t
        ORDER BY ro.updated_at DESC
        LIMIT 10
        '''
        records = list(session.run(candidates_query, {'topic_hint': topic_hint}))
        candidates = []
        text_tokens = {tok for tok in text.lower().split() if len(tok) >= 4}
        for r in records:
            ro = dict(r['ro'])
            t = dict(r['t']) if r['t'] is not None else None
            objective_text_existing = (ro.get('objective') or '').strip().lower()
            name_existing = (ro.get('name') or '').strip().lower()
            existing_tokens = {tok for tok in f"{name_existing} {objective_text_existing}".split() if len(tok) >= 4}
            overlap = len(text_tokens & existing_tokens)
            same_topic = False
            if topic_hint and t is not None:
                same_topic = t.get('topic_id') == topic_hint or topic_hint.lower() in (t.get('name') or '').lower()
            exact_objective_match = text.strip().lower() == objective_text_existing
            exact_name_match = bool(name_hint) and (name_hint.strip().lower() == name_existing)
            candidates.append({
                'objective': ro,
                'topic': t,
                'token_overlap': overlap,
                'same_topic': same_topic,
                'exact_objective_match': exact_objective_match,
                'exact_name_match': exact_name_match,
            })

        candidates = [c for c in candidates if c['token_overlap'] >= 2 or c['exact_objective_match'] or c['exact_name_match']]
        candidates.sort(key=lambda x: (x['exact_objective_match'], x['exact_name_match'], x['same_topic'], x['token_overlap']), reverse=True)

        if candidates:
            top = candidates[0]
            if top['exact_objective_match'] and (top['same_topic'] or not topic_hint):
                return {'status': 'matched_existing', 'objective': top['objective'], 'topic': top['topic'], 'candidates': candidates[:5]}
            if top['exact_name_match'] and top['same_topic'] and top['token_overlap'] >= 2:
                return {'status': 'matched_existing', 'objective': top['objective'], 'topic': top['topic'], 'candidates': candidates[:5]}
            return {'status': 'ambiguous', 'candidates': candidates[:5]}

        now = now_iso()
        topic_id = topic_hint if topic_hint and topic_hint.startswith('topic-') else None
        if not topic_id:
            topic_id = f"topic-{slugify(topic_hint or name_hint or text[:50])}"
        topic_name = topic_hint or name_hint or text[:80]
        objective_id = stable_id('objective', topic_id, text)
        name = name_hint or text[:80]

        create_query = '''
        MERGE (t:Topic {topic_id: $topic_id})
        SET t.name = coalesce(t.name, $topic_name),
            t.summary = coalesce(t.summary, $topic_name),
            t.created_at = coalesce(t.created_at, $now),
            t.updated_at = $now,
            t.status = coalesce(t.status, 'active')
        MERGE (ro:ResearchObjective {objective_id: $objective_id})
        SET ro.name = $name,
            ro.objective = $objective,
            ro.success_criteria = $success_criteria,
            ro.status = 'active',
            ro.priority = 1,
            ro.created_at = coalesce(ro.created_at, $now),
            ro.updated_at = $now
        MERGE (ro)-[:BELONGS_TO]->(t)
        RETURN ro, t
        '''
        rec = session.run(create_query, {
            'topic_id': topic_id,
            'topic_name': topic_name,
            'objective_id': objective_id,
            'name': name,
            'objective': text,
            'success_criteria': success_criteria,
            'now': now,
        }).single()
        return {'status': 'created_new', 'objective': dict(rec['ro']), 'topic': dict(rec['t'])}


def get_research_context(
    objective_id: Optional[str] = None,
    topic_id: Optional[str] = None,
    thinking_limit: int = 5,
    source_limit: int = 5,
    chunk_limit: int = 6,
    include_embedding: bool = False,
) -> Dict[str, Any]:
    with session_scope() as session:
        payload: Dict[str, Any] = {'objective': None, 'topic': None, 'latest_thinking_summaries': [], 'relevant_source_summaries': [], 'representative_chunks': []}

        if objective_id:
            rec = session.run('''
            MATCH (ro:ResearchObjective {objective_id: $objective_id})-[:BELONGS_TO]->(t:Topic)
            RETURN ro, t
            ''', {'objective_id': objective_id}).single()
            if rec:
                payload['objective'] = dict(rec['ro'])
                payload['topic'] = dict(rec['t'])
                topic_id = rec['t']['topic_id']

        if topic_id and not payload['topic']:
            rec = session.run('MATCH (t:Topic {topic_id: $topic_id}) RETURN t', {'topic_id': topic_id}).single()
            if rec:
                payload['topic'] = dict(rec['t'])

        if objective_id:
            payload['latest_thinking_summaries'] = _records_to_dicts(session.run('''
            MATCH (th:ThinkingSummary)-[:BELONGS_TO]->(:ResearchObjective {objective_id: $objective_id})
            RETURN th ORDER BY th.updated_at DESC LIMIT $limit
            ''', {'objective_id': objective_id, 'limit': thinking_limit}), 'th', include_embedding=include_embedding)
            payload['relevant_source_summaries'] = _records_to_dicts(session.run('''
            MATCH (s:SourceSummary)-[:BELONGS_TO]->(:ResearchObjective {objective_id: $objective_id})
            RETURN s ORDER BY s.updated_at DESC LIMIT $limit
            ''', {'objective_id': objective_id, 'limit': source_limit}), 's', include_embedding=include_embedding)
        elif topic_id:
            payload['latest_thinking_summaries'] = _records_to_dicts(session.run('''
            MATCH (th:ThinkingSummary)-[:BELONGS_TO]->(:Topic {topic_id: $topic_id})
            RETURN DISTINCT th ORDER BY th.updated_at DESC LIMIT $limit
            ''', {'topic_id': topic_id, 'limit': thinking_limit}), 'th', include_embedding=include_embedding)
            payload['relevant_source_summaries'] = _records_to_dicts(session.run('''
            MATCH (s:SourceSummary)-[:BELONGS_TO]->(:Topic {topic_id: $topic_id})
            RETURN DISTINCT s ORDER BY s.updated_at DESC LIMIT $limit
            ''', {'topic_id': topic_id, 'limit': source_limit}), 's', include_embedding=include_embedding)

        source_ids = [s['node_id'] for s in payload['relevant_source_summaries'][:3] if 'node_id' in s]
        thinking_ids = [t['node_id'] for t in payload['latest_thinking_summaries'][:3] if 'node_id' in t]
        payload['representative_chunks'].extend(_fetch_chunk_previews(
            session,
            'SourceChunk',
            'SourceSummary',
            source_ids,
            chunk_limit,
            include_embedding,
        ))
        payload['representative_chunks'].extend(_fetch_chunk_previews(
            session,
            'ThinkingChunk',
            'ThinkingSummary',
            thinking_ids,
            chunk_limit,
            include_embedding,
        ))

        payload['context_preview'] = {
            'objective': _compact_node_view(payload['objective'], fallback_labels=['ResearchObjective']) if payload['objective'] else None,
            'topic': _compact_node_view(payload['topic'], fallback_labels=['Topic']) if payload['topic'] else None,
            'latest_thinking_summaries': [_compact_node_view(item) for item in payload['latest_thinking_summaries']],
            'relevant_source_summaries': [_compact_node_view(item) for item in payload['relevant_source_summaries']],
            'representative_chunks': [
                {
                    'node_preview': item.get('node_preview'),
                    'parent_summary': item.get('parent_summary'),
                    'parent_type': item.get('parent_type'),
                    'topic': item.get('topic'),
                    'objective': item.get('objective'),
                }
                for item in payload['representative_chunks']
            ],
        }

        return payload


def upsert_source_batch(topic_id: str, sources: List[Dict[str, Any]], objective_id: Optional[str] = None) -> Dict[str, Any]:
    now = now_iso()
    created, reused, skipped = [], [], []
    with session_scope() as session:
        for src in sources:
            canonical_url = (src.get('canonical_url') or '').strip()
            summary = (src.get('summary') or '').strip()
            title = (src.get('title') or '').strip()
            if not canonical_url or not summary or not title:
                skipped.append({'reason': 'missing_required_fields', 'source': src})
                continue

            existing = session.run('MATCH (s:SourceSummary {canonical_url: $url}) RETURN s', {'url': canonical_url}).single()
            if existing:
                node = dict(existing['s'])
                reused.append(node)
                if objective_id:
                    session.run('''
                    MATCH (s:SourceSummary {node_id: $node_id}), (ro:ResearchObjective {objective_id: $objective_id})
                    MERGE (s)-[:BELONGS_TO]->(ro)
                    ''', {'node_id': node['node_id'], 'objective_id': objective_id})
                continue

            node_id = src.get('node_id') or stable_id('src', canonical_url, title)
            chunks = src.get('chunks') or chunk_text(summary)
            source_published_at = src.get('source_published_at')
            source_kind = src.get('source_kind') or 'web'
            embedding = embed_text(summary)

            session.run('''
            MATCH (t:Topic {topic_id: $topic_id})
            MERGE (s:SourceSummary {node_id: $node_id})
            SET s.title = $title,
                s.summary = $summary,
                s.canonical_url = $canonical_url,
                s.normalized_url = $canonical_url,
                s.source_kind = $source_kind,
                s.source_published_at = $source_published_at,
                s.created_at = coalesce(s.created_at, $now),
                s.updated_at = $now,
                s.status = 'active',
                s.embedding = $embedding
            MERGE (s)-[:BELONGS_TO]->(t)
            ''', {
                'topic_id': topic_id,
                'node_id': node_id,
                'title': title,
                'summary': summary,
                'canonical_url': canonical_url,
                'source_kind': source_kind,
                'source_published_at': source_published_at,
                'now': now,
                'embedding': embedding,
            })
            if objective_id:
                session.run('''
                MATCH (s:SourceSummary {node_id: $node_id}), (ro:ResearchObjective {objective_id: $objective_id})
                MERGE (s)-[:BELONGS_TO]->(ro)
                ''', {'node_id': node_id, 'objective_id': objective_id})

            for idx, chunk in enumerate(chunks, start=1):
                chunk_id = stable_id('srcchunk', node_id, str(idx), chunk[:80])
                session.run('''
                MATCH (s:SourceSummary {node_id: $node_id})
                MERGE (c:SourceChunk {node_id: $chunk_id})
                SET c.chunk_index = $chunk_index,
                    c.summary = $chunk_summary,
                    c.text = $chunk_text,
                    c.created_at = coalesce(c.created_at, $now),
                    c.updated_at = $now,
                    c.embedding = $embedding
                MERGE (c)-[:CHILD_OF]->(s)
                ''', {
                    'node_id': node_id,
                    'chunk_id': chunk_id,
                    'chunk_index': idx,
                    'chunk_summary': chunk[:180],
                    'chunk_text': chunk,
                    'now': now,
                    'embedding': embed_text(chunk),
                })
            created.append({'node_id': node_id, 'title': title, 'canonical_url': canonical_url})
    return {'created': created, 'reused_existing': reused, 'skipped': skipped}


def upsert_thinking_batch(topic_id: str, thinkings: List[Dict[str, Any]], objective_id: Optional[str] = None) -> Dict[str, Any]:
    now = now_iso()
    created, skipped = [], []
    with session_scope() as session:
        for item in thinkings:
            title = (item.get('title') or '').strip()
            summary = (item.get('summary') or '').strip()
            thinking_kind = (item.get('thinking_kind') or 'analysis').strip()
            if not title or not summary:
                skipped.append({'reason': 'missing_required_fields', 'thinking': item})
                continue

            node_id = item.get('node_id') or stable_id('th', topic_id, title, summary)
            chunks = item.get('chunks') or chunk_text(summary)
            refs = item.get('references') or []
            session.run('''
            MATCH (t:Topic {topic_id: $topic_id})
            MERGE (th:ThinkingSummary {node_id: $node_id})
            SET th.title = $title,
                th.summary = $summary,
                th.thinking_kind = $thinking_kind,
                th.created_at = coalesce(th.created_at, $now),
                th.updated_at = $now,
                th.status = 'active',
                th.embedding = $embedding
            MERGE (th)-[:BELONGS_TO]->(t)
            ''', {
                'topic_id': topic_id,
                'node_id': node_id,
                'title': title,
                'summary': summary,
                'thinking_kind': thinking_kind,
                'now': now,
                'embedding': embed_text(summary),
            })
            if objective_id:
                session.run('''
                MATCH (th:ThinkingSummary {node_id: $node_id}), (ro:ResearchObjective {objective_id: $objective_id})
                MERGE (th)-[:BELONGS_TO]->(ro)
                ''', {'node_id': node_id, 'objective_id': objective_id})

            for ref_id in refs:
                session.run('''
                MATCH (th:ThinkingSummary {node_id: $node_id})
                MATCH (n)
                WHERE (n:SourceSummary OR n:ThinkingSummary) AND n.node_id = $ref_id
                MERGE (th)-[:REFERENCES]->(n)
                ''', {'node_id': node_id, 'ref_id': ref_id})

            for idx, chunk in enumerate(chunks, start=1):
                chunk_id = stable_id('thchunk', node_id, str(idx), chunk[:80])
                session.run('''
                MATCH (th:ThinkingSummary {node_id: $node_id})
                MERGE (c:ThinkingChunk {node_id: $chunk_id})
                SET c.chunk_index = $chunk_index,
                    c.summary = $chunk_summary,
                    c.text = $chunk_text,
                    c.created_at = coalesce(c.created_at, $now),
                    c.updated_at = $now,
                    c.embedding = $embedding
                MERGE (c)-[:CHILD_OF]->(th)
                ''', {
                    'node_id': node_id,
                    'chunk_id': chunk_id,
                    'chunk_index': idx,
                    'chunk_summary': chunk[:180],
                    'chunk_text': chunk,
                    'now': now,
                    'embedding': embed_text(chunk),
                })
            created.append({'node_id': node_id, 'title': title})
    return {'created': created, 'skipped': skipped}


def search_graph(
    query: str,
    topic_id: Optional[str] = None,
    objective_id: Optional[str] = None,
    limit: int = 8,
    include_embedding: bool = False,
) -> Dict[str, Any]:
    q = (query or '').strip()
    if not q:
        return {'hits': []}

    embedding = embed_text(q)

    def _collect(session, index_name: str, label: str, per_index: int) -> List[Dict[str, Any]]:
        cypher = f'''
        CALL db.index.vector.queryNodes("{index_name}", $k, $embedding)
        YIELD node, score
        OPTIONAL MATCH (node)-[:BELONGS_TO]->(t:Topic)
        OPTIONAL MATCH (node)-[:BELONGS_TO]->(ro:ResearchObjective)
        OPTIONAL MATCH (node)-[:CHILD_OF]->(parent)
        OPTIONAL MATCH (parent)-[:BELONGS_TO]->(pt:Topic)
        OPTIONAL MATCH (parent)-[:BELONGS_TO]->(pro:ResearchObjective)
        WHERE ($topic_id IS NULL OR t.topic_id = $topic_id OR pt.topic_id = $topic_id)
          AND ($objective_id IS NULL OR ro.objective_id = $objective_id OR pro.objective_id = $objective_id)
        RETURN node, score, t, ro, parent, pt, pro
        '''
        out = []
        for rec in session.run(cypher, {'k': per_index, 'embedding': embedding, 'topic_id': topic_id, 'objective_id': objective_id}):
            node = _node_to_dict(rec['node'], include_embedding=include_embedding)
            parent = _node_to_dict(rec['parent'], include_embedding=False) if rec['parent'] is not None else None
            out.append({
                'score': float(rec['score']),
                'node': node,
                'node_preview': _compact_node_view(node),
                'match_reasons': [label, 'vector'],
                'one_hop_preview': {
                    'topic': _compact_node_view(_node_to_dict(rec['t'], include_embedding=False)) if rec['t'] is not None else None,
                    'objective': _compact_node_view(_node_to_dict(rec['ro'], include_embedding=False)) if rec['ro'] is not None else None,
                    'parent_summary': _compact_node_view(parent) if parent is not None else None,
                    'parent_topic': _compact_node_view(_node_to_dict(rec['pt'], include_embedding=False)) if rec['pt'] is not None else None,
                    'parent_objective': _compact_node_view(_node_to_dict(rec['pro'], include_embedding=False)) if rec['pro'] is not None else None,
                },
            })
        return out

    with session_scope() as session:
        hits: List[Dict[str, Any]] = []
        hits.extend(_collect(session, 'source_summary_embedding_idx', 'SourceSummary', limit))
        hits.extend(_collect(session, 'thinking_summary_embedding_idx', 'ThinkingSummary', limit))
        half = max(2, limit // 2)
        hits.extend(_collect(session, 'source_chunk_embedding_idx', 'SourceChunk', half))
        hits.extend(_collect(session, 'thinking_chunk_embedding_idx', 'ThinkingChunk', half))

        deduped: Dict[str, Dict[str, Any]] = {}
        for hit in hits:
            node_id = hit['node'].get('node_id')
            if not node_id:
                continue
            prev = deduped.get(node_id)
            if prev is None or hit['score'] > prev['score']:
                deduped[node_id] = hit
        ranked = sorted(deduped.values(), key=lambda x: x['score'], reverse=True)
        trimmed_hits = ranked[:limit]
        slim_hits = [_slim_hit_view(hit) for hit in trimmed_hits]
        return {
            'hits': trimmed_hits,
            'slim_hits': slim_hits,
            'primary_results': slim_hits,
            'hint': 'Use get_node_bundle(node_id=...) for deeper inspection.',
            'defaults': {'include_embedding': include_embedding},
        }


def get_node_bundle(node_id: str, include_embedding: bool = False) -> Dict[str, Any]:
    with session_scope() as session:
        rec = session.run('MATCH (n {node_id: $node_id}) RETURN n', {'node_id': node_id}).single()
        if not rec:
            return {'status': 'not_found', 'node_id': node_id}
        node = _node_to_dict(rec['n'], include_embedding=include_embedding)
        parents = [_node_to_dict(r['p'], include_embedding=include_embedding) for r in session.run('MATCH (n {node_id: $node_id})-[:BELONGS_TO]->(p) RETURN p', {'node_id': node_id})]
        refs_out = [_node_to_dict(r['x'], include_embedding=include_embedding) for r in session.run('MATCH (n {node_id: $node_id})-[:REFERENCES]->(x) RETURN x', {'node_id': node_id})]
        refs_in = [_node_to_dict(r['x'], include_embedding=include_embedding) for r in session.run('MATCH (x)-[:REFERENCES]->(n {node_id: $node_id}) RETURN x', {'node_id': node_id})]

        child_records = list(session.run('''
        MATCH (c)-[:CHILD_OF]->(n {node_id: $node_id})
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(t:Topic)
        OPTIONAL MATCH (c)-[:BELONGS_TO]->(ro:ResearchObjective)
        RETURN c, t, ro ORDER BY c.chunk_index ASC LIMIT 10
        ''', {'node_id': node_id}))
        children = []
        for rec in child_records:
            child = _node_to_dict(rec['c'], include_embedding=include_embedding)
            children.append({
                'node': child,
                'node_preview': _compact_node_view(child),
                'topic': _compact_node_view(_node_to_dict(rec['t'], include_embedding=False)) if rec['t'] is not None else None,
                'objective': _compact_node_view(_node_to_dict(rec['ro'], include_embedding=False)) if rec['ro'] is not None else None,
            })

        return {
            'status': 'ok',
            'node': node,
            'node_preview': _compact_node_view(node, fallback_labels=node.get('labels', [])),
            'parents': parents,
            'parent_previews': [_compact_node_view(item) for item in parents],
            'references_out': refs_out,
            'references_in': refs_in,
            'reference_previews': {
                'out': [_compact_node_view(item) for item in refs_out],
                'in': [_compact_node_view(item) for item in refs_in],
            },
            'primary_view': {
                'node': _compact_node_view(node, fallback_labels=node.get('labels', [])),
                'parents': [_compact_node_view(item) for item in parents],
                'references_out': [_compact_node_view(item) for item in refs_out],
                'references_in': [_compact_node_view(item) for item in refs_in],
            },
            'child_preview': children,
            'defaults': {'include_embedding': include_embedding},
        }


def review_objective_state(
    objective_id: str,
    query: Optional[str] = None,
    thinking_limit: int = 5,
    source_limit: int = 5,
    chunk_limit: int = 6,
    search_limit: int = 6,
) -> Dict[str, Any]:
    context = get_research_context(
        objective_id=objective_id,
        thinking_limit=thinking_limit,
        source_limit=source_limit,
        chunk_limit=chunk_limit,
        include_embedding=False,
    )
    objective = context.get('objective') or {}
    topic = context.get('topic') or {}
    effective_query = (query or objective.get('objective') or objective.get('name') or '').strip()
    search_result = search_graph(
        query=effective_query,
        topic_id=topic.get('topic_id'),
        objective_id=objective_id,
        limit=search_limit,
        include_embedding=False,
    ) if effective_query else {'hits': []}

    runtime_dir = Path(__file__).resolve().parents[2] / 'runtime'
    staging_candidates = sorted(runtime_dir.glob(f"{objective_id}*-staging.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    latest_staging = str(staging_candidates[0]) if staging_candidates else None

    thinking_summaries = context.get('latest_thinking_summaries', [])
    source_summaries = context.get('relevant_source_summaries', [])
    representative_chunks = context.get('representative_chunks', [])

    review_summary = _build_review_summary(objective, thinking_summaries, source_summaries)
    open_questions = _infer_open_questions(objective, thinking_summaries, source_summaries)

    return {
        'objective': objective,
        'objective_preview': _compact_node_view(objective, fallback_labels=['ResearchObjective']) if objective else None,
        'topic': topic,
        'topic_preview': _compact_node_view(topic, fallback_labels=['Topic']) if topic else None,
        'effective_query': effective_query,
        'latest_thinking_summaries': [_compact_node_view(item) for item in thinking_summaries],
        'relevant_source_summaries': [_compact_node_view(item) for item in source_summaries],
        'representative_chunks': [
            {
                'node_preview': item.get('node_preview'),
                'parent_summary': item.get('parent_summary'),
                'parent_type': item.get('parent_type'),
                'topic': item.get('topic'),
                'objective': item.get('objective'),
            }
            for item in representative_chunks[:chunk_limit]
        ],
        'top_search_hits': search_result.get('primary_results', search_result.get('slim_hits', search_result.get('hits', []))),
        'review_summary': review_summary,
        'open_questions': open_questions,
        'staging_file': latest_staging,
    }


def reindex_all_embeddings() -> Dict[str, Any]:
    updated = {'SourceSummary': 0, 'SourceChunk': 0, 'ThinkingSummary': 0, 'ThinkingChunk': 0}
    with session_scope() as session:
        for label, field in [
            ('SourceSummary', 'summary'),
            ('SourceChunk', 'text'),
            ('ThinkingSummary', 'summary'),
            ('ThinkingChunk', 'text'),
        ]:
            recs = list(session.run(f"MATCH (n:{label}) RETURN n"))
            for rec in recs:
                n = rec['n']
                value = n.get(field) or ''
                emb = embed_text(value)
                session.run(f"MATCH (n:{label} {{node_id: $node_id}}) SET n.embedding = $embedding, n.updated_at = $now", {
                    'node_id': n['node_id'],
                    'embedding': emb,
                    'now': now_iso(),
                })
                updated[label] += 1
    return {'updated': updated}
