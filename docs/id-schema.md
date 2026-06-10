# ID Schema

This project keeps object IDs and execution IDs separate.

| ID | Meaning | Lifecycle | Storage | Notes |
| --- | --- | --- | --- | --- |
| `conversation_id` | One conversation thread | Long-lived | PostgreSQL, Redis memory key | Also used as Agent `session_id`. |
| `session_id` | Runtime memory session | Same as conversation | Redis `stm:session:{session_id}` | Internal memory API name. |
| `trace_id` | One Agent run / request chain | Short or medium | OTel, trace store, response | Prefer OpenTelemetry trace_id when OTel is active. |
| `span_id` | One operation inside a trace | Short | OTel | Usually not exposed as a business ID. |
| `job_id` | One background job | Medium | Redis/PostgreSQL/in-memory job state | Upload, reindex, KG rebuild/enrich keep their own job IDs. |
| `document_id` | One uploaded source document | Long-lived | PostgreSQL, Milvus metadata | Used for source filtering and KG provenance. |
| `chunk_id` | One document chunk | Long-lived | PostgreSQL | Usually equals Milvus `vector_id`. |
| `vector_id` | Milvus primary key | Long-lived | Milvus | Defaults to `chunk_id` for traceable deletion/reindex. |
| `entity_id` | One KG entity | Long-lived | PostgreSQL | Stored in `kg_entities.id`. |
| `relation_id` | One KG relation | Long-lived | PostgreSQL | Stored in `kg_relations.id`. |
| `source_document_id` | KG/RAG provenance document | Long-lived reference | PostgreSQL | Points back to `documents.id`. |
| `source_chunk_id` | KG/RAG provenance chunk | Long-lived reference | PostgreSQL | Points back to `document_chunks.id`. |

## Rules

- `conversation_id` identifies a whole chat thread.
- `trace_id` identifies one Agent execution inside a conversation.
- One `conversation_id` can contain many `trace_id` values.
- One `trace_id` contains many `span_id` values.
- `job_id` is a business task ID and should not be replaced by `trace_id`.
- `document_id`, `chunk_id`, and `vector_id` model knowledge storage and should not be replaced by trace IDs.

## OpenTelemetry

When OpenTelemetry is enabled, use the active OTel trace ID as the Agent run
`trace_id`.

```text
conversation_id: long-lived user thread
trace_id: OTel trace_id for one Agent run
span_id: OTel span_id for one operation
```

This lets the frontend, Redis TraceStore, logs, and Jaeger/Grafana use the same
execution chain ID.
