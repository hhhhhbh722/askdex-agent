# Agent Improvement TODO

## Done in this pass

- [x] Standardize tool descriptions and parameter schema in the ReAct prompt.
- [x] Fix Plan & Execute tool argument mapping for registered tools.
- [x] Route chat requests through intent recognition before Agent execution.
- [x] Wrap tool calls with timeout, duration, and structured error reporting.
- [x] Enable conditional reflection for risky or low-confidence runs.
- [x] Feed recent Agent trace summaries into `/api/v1/metrics` and MonitorView.

## Next candidates

- [ ] Revisit long-term memory design before connecting a dedicated Milvus memory collection.
- [ ] Revisit Agent cache hierarchy: task-level scratchpad, session/user Redis memory, long-term vector memory.
- [ ] Persist Agent trace root/spans into PostgreSQL `trace_logs` after deciding the trace detail API shape.
- [ ] Add a frontend memory/debug panel for the active session.
- [ ] Add regression tests for ReAct parsing, tool argument coercion, and intent routing.
- [ ] Add trace detail API backed by `trace_logs` for MonitorView drill-down after backend restart.
- [ ] Add explicit long-term memory management UI: pin, forget, and inspect memories.
