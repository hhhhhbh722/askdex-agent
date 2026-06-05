const BASE = "/api/v1";

// ---- 类型 ----

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  model?: string;
  temperature?: number;
  max_tokens?: number;
  conversation_id?: string;
  mode?: "react" | "plan_execute";
}

export interface ChatResponse {
  id: string;
  model: string;
  content: string;
  trace_id?: string;
  usage?: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
  steps?: AgentStep[];
  mode?: string;
}

export interface AgentStep {
  step?: number;
  phase?: string;
  ts?: number;
  action?: string;
  action_input?: unknown;
  observation?: string;
  raw_llm?: string;
  parsed?: Record<string, unknown>;
  final?: boolean;
  error?: string;
  status?: string;
  subtask_id?: string;
  title?: string;
  action_type?: string;
  tool_name?: string;
  llm_output?: string;
  record?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface StreamChunk {
  content?: string;
  trace_id?: string;
  done?: boolean;
  error?: string;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  mime_type?: string;
  status: string;
  group: string;
  parent_group: string;
  child_group: string;
  chunk_count: number;
  created_at?: string;
}

export interface UploadResponse {
  document_id: string;
  filename: string;
  status: string;
  chunk_count: number;
  group: string;
  parent_group: string;
  child_group: string;
  message: string;
}

export interface BatchUploadResponse {
  id?: string;
  job_id?: string;
  status: string;
  total: number;
  processed?: number;
  success: number;
  failed: number;
  current?: string;
  group?: string;
  parent_group?: string;
  child_group?: string;
  results: UploadResponse[];
}

export interface HealthStatus {
  status: string;
  database?: string;
  api?: string;
  redis?: string;
  milvus?: string;
  app_env?: string;
}

export interface FullHealthStatus {
  status: string;
  api: string;
  database: string;
  redis: string;
  milvus: string;
  app_env?: string;
}

// ---- 非流式对话 ----

export async function chat(request: ChatRequest): Promise<ChatResponse> {
  const resp = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(err || `Chat failed (${resp.status})`);
  }
  return resp.json();
}

// ---- SSE 流式对话 ----

export async function chatStream(
  request: ChatRequest,
  onToken: (chunk: StreamChunk) => void,
  signal?: AbortSignal
): Promise<void> {
  const resp = await fetch(`${BASE}/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(request),
    signal,
  });

  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(err || `Stream failed (${resp.status})`);
  }

  const reader = resp.body?.getReader();
  if (!reader) throw new Error("浏览器不支持流式响应");

  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const line = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (line.startsWith("data:")) {
        const payload = line.slice(5).trim();
        if (payload) {
          try {
            const parsed = JSON.parse(payload) as StreamChunk;
            onToken(parsed);
            if (parsed.done || parsed.error) return;
          } catch {
            // 忽略解析失败的 SSE 行
          }
        }
      }
      boundary = buffer.indexOf("\n\n");
    }

    if (done) break;
  }
}

// ---- 文档 ----

export async function uploadDocument(
  file: File,
  group = "",
  parentGroup = "",
  childGroup = ""
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("group", group);
  form.append("parent_group", parentGroup);
  form.append("child_group", childGroup);

  const resp = await fetch(`${BASE}/documents/upload`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(err || `Upload failed (${resp.status})`);
  }
  return resp.json();
}

export async function uploadDocuments(
  files: File[],
  group = "",
  parentGroup = "",
  childGroup = ""
): Promise<BatchUploadResponse> {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  form.append("group", group);
  form.append("parent_group", parentGroup);
  form.append("child_group", childGroup);

  const resp = await fetch(`${BASE}/documents/batch-upload`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const err = await resp.text().catch(() => "");
    throw new Error(err || `Batch upload failed (${resp.status})`);
  }
  return resp.json();
}

export async function getUploadJob(jobId: string): Promise<BatchUploadResponse> {
  const resp = await fetch(`${BASE}/documents/jobs/${jobId}`);
  if (!resp.ok) throw new Error(`Upload job failed (${resp.status})`);
  return resp.json();
}

export async function listDocuments(): Promise<DocumentInfo[]> {
  const resp = await fetch(`${BASE}/documents`);
  if (!resp.ok) throw new Error(`List failed (${resp.status})`);
  return resp.json();
}

export async function deleteDocument(id: string): Promise<void> {
  const resp = await fetch(`${BASE}/documents/${id}`, { method: "DELETE" });
  if (!resp.ok) throw new Error(`Delete failed (${resp.status})`);
}

export async function updateDocumentGroup(
  id: string,
  payload: { group?: string; parent_group?: string; child_group?: string }
): Promise<{ id: string; group: string; parent_group: string; child_group: string }> {
  const resp = await fetch(`${BASE}/documents/${id}/group`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw new Error(`Update group failed (${resp.status})`);
  return resp.json();
}

// ---- 对话历史 ----

export interface ConversationSummary {
  id: string;
  title: string;
  preview: string;
  message_count: number;
  mode: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface ConversationDetail {
  id: string;
  title: string;
  mode: string;
  messages: { role: string; content: string; created_at: string | null }[];
  created_at: string | null;
}

export async function saveConversation(payload: {
  id?: string;
  title: string;
  messages: { role: string; content: string }[];
  mode?: string;
}): Promise<{ id: string }> {
  const resp = await fetch(`${BASE}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) throw new Error(`Save failed (${resp.status})`);
  return resp.json();
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const resp = await fetch(`${BASE}/conversations`);
  if (!resp.ok) throw new Error(`List failed (${resp.status})`);
  return resp.json();
}

export async function getConversation(id: string): Promise<ConversationDetail> {
  const resp = await fetch(`${BASE}/conversations/${id}`);
  if (!resp.ok) throw new Error(`Get failed (${resp.status})`);
  return resp.json();
}

export async function deleteConversation(id: string): Promise<void> {
  const resp = await fetch(`${BASE}/conversations/${id}`, { method: "DELETE" });
  if (!resp.ok) throw new Error(`Delete failed (${resp.status})`);
}

// ---- 完整健康检查 ----

export async function healthFull(): Promise<FullHealthStatus> {
  const resp = await fetch(`${BASE}/health/full`);
  if (!resp.ok) throw new Error(`Health check failed (${resp.status})`);
  return resp.json();
}

// ---- 健康检查 ----

export async function healthCheck(): Promise<HealthStatus> {
  const resp = await fetch(`${BASE}/health`);
  return resp.json();
}
