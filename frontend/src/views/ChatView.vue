<template>
  <div class="chat-layout">
    <div class="messages" ref="msgContainer">
      <div v-if="messages.length === 0" class="welcome">
        <div class="welcome-icon">AI</div>
        <h2>AI Agent 对话工作台</h2>
        <p>Agent 会自动调用知识库、计算器等工具回答你的问题。</p>
        <div class="quick-actions">
          <button v-for="q in quickQuestions" :key="q" @click="sendQuick(q)">{{ q }}</button>
        </div>
      </div>

      <div v-for="(msg, i) in messages" :key="i" class="msg-row" :class="msg.role">
        <div class="msg-avatar">{{ msg.role === "user" ? "我" : "AI" }}</div>
        <div class="msg-bubble">
          <div v-if="msg.steps?.length" class="think-box">
            <details class="think-detail" open>
              <summary>
                <span>Agent 执行步骤</span>
                <span class="think-count">{{ msg.steps.length }} steps</span>
              </summary>

              <div class="think-steps">
                <details v-for="(s, si) in msg.steps" :key="si" class="think-step-card" open>
                  <summary class="think-step-summary">
                    <span class="think-step-num">Step {{ si + 1 }}</span>
                    <span class="think-step-title">{{ stepTitle(s, si) }}</span>
                    <span class="think-phase-badge">{{ stepPhase(s) }}</span>
                    <span v-if="stepStatus(s)" class="think-status-badge" :class="stepStatusClass(s)">
                      {{ stepStatus(s) }}
                    </span>
                  </summary>

                  <div class="think-step-body">
                    <div v-if="stepThought(s)" class="think-field">
                      <div class="think-field-label">Thought</div>
                      <div class="think-field-text">{{ stepThought(s) }}</div>
                    </div>

                    <div class="think-grid">
                      <div v-if="stepAction(s)" class="think-field">
                        <div class="think-field-label">Action</div>
                        <div class="think-pill">{{ stepAction(s) }}</div>
                      </div>

                      <div v-if="hasValue(stepToolArgs(s))" class="think-field">
                        <div class="think-field-label">Action Input</div>
                        <pre class="think-code">{{ formatValue(stepToolArgs(s)) }}</pre>
                      </div>
                    </div>

                    <div v-if="stepObservation(s)" class="think-field">
                      <div class="think-field-label">Observation</div>
                      <div class="think-obs-content">{{ stepObservation(s) }}</div>
                    </div>

                    <div v-if="stepOutput(s)" class="think-field">
                      <div class="think-field-label">LLM Output</div>
                      <div class="think-obs-content">{{ stepOutput(s) }}</div>
                    </div>

                    <div v-if="stepError(s)" class="think-field">
                      <div class="think-field-label danger">Error</div>
                      <div class="think-error">{{ stepError(s) }}</div>
                    </div>

                    <details class="think-raw">
                      <summary>查看原始 step JSON</summary>
                      <pre class="think-code">{{ formatValue(s) }}</pre>
                    </details>
                  </div>
                </details>
              </div>
            </details>
          </div>

          <div v-if="msg.role === 'assistant'" class="msg-html" v-html="renderMarkdown(msg.content)"></div>
          <div v-else class="msg-text">{{ msg.content }}</div>

          <div class="msg-meta">
            <span class="trace" v-if="msg.trace_id">trace {{ msg.trace_id.slice(0, 8) }}</span>
            <button class="btn-icon" @click="copyText(msg.content)" title="复制">Copy</button>
          </div>
        </div>
      </div>

      <div v-if="streaming" class="msg-row assistant">
        <div class="msg-avatar">AI</div>
        <div class="msg-bubble"><span class="cursor">Agent 思考中...</span></div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-row">
        <select class="mode-select" v-model="mode">
          <option value="react">ReAct</option>
          <option value="plan_execute">Plan & Execute</option>
        </select>
        <textarea
          ref="inputEl"
          class="chat-input"
          v-model="input"
          placeholder="输入问题，Enter 发送..."
          rows="1"
          @keydown.enter.exact.prevent="send"
          @input="autoResize"
        ></textarea>
        <button class="btn-send" @click="send" :disabled="!input.trim() || streaming">发送</button>
      </div>
      <div class="input-hint">
        <span>{{ input.length }} / 4000</span>
        <span>{{ mode === "react" ? "ReAct 推理循环" : "Plan & Execute" }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref } from "vue";
import { marked } from "marked";
import { chat, saveConversation, type AgentStep } from "../api";

interface Msg {
  role: "user" | "assistant";
  content: string;
  trace_id?: string;
  steps?: AgentStep[];
}

const mode = ref<"react" | "plan_execute">("react");
const input = ref("");
const messages = ref<Msg[]>([]);
const streaming = ref(false);
const inputEl = ref<HTMLTextAreaElement | null>(null);
const msgContainer = ref<HTMLDivElement | null>(null);
const sessionId = ref(crypto.randomUUID());

const quickQuestions = ["1+1等于几？", "帮我查询知识库", "解释一下当前 Agent 步骤"];

async function send() {
  const text = input.value.trim();
  if (!text || streaming.value) return;

  messages.value.push({ role: "user", content: text });
  input.value = "";
  autoResize();
  scrollBottom();
  streaming.value = true;

  try {
    const resp = await chat({
      messages: messages.value.map((m) => ({ role: m.role, content: m.content })),
      temperature: 0.7,
      conversation_id: sessionId.value,
      mode: mode.value,
    });
    messages.value.push({
      role: "assistant",
      content: resp.content,
      trace_id: resp.trace_id || "",
      steps: resp.steps || [],
    });
  } catch (e: unknown) {
    messages.value.push({ role: "assistant", content: `请求失败: ${(e as Error).message}` });
  } finally {
    streaming.value = false;
    scrollBottom();
  }

  try {
    await saveConversation({
      id: sessionId.value,
      title: text.slice(0, 50),
      messages: messages.value
        .filter((m) => !m.content.includes("请求失败"))
        .map((m) => ({ role: m.role, content: m.content })),
      mode: mode.value,
    });
  } catch {
    // 历史保存失败不影响本轮对话。
  }
}

function sendQuick(q: string) {
  input.value = q;
  send();
}

function copyText(t: string) {
  navigator.clipboard.writeText(t);
}

function renderMarkdown(t: string) {
  return marked.parse(t, { breaks: true }) as string;
}

function autoResize() {
  nextTick(() => {
    if (!inputEl.value) return;
    inputEl.value.style.height = "auto";
    inputEl.value.style.height = `${Math.min(inputEl.value.scrollHeight, 150)}px`;
  });
}

function scrollBottom() {
  nextTick(() => {
    if (msgContainer.value) msgContainer.value.scrollTop = msgContainer.value.scrollHeight;
  });
}

function stepPhase(step: AgentStep) {
  return String(step.phase || step.action_type || (step.final ? "final" : "reasoning"));
}

function stepStatus(step: AgentStep) {
  if (stepError(step)) return "error";
  if (step.final) return "final";
  if (step.status) return String(step.status);
  if (step.observation || step.llm_output || step.record) return "done";
  return "";
}

function stepStatusClass(step: AgentStep) {
  const status = stepStatus(step);
  return {
    ok: status === "ok" || status === "done" || status === "final",
    danger: status === "error",
  };
}

function stepTitle(step: AgentStep, index: number) {
  const record = asRecord(step.record);
  const parsed = asRecord(step.parsed);
  return String(
    step.title ||
      record?.title ||
      step.action ||
      parsed?.action ||
      step.subtask_id ||
      `Step ${index + 1}`,
  );
}

function stepThought(step: AgentStep) {
  const parsed = asRecord(step.parsed);
  return stringOrEmpty(parsed?.thought);
}

function stepAction(step: AgentStep) {
  const record = asRecord(step.record);
  const parsed = asRecord(step.parsed);
  return stringOrEmpty(step.action || record?.tool_name || step.tool_name || parsed?.action || step.action_type);
}

function stepToolArgs(step: AgentStep) {
  const record = asRecord(step.record);
  const parsed = asRecord(step.parsed);
  return step.action_input ?? record?.action_input ?? record?.tool_args_hint ?? parsed?.action_input ?? "";
}

function stepObservation(step: AgentStep) {
  const record = asRecord(step.record);
  return stringOrEmpty(step.observation || record?.observation);
}

function stepOutput(step: AgentStep) {
  const record = asRecord(step.record);
  const parsed = asRecord(step.parsed);
  return stringOrEmpty(step.llm_output || record?.llm_output || step.raw_llm || parsed?.final_answer);
}

function stepError(step: AgentStep) {
  const record = asRecord(step.record);
  return stringOrEmpty(step.error || record?.error);
}

function formatValue(value: unknown) {
  if (!hasValue(value)) return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function hasValue(value: unknown) {
  if (value === undefined || value === null || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function stringOrEmpty(value: unknown) {
  return typeof value === "string" ? value : value == null ? "" : String(value);
}
</script>
