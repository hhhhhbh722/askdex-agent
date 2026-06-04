<template>
  <div class="chat-layout">
    <div class="messages" ref="msgContainer">
      <div v-if="messages.length === 0" class="welcome">
        <div class="welcome-icon">🤖</div>
        <h2>AI Agent 对话工作台</h2>
        <p>Agent 会自动调用知识库、计算器等工具回答你的问题。</p>
        <div class="quick-actions">
          <button v-for="q in quickQuestions" :key="q" @click="sendQuick(q)">{{ q }}</button>
        </div>
      </div>

      <div v-for="(msg, i) in messages" :key="i" class="msg-row" :class="msg.role">
        <div class="msg-avatar">{{ msg.role === 'user' ? '👤' : '🤖' }}</div>
        <div class="msg-bubble">
          <!-- Agent 思考过程 -->
          <div v-if="msg.steps?.length" class="think-box">
            <details class="think-detail" open>
              <summary>🧠 Agent 思考过程（{{ msg.steps.length }} 步）</summary>
              <div class="think-steps">
                <div v-for="(s, si) in msg.steps" :key="si" class="think-step-card">
                  <div class="think-step-header">
                    <span class="think-step-num">Step {{ si + 1 }}</span>
                    <span class="think-phase-badge">{{ s.phase || 'reasoning' }}</span>
                    <span v-if="s.action" class="think-action-badge">{{ s.action }}</span>
                  </div>
                  <div v-if="s.action_input" class="think-input">
                    <code>{{ JSON.stringify(s.action_input) }}</code>
                  </div>
                  <div v-if="s.observation && s.observation.length > 10" class="think-obs">
                    <div class="think-obs-label">📋 检索结果：</div>
                    <div class="think-obs-content">{{ s.observation }}</div>
                  </div>
                  <div v-else-if="s.observation" class="think-obs-empty">
                    ⚠️ 工具返回为空
                  </div>
                </div>
              </div>
            </details>
          </div>

          <div v-if="msg.role === 'assistant'" class="msg-html" v-html="renderMarkdown(msg.content)"></div>
          <div v-else class="msg-text">{{ msg.content }}</div>

          <div class="msg-meta">
            <span class="trace" v-if="msg.trace_id">🔗 {{ msg.trace_id.slice(0, 8) }}</span>
            <button class="btn-icon" @click="copyText(msg.content)">📋</button>
          </div>
        </div>
      </div>

      <div v-if="streaming" class="msg-row assistant">
        <div class="msg-avatar">🤖</div>
        <div class="msg-bubble"><span class="cursor">🤔 Agent 思考中...</span></div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-row">
        <select class="mode-select" v-model="mode">
          <option value="react">🧠 ReAct</option>
          <option value="plan_execute">📋 Plan & Execute</option>
        </select>
        <textarea ref="inputEl" class="chat-input" v-model="input"
          placeholder="输入问题，Enter 发送..." rows="1"
          @keydown.enter.exact.prevent="send" @input="autoResize"></textarea>
        <button class="btn-send" @click="send" :disabled="!input.trim() || streaming">➤</button>
      </div>
      <div class="input-hint">
        <span>{{ input.length }} / 4000</span>
        <span>{{ mode === 'react' ? 'ReAct 推理循环' : 'Plan & Execute' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick } from "vue";
import { marked } from "marked";
import { chat, saveConversation } from "../api";

interface Msg { role: "user" | "assistant"; content: string; trace_id?: string; steps?: any[] }

const mode = ref<"react" | "plan_execute">("react");
const input = ref("");
const messages = ref<Msg[]>([]);
const streaming = ref(false);
const inputEl = ref<HTMLTextAreaElement | null>(null);
const msgContainer = ref<HTMLDivElement | null>(null);
const sessionId = ref(crypto.randomUUID());

const quickQuestions = ["1+1等于几？", "小夜的蛋多重？", "帮我搜索AI最新进展"];

async function send() {
  const text = input.value.trim();
  if (!text || streaming.value) return;
  messages.value.push({ role: "user", content: text });
  input.value = ""; autoResize(); scrollBottom();
  streaming.value = true;

  try {
    const resp = await chat({ messages: [{ role: "user", content: text }], temperature: 0.7 });
    messages.value.push({
      role: "assistant", content: resp.content,
      trace_id: resp.trace_id || "", steps: (resp as any).steps || [],
    });
  } catch (e: unknown) {
    messages.value.push({ role: "assistant", content: `请求失败: ${(e as Error).message}` });
  }
  streaming.value = false; scrollBottom();

  try {
    await saveConversation({ id: sessionId.value, title: text.slice(0, 50),
      messages: messages.value.filter(m => !m.content.includes("请求失败")).map(m => ({ role: m.role, content: m.content })),
      mode: mode.value });
  } catch { /* ignore */ }
}

function sendQuick(q: string) { input.value = q; send(); }
function copyText(t: string) { navigator.clipboard.writeText(t); }
function renderMarkdown(t: string) { return marked.parse(t, { breaks: true }) as string; }
function autoResize() { nextTick(() => { if (inputEl.value) { inputEl.value.style.height = "auto"; inputEl.value.style.height = Math.min(inputEl.value.scrollHeight, 150) + "px"; } }); }
function scrollBottom() { nextTick(() => { if (msgContainer.value) msgContainer.value.scrollTop = msgContainer.value.scrollHeight; }); }
</script>
