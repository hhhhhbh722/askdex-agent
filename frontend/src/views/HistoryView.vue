<template>
  <div class="page">
    <header class="page-header">
      <h2>🕐 对话历史</h2>
      <p>已持久化到数据库，跨设备可查。</p>
    </header>

    <section class="card">
      <div class="card-head">
        <input class="form-input" style="max-width:400px" v-model="searchQuery" placeholder="🔍 搜索对话标题..." />
        <div>
          <button class="btn-text" @click="refresh">🔄 刷新</button>
          <span class="hint">共 {{ filteredSessions.length }} 个会话</span>
        </div>
      </div>

      <div class="session-list" v-if="filteredSessions.length">
        <div v-for="s in filteredSessions" :key="s.id" class="session-card">
          <div class="session-main" @click="openSession(s.id)">
            <h4>{{ s.title }}</h4>
            <p class="session-preview">{{ s.preview || '(空对话)' }}</p>
            <div class="session-meta">
              <span>{{ formatDate(s.created_at) }}</span>
              <span>{{ s.message_count }} 条消息</span>
              <span class="tag">{{ s.mode }}</span>
            </div>
          </div>
          <div class="session-actions">
            <button class="btn-text" @click="openSession(s.id)">📖 查看</button>
            <button class="btn-text" @click="confirmDelete(s.id, s.title)">🗑</button>
          </div>
        </div>
      </div>
      <div v-else class="empty-state">
        <span>📭 {{ searchQuery ? '未找到匹配' : '暂无对话历史，去聊天页开始对话吧' }}</span>
      </div>
    </section>

    <!-- 回放弹窗 -->
    <div v-if="replayOpen" class="modal-overlay" @click.self="replayOpen = false">
      <div class="modal modal-wide">
        <h3>📖 {{ replayTitle }}</h3>
        <div class="replay-msgs">
          <div v-for="(m, i) in replayMessages" :key="i" class="msg-row" :class="m.role">
            <div class="msg-avatar">{{ m.role === 'user' ? '👤' : '🤖' }}</div>
            <div class="msg-bubble">
              <div v-if="m.role === 'assistant'" class="msg-html" v-html="renderMarkdown(m.content)"></div>
              <div v-else class="msg-text">{{ m.content }}</div>
            </div>
          </div>
        </div>
        <button class="btn-text" @click="replayOpen = false" style="margin-top:16px">关闭</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { marked } from "marked";
import { listConversations, getConversation, deleteConversation, type ConversationSummary } from "../api";

const sessions = ref<ConversationSummary[]>([]);
const searchQuery = ref("");
const replayOpen = ref(false);
const replayTitle = ref("");
const replayMessages = ref<{ role: string; content: string }[]>([]);

const filteredSessions = computed(() => {
  if (!searchQuery.value) return sessions.value;
  const q = searchQuery.value.toLowerCase();
  return sessions.value.filter(s => s.title.toLowerCase().includes(q));
});

onMounted(refresh);

async function refresh() {
  try {
    sessions.value = await listConversations();
  } catch { /* backend may be down */ }
}

async function openSession(id: string) {
  try {
    const detail = await getConversation(id);
    replayTitle.value = detail.title;
    replayMessages.value = detail.messages;
    replayOpen.value = true;
  } catch { alert("加载对话失败"); }
}

async function confirmDelete(id: string, title: string) {
  if (!confirm(`确定删除「${title}」？此操作不可撤销。`)) return;
  try {
    await deleteConversation(id);
    sessions.value = sessions.value.filter(s => s.id !== id);
  } catch { alert("删除失败"); }
}

function formatDate(d: string | null): string {
  if (!d) return "-";
  return new Date(d).toLocaleString();
}

function renderMarkdown(t: string) { return marked.parse(t, { breaks: true }) as string; }
</script>
