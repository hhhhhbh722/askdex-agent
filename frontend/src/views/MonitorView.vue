<template>
  <div class="page">
    <header class="page-header">
      <h2>📊 系统监控</h2>
      <p>服务状态、Token 用量与调用链路追踪。</p>
    </header>

    <!-- 状态卡片 -->
    <div class="status-grid">
      <div class="status-card" v-for="s in services" :key="s.name">
        <div class="status-indicator" :class="s.status"></div>
        <div class="status-info">
          <span class="status-name">{{ s.name }}</span>
          <span class="status-text">{{ s.status === 'ok' ? '正常' : s.status === 'checking' ? '检测中...' : '离线' }}</span>
        </div>
        <span class="status-icon">{{ s.icon }}</span>
      </div>
    </div>

    <div class="monitor-grid">
      <!-- Token 用量 -->
      <section class="card">
        <h3>💳 Token 用量</h3>
        <div class="stat-big">{{ totalTokens.toLocaleString() }}</div>
        <p class="hint">今日消耗</p>
        <div class="stat-row">
          <div><span class="label">Prompt</span><span>{{ promptTokens.toLocaleString() }}</span></div>
          <div><span class="label">Completion</span><span>{{ completionTokens.toLocaleString() }}</span></div>
          <div><span class="label">估算费用</span><span>${{ costEstimate }}</span></div>
        </div>
        <div class="token-bar">
          <div class="token-prompt" :style="{ width: promptPct + '%' }"></div>
          <div class="token-completion" :style="{ width: completionPct + '%' }"></div>
        </div>
      </section>

      <!-- 请求统计 -->
      <section class="card">
        <h3>📈 请求统计</h3>
        <div class="stat-row">
          <div><span class="label">总请求</span><span>{{ stats.totalRequests }}</span></div>
          <div><span class="label">成功率</span><span class="green">{{ stats.successRate }}%</span></div>
          <div><span class="label">平均延迟</span><span>{{ stats.avgLatency }}ms</span></div>
        </div>
        <div class="stat-row" style="margin-top:12px">
          <div><span class="label">熔断器</span><span :class="breakerOpen ? 'red' : 'green'">{{ breakerOpen ? 'OPEN' : 'CLOSED' }}</span></div>
          <div><span class="label">模型</span><span>{{ config.model }}</span></div>
          <div><span class="label">会话数</span><span>{{ sessionCount }}</span></div>
        </div>
      </section>

      <!-- 调用链路 -->
      <section class="card card-wide">
        <h3>🔗 最近调用链路</h3>
        <table class="data-table" v-if="traces.length">
          <thead>
            <tr><th>Trace ID</th><th>操作</th><th>耗时</th><th>状态</th><th>详情</th></tr>
          </thead>
          <tbody>
            <tr v-for="t in traces" :key="t.id">
              <td><code>{{ t.id }}</code></td>
              <td>{{ t.operation }}</td>
              <td>{{ t.duration }}ms</td>
              <td><span :class="['badge', t.error ? 'error' : 'ready']">{{ t.error ? '失败' : '成功' }}</span></td>
              <td><button class="btn-text" @click="openTrace(t)">查看</button></td>
            </tr>
          </tbody>
        </table>
        <div v-else class="empty-state"><span>📭 暂无调用记录</span></div>
      </section>
    </div>

    <!-- Trace 详情弹窗 -->
    <div v-if="traceOpen" class="modal-overlay" @click.self="traceOpen = false">
      <div class="modal modal-wide">
        <h3>🔗 Trace: {{ selectedTrace?.id }}</h3>
        <div class="trace-tree" v-if="selectedTrace">
          <div class="trace-node root">
            <span class="trace-op">📌 {{ selectedTrace.operation }}</span>
            <span class="trace-dur">{{ selectedTrace.duration }}ms</span>
            <span :class="['badge', selectedTrace.error ? 'error' : 'ready']">{{ selectedTrace.error ? '失败' : '成功' }}</span>
          </div>
          <div class="trace-node child" v-for="s in selectedTrace.spans" :key="s.id">
            <span class="trace-op">↳ {{ s.operation }}</span>
            <span class="trace-dur">{{ s.duration }}ms</span>
          </div>
        </div>
        <button class="btn-text" @click="traceOpen = false" style="margin-top:16px">关闭</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from "vue";
import { healthFull } from "../api";
import type { FullHealthStatus } from "../api";

interface TraceSpan { id: string; operation: string; duration: number }
interface Trace { id: string; operation: string; duration: number; error?: string; spans: TraceSpan[] }

const services = ref([
  { name: "API 服务", status: "checking", icon: "🌐" },
  { name: "数据库", status: "checking", icon: "🗄️" },
  { name: "Redis", status: "checking", icon: "⚡" },
  { name: "Milvus", status: "checking", icon: "🔢" },
]);

const totalTokens = ref(0);
const promptTokens = ref(0);
const completionTokens = ref(0);
const costEstimate = ref("0.00");
const stats = ref({ totalRequests: 0, successRate: 100, avgLatency: 0 });
const breakerOpen = ref(false);
const sessionCount = ref(0);
const traces = ref<Trace[]>([]);
const traceOpen = ref(false);
const selectedTrace = ref<Trace | null>(null);
const config = ref({ model: "deepseek-chat" });

const promptPct = computed(() => totalTokens.value ? (promptTokens.value / totalTokens.value) * 100 : 0);
const completionPct = computed(() => totalTokens.value ? (completionTokens.value / totalTokens.value) * 100 : 0);

onMounted(async () => {
  // 调用完整健康检查
  try {
    const h = await healthFull();
    services.value[0].status = h.api === "up" ? "ok" : "down";
    services.value[1].status = h.database === "up" ? "ok" : "down";
    services.value[2].status = h.redis === "up" ? "ok" : "down";
    services.value[3].status = h.milvus === "up" ? "ok" : "down";
  } catch {
    services.value.forEach(s => s.status = "down");
  }

  // 从后端拉取真实统计
  try {
    const r = await fetch("/api/v1/metrics");
    const m = await r.json();
    totalTokens.value = m.total_tokens || 0;
    promptTokens.value = m.prompt_tokens || 0;
    completionTokens.value = m.completion_tokens || 0;
    stats.value.totalRequests = m.total_requests || 0;
    sessionCount.value = 0;
    // DeepSeek v4-pro 定价估算
    costEstimate.value = ((promptTokens.value / 1_000_000 * 0.27) + (completionTokens.value / 1_000_000 * 1.10)).toFixed(4);
  } catch { /* ignore */ }
});

function openTrace(t: Trace) {
  selectedTrace.value = t;
  traceOpen.value = true;
}
</script>
