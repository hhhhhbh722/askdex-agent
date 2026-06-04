<template>
  <div class="page">
    <header class="page-header">
      <h2>⚙️ Agent 配置</h2>
      <p>调整模型、工具与编排策略参数。</p>
    </header>

    <div class="settings-grid">
      <!-- 模型配置 -->
      <section class="card">
        <h3>🧠 模型配置</h3>
        <div class="form-group">
          <label>模型 ID</label>
          <input class="form-input" v-model="config.model" placeholder="deepseek-chat" />
        </div>
        <div class="form-group">
          <label>API Base URL</label>
          <input class="form-input" v-model="config.apiBase" placeholder="https://api.deepseek.com/v1" />
        </div>
        <div class="form-group">
          <label>API Key</label>
          <input class="form-input" type="password" v-model="config.apiKey" placeholder="sk-..." />
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Temperature ({{ config.temperature }})</label>
            <input class="form-input" type="range" min="0" max="2" step="0.1" v-model.number="config.temperature" />
          </div>
          <div class="form-group">
            <label>Max Tokens</label>
            <input class="form-input" type="number" v-model.number="config.maxTokens" min="1" max="32000" />
          </div>
        </div>
      </section>

      <!-- 工具管理 -->
      <section class="card">
        <h3>🔧 工具开关</h3>
        <div class="toggle-list">
          <div class="toggle-item" v-for="tool in tools" :key="tool.name">
            <div class="toggle-info">
              <span class="toggle-name">{{ tool.icon }} {{ tool.name }}</span>
              <span class="toggle-desc">{{ tool.desc }}</span>
            </div>
            <label class="switch">
              <input type="checkbox" v-model="tool.enabled" />
              <span class="slider"></span>
            </label>
          </div>
        </div>
      </section>

      <!-- 编排策略 -->
      <section class="card">
        <h3>📋 编排策略</h3>
        <div class="form-group">
          <label>默认模式</label>
          <select class="form-input" v-model="config.defaultMode">
            <option value="react">ReAct (推理+行动循环)</option>
            <option value="plan_execute">Plan & Execute (先规划再执行)</option>
          </select>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>ReAct 最大步数</label>
            <input class="form-input" type="number" v-model.number="config.reactMaxSteps" min="1" max="30" />
          </div>
          <div class="form-group">
            <label>Plan 重试次数</label>
            <input class="form-input" type="number" v-model.number="config.maxReplan" min="0" max="10" />
          </div>
        </div>
        <div class="form-group">
          <label>反思最低质量分 ({{ config.reflectionMinQuality }})</label>
          <input class="form-input" type="range" min="0" max="100" step="5" v-model.number="config.reflectionMinQuality" />
        </div>
        <div class="toggle-item">
          <div class="toggle-info">
            <span class="toggle-name">Plan 失败自动回退 ReAct</span>
          </div>
          <label class="switch">
            <input type="checkbox" v-model="config.fallbackReact" />
            <span class="slider"></span>
          </label>
        </div>
        <div class="toggle-item">
          <div class="toggle-info">
            <span class="toggle-name">启用反思 (Reflection)</span>
          </div>
          <label class="switch">
            <input type="checkbox" v-model="config.enableReflection" />
            <span class="slider"></span>
          </label>
        </div>
      </section>

      <!-- System Prompt -->
      <section class="card">
        <h3>📝 System Prompt</h3>
        <textarea class="form-textarea" v-model="config.systemPrompt" rows="6"
          placeholder="输入默认 System Prompt..."></textarea>
        <div class="card-actions">
          <button class="btn-primary" @click="saveConfig">💾 保存配置</button>
          <button class="btn-text" @click="resetConfig">↩ 恢复默认</button>
        </div>
        <p class="hint" v-if="saved">✅ 配置已保存（本地存储）</p>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from "vue";

const saved = ref(false);

const defaults = {
  model: "deepseek-chat",
  apiBase: "https://api.deepseek.com/v1",
  apiKey: "",
  temperature: 0.7,
  maxTokens: 4096,
  defaultMode: "react",
  reactMaxSteps: 10,
  maxReplan: 2,
  reflectionMinQuality: 60,
  fallbackReact: true,
  enableReflection: true,
  systemPrompt: "你是企业级 AI Agent，请使用中文回答。可用工具: calculator(计算)、search(搜索)、database(数据库查询)。",
};

const config = reactive({ ...defaults });

const tools = ref([
  { name: "calculator", icon: "🔢", desc: "安全数学表达式求值", enabled: true },
  { name: "search", icon: "🔍", desc: "DuckDuckGo 网页搜索", enabled: true },
  { name: "database", icon: "🗄️", desc: "SQL 数据库查询 (仅 SELECT)", enabled: true },
]);

function saveConfig() {
  localStorage.setItem("agent-config", JSON.stringify(config));
  saved.value = true;
  setTimeout(() => (saved.value = false), 2000);
}

function resetConfig() {
  Object.assign(config, defaults);
  localStorage.removeItem("agent-config");
}

// 页面加载时恢复
try {
  const savedConfig = localStorage.getItem("agent-config");
  if (savedConfig) Object.assign(config, JSON.parse(savedConfig));
} catch { /* ignore */ }
</script>
