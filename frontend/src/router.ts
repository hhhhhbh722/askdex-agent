import { createRouter, createWebHashHistory } from "vue-router";
import ChatView from "./views/ChatView.vue";
import KnowledgeView from "./views/KnowledgeView.vue";
import SettingsView from "./views/SettingsView.vue";
import HistoryView from "./views/HistoryView.vue";
import MonitorView from "./views/MonitorView.vue";

const routes = [
  { path: "/", name: "chat", component: ChatView, meta: { title: "对话工作台", icon: "💬" } },
  { path: "/knowledge", name: "knowledge", component: KnowledgeView, meta: { title: "知识库管理", icon: "📄" } },
  { path: "/settings", name: "settings", component: SettingsView, meta: { title: "Agent 配置", icon: "⚙️" } },
  { path: "/history", name: "history", component: HistoryView, meta: { title: "对话历史", icon: "🕐" } },
  { path: "/monitor", name: "monitor", component: MonitorView, meta: { title: "系统监控", icon: "📊" } },
];

export default createRouter({
  history: createWebHashHistory(),
  routes,
});

export { routes };
