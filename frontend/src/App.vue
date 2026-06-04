<template>
  <div class="app-shell">
    <!-- 侧栏导航 -->
    <aside class="sidebar">
      <div class="sidebar-brand" @click="$router.push('/')" style="cursor:pointer">
        <svg viewBox="0 0 24 24" class="logo-icon" aria-hidden="true">
          <path d="M12 2a4 4 0 0 1 4 4v1h2a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h2V6a4 4 0 0 1 4-4Zm0 2a2 2 0 0 0-2 2v1h4V6a2 2 0 0 0-2-2Z" />
        </svg>
        <span>AI Agent</span>
      </div>

      <nav class="nav-menu">
        <router-link
          v-for="r in navRoutes"
          :key="r.name"
          :to="r.path"
          class="nav-item"
          active-class="nav-active"
        >
          <span class="nav-icon">{{ r.meta.icon }}</span>
          <span class="nav-label">{{ r.meta.title }}</span>
        </router-link>
      </nav>

      <div class="sidebar-footer">
        <router-link to="/monitor" class="status-bar" :class="backendStatus">
          <span class="dot"></span>
          {{ backendStatus === 'ok' ? '系统正常' : '后端离线' }}
        </router-link>
      </div>
    </aside>

    <!-- 主内容区 -->
    <main class="main-view">
      <router-view />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import { routes } from "./router";
import { healthCheck } from "./api";

const navRoutes = routes;
const backendStatus = ref<"ok" | "down">("down");

onMounted(async () => {
  try {
    const h = await healthCheck();
    backendStatus.value = h.status === "ok" ? "ok" : "down";
  } catch {
    backendStatus.value = "down";
  }
});
</script>
