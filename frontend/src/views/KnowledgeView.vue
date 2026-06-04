<template>
  <div class="page">
    <header class="page-header">
      <h2>📄 知识库管理</h2>
      <p>上传文档构建知识库，Agent 将基于文档内容回答。</p>
    </header>

    <!-- 上传区 -->
    <section class="card">
      <h3>上传文档</h3>
      <label class="drop-zone" :class="{ dragging }"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="handleDrop">
        <input type="file" hidden ref="fileInput" multiple
          accept=".txt,.pdf,.md,.csv,.json,.xml,.html"
          @change="handleFiles" />
        <div class="drop-hint">
          <span class="drop-icon">📤</span>
          <span v-if="uploading">⏳ 正在上传解析中...</span>
          <span v-else>拖拽文件到此处，或点击选择文件</span>
          <small>支持 TXT / PDF / Markdown / CSV / JSON</small>
        </div>
      </label>
      <div v-if="uploadProgress.length" class="progress-list">
        <div v-for="p in uploadProgress" :key="p.name" class="progress-item">
          <span>{{ p.name }}</span>
          <span>{{ p.status === 'ok' ? '✅' : p.status === 'fail' ? '❌' : '⏳' }}</span>
        </div>
      </div>
    </section>

    <!-- 文档列表 -->
    <section class="card">
      <div class="card-head">
        <h3>已入库文档 ({{ documents.length }})</h3>
        <button class="btn-text" @click="refresh">🔄 刷新</button>
      </div>
      <table class="data-table" v-if="documents.length">
        <thead>
          <tr>
            <th>文件名</th>
            <th>类型</th>
            <th>分区</th>
            <th>分块数</th>
            <th>上传时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="d in documents" :key="d.id">
            <td class="filename">{{ d.filename }}</td>
            <td><span class="tag">{{ (d as any).mime_type || d.mime_type || '-' }}</span></td>
            <td>
              <select class="form-input" style="width:auto;padding:2px 6px;font-size:11px"
                :value="getTag(d.id)" @change="setTag(d.id, ($event.target as HTMLSelectElement).value)">
                <option value="">未分区</option>
                <option value="技术文档">技术文档</option>
                <option value="产品需求">产品需求</option>
                <option value="会议纪要">会议纪要</option>
                <option value="合同协议">合同协议</option>
                <option value="其他">其他</option>
              </select>
            </td>
            <td>{{ (d as any).chunk_count || '-' }}</td>
            <td>{{ (d as any).created_at ? new Date((d as any).created_at).toLocaleString() : '-' }}</td>
            <td>
              <button class="btn-text" @click="confirmDelete(d.id, d.filename)">🗑 删除</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">
        <span>📭 暂无文档，上传第一个文档开始构建知识库</span>
      </div>
    </section>

    <!-- RAG 检索测试 -->
    <section class="card">
      <h3>🔍 RAG 检索测试</h3>
      <div class="input-row" style="margin-bottom:12px">
        <input class="form-input" v-model="retrieveQ" placeholder="输入查询，测试知识库检索效果..." @keydown.enter="doRetrieve" />
        <button class="btn-primary" @click="doRetrieve" :disabled="!retrieveQ.trim()">搜索</button>
      </div>
      <div v-if="retrieveResults.length" class="retrieve-results">
        <div v-for="(r, i) in retrieveResults" :key="i" class="retrieve-item">
          <div class="retrieve-score">#{{ i+1 }} 相关度: {{ (r.score*100).toFixed(1) }}%</div>
          <div class="retrieve-text">{{ r.content?.slice(0, 400) || '(内容未加载)' }}</div>
        </div>
      </div>
      <div v-else-if="retrieveDone" class="empty-state"><span>📭 未检索到相关内容</span></div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from "vue";
import { uploadDocument, listDocuments, deleteDocument, type DocumentInfo } from "../api";

const documents = ref<DocumentInfo[]>([]);
const uploading = ref(false);
const dragging = ref(false);
const uploadProgress = ref<{ name: string; status: string }[]>([]);
const fileInput = ref<HTMLInputElement | null>(null);
const tags = ref<Record<string, string>>({});
const retrieveQ = ref("");
const retrieveResults = ref<any[]>([]);
const retrieveDone = ref(false);

onMounted(async () => {
  await refresh();
  const saved = localStorage.getItem("kb-tags");
  if (saved) tags.value = JSON.parse(saved);
});

async function refresh() {
  try { documents.value = await listDocuments(); } catch { /* offline */ }
}

async function uploadFiles(files: FileList) {
  uploading.value = true;
  uploadProgress.value = [];
  for (const f of Array.from(files)) {
    uploadProgress.value.push({ name: f.name, status: "pending" });
    try { await uploadDocument(f); uploadProgress.value.find(p => p.name === f.name)!.status = "ok"; }
    catch { uploadProgress.value.find(p => p.name === f.name)!.status = "fail"; }
  }
  uploading.value = false;
  await refresh();
}

function handleFiles(e: Event) {
  const files = (e.target as HTMLInputElement).files;
  if (files?.length) uploadFiles(files);
}

function handleDrop(e: DragEvent) {
  dragging.value = false;
  if (e.dataTransfer?.files?.length) uploadFiles(e.dataTransfer.files);
}

function getTag(docId: string): string { return tags.value[docId] || ""; }

function setTag(docId: string, value: string) {
  tags.value[docId] = value;
  localStorage.setItem("kb-tags", JSON.stringify(tags.value));
}

async function confirmDelete(id: string, filename: string) {
  if (!confirm(`确定删除「${filename}」及其所有分块数据？`)) return;
  try {
    await deleteDocument(id);
    documents.value = documents.value.filter(d => d.id !== id);
  } catch { alert("删除失败，请检查后端是否运行"); }
}

async function doRetrieve() {
  if (!retrieveQ.value.trim()) return;
  try {
    const resp = await fetch(`/api/v1/retrieve?q=${encodeURIComponent(retrieveQ.value)}&top_k=5`);
    const data = await resp.json();
    retrieveResults.value = data.results || [];
    retrieveDone.value = true;
  } catch { retrieveResults.value = []; }
}
</script>
