<template>
  <div class="page">
    <header class="page-header">
      <h2>📄 知识库管理</h2>
      <p>批量上传文档并按二级分组管理，Agent 将基于入库内容回答。</p>
    </header>

    <section class="card">
      <div class="card-head">
        <h3>批量上传</h3>
        <span class="hint">一级/二级分组会写入后端，刷新后仍然保留</span>
      </div>

      <div class="form-row upload-controls">
        <div class="form-group">
          <label>一级分组</label>
          <select class="form-input" v-model="uploadParentGroup" @change="uploadChildGroup = ''">
            <option value="">未分组</option>
            <option v-for="g in parentGroups" :key="g" :value="g">{{ g }}</option>
          </select>
        </div>
        <div class="form-group">
          <label>二级分组</label>
          <select class="form-input" v-model="uploadChildGroup" :disabled="!uploadParentGroup">
            <option value="">无二级分组</option>
            <option v-for="g in childOptions(uploadParentGroup)" :key="g" :value="g">{{ g }}</option>
          </select>
        </div>
      </div>

      <div class="form-row upload-controls">
        <div class="form-group">
          <label>新建一级分组</label>
          <div class="input-row compact-row">
            <input class="form-input" v-model="newParentGroup" placeholder="例如：课程资料" @keydown.enter.prevent="addParentGroup" />
            <button class="btn-primary" type="button" @click="addParentGroup" :disabled="!newParentGroup.trim()">添加</button>
          </div>
        </div>
        <div class="form-group">
          <label>新建二级分组</label>
          <div class="input-row compact-row">
            <input class="form-input" v-model="newChildGroup" placeholder="例如：第 5 章 Spring Boot" @keydown.enter.prevent="addChildGroup" />
            <button class="btn-primary" type="button" @click="addChildGroup" :disabled="!uploadParentGroup || !newChildGroup.trim()">添加</button>
          </div>
        </div>
      </div>

      <label class="drop-zone" :class="{ dragging }"
        @dragover.prevent="dragging = true"
        @dragleave.prevent="dragging = false"
        @drop.prevent="handleDrop">
        <input type="file" hidden ref="fileInput" multiple
          accept=".txt,.pdf,.md,.csv,.json,.xml,.html"
          @change="handleFiles" />
        <div class="drop-hint">
          <span class="drop-icon">📤</span>
          <span v-if="uploading">正在上传解析 {{ uploadProgress.filter(p => p.status === 'ok').length }} / {{ uploadProgress.length }}...</span>
          <span v-else>拖拽多个文件到此处，或点击选择文件</span>
          <small>支持 TXT / PDF / Markdown / CSV / JSON / XML / HTML</small>
        </div>
      </label>

      <div v-if="uploadProgress.length" class="progress-list">
        <div v-for="p in uploadProgress" :key="p.name" class="progress-item">
          <span class="doc-name">{{ p.name }}</span>
          <span class="tag">{{ p.group || "未分组" }}</span>
          <span :class="['badge', p.status === 'ok' ? 'ready' : p.status === 'fail' ? 'error' : 'pending']">
            {{ p.status === 'ok' ? '成功' : p.status === 'fail' ? '失败' : p.status === 'running' ? '处理中' : '等待中' }}
          </span>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-head">
        <h3>已入库文档 ({{ filteredDocuments.length }} / {{ documents.length }})</h3>
        <div class="toolbar-actions">
          <select class="form-input compact-select" v-model="activeParentGroup" @change="activeChildGroup = ''">
            <option value="">全部一级</option>
            <option value="__ungrouped">未分组</option>
            <option v-for="g in parentGroups" :key="g" :value="g">{{ g }}</option>
          </select>
          <select class="form-input compact-select" v-model="activeChildGroup" :disabled="!activeParentGroup || activeParentGroup === '__ungrouped'">
            <option value="">全部二级</option>
            <option v-for="g in childOptions(activeParentGroup)" :key="g" :value="g">{{ g }}</option>
          </select>
          <button class="btn-text" @click="refresh">🔄 刷新</button>
        </div>
      </div>

      <table class="data-table" v-if="filteredDocuments.length">
        <thead>
          <tr>
            <th>文件名</th>
            <th>类型</th>
            <th>一级分组</th>
            <th>二级分组</th>
            <th>分块数</th>
            <th>上传时间</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="d in filteredDocuments" :key="d.id">
            <td class="filename">{{ d.filename }}</td>
            <td><span class="tag">{{ d.mime_type || '-' }}</span></td>
            <td>
              <select class="form-input compact-select"
                :value="d.parent_group || ''"
                @change="changeGroup(d, ($event.target as HTMLSelectElement).value, '')">
                <option value="">未分组</option>
                <option v-for="g in parentGroups" :key="g" :value="g">{{ g }}</option>
              </select>
            </td>
            <td>
              <select class="form-input compact-select"
                :value="d.child_group || ''"
                :disabled="!d.parent_group"
                @change="changeGroup(d, d.parent_group, ($event.target as HTMLSelectElement).value)">
                <option value="">无二级</option>
                <option v-for="g in childOptions(d.parent_group)" :key="g" :value="g">{{ g }}</option>
              </select>
            </td>
            <td>{{ d.chunk_count || '-' }}</td>
            <td>{{ d.created_at ? new Date(d.created_at).toLocaleString() : '-' }}</td>
            <td>
              <button class="btn-text" @click="confirmDelete(d.id, d.filename)">🗑 删除</button>
            </td>
          </tr>
        </tbody>
      </table>
      <div v-else class="empty-state">
        <span>📭 {{ documents.length ? '当前分组暂无文档' : '暂无文档，上传第一个文档开始构建知识库' }}</span>
      </div>
    </section>

    <section class="card">
      <div class="card-head">
        <h3>🔍 RAG 检索测试</h3>
        <div class="toolbar-actions">
          <select class="form-input compact-select" v-model="retrieveParentGroup" @change="retrieveChildGroup = ''">
            <option value="">全部一级</option>
            <option v-for="g in parentGroups" :key="g" :value="g">{{ g }}</option>
          </select>
          <select class="form-input compact-select" v-model="retrieveChildGroup" :disabled="!retrieveParentGroup">
            <option value="">全部二级</option>
            <option v-for="g in childOptions(retrieveParentGroup)" :key="g" :value="g">{{ g }}</option>
          </select>
        </div>
      </div>
      <div class="input-row" style="margin-bottom:12px">
        <input class="form-input" v-model="retrieveQ" placeholder="输入查询，测试知识库检索效果..." @keydown.enter="doRetrieve" />
        <button class="btn-primary" @click="doRetrieve" :disabled="!retrieveQ.trim()">搜索</button>
      </div>
      <div v-if="retrieveResults.length" class="retrieve-results">
        <div v-for="(r, i) in retrieveResults" :key="i" class="retrieve-item">
          <div class="retrieve-score">#{{ i + 1 }} 相关度: {{ (r.score * 100).toFixed(1) }}%</div>
          <div class="retrieve-source">{{ r.source || "知识库" }}</div>
          <div class="retrieve-text">{{ r.content?.slice(0, 400) || '(内容未加载)' }}</div>
        </div>
      </div>
      <div v-else-if="retrieveDone" class="empty-state"><span>📭 未检索到相关内容</span></div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from "vue";
import {
  deleteDocument,
  getUploadJob,
  listDocuments,
  updateDocumentGroup,
  uploadDocuments,
  type DocumentInfo,
} from "../api";

type UploadStatus = "pending" | "running" | "ok" | "fail";

const documents = ref<DocumentInfo[]>([]);
const uploading = ref(false);
const dragging = ref(false);
const uploadProgress = ref<{ name: string; status: UploadStatus; group: string }[]>([]);
const fileInput = ref<HTMLInputElement | null>(null);
const customParents = ref<string[]>([]);
const customChildren = ref<Record<string, string[]>>({});
const uploadParentGroup = ref("");
const uploadChildGroup = ref("");
const newParentGroup = ref("");
const newChildGroup = ref("");
const activeParentGroup = ref("");
const activeChildGroup = ref("");
const retrieveParentGroup = ref("");
const retrieveChildGroup = ref("");
const retrieveQ = ref("");
const retrieveResults = ref<any[]>([]);
const retrieveDone = ref(false);

const parentGroups = computed(() => {
  const fromDocs = documents.value.map((d) => d.parent_group || d.group).filter(Boolean);
  return [...new Set([...customParents.value, ...fromDocs])].sort();
});

const childGroupsByParent = computed(() => {
  const tree: Record<string, Set<string>> = {};
  for (const parent of parentGroups.value) tree[parent] = new Set(customChildren.value[parent] || []);
  for (const doc of documents.value) {
    if (!doc.parent_group || !doc.child_group) continue;
    if (!tree[doc.parent_group]) tree[doc.parent_group] = new Set();
    tree[doc.parent_group].add(doc.child_group);
  }
  return Object.fromEntries(
    Object.entries(tree).map(([parent, children]) => [parent, [...children].sort()])
  );
});

const filteredDocuments = computed(() => {
  if (!activeParentGroup.value) return documents.value;
  if (activeParentGroup.value === "__ungrouped") return documents.value.filter((d) => !d.parent_group);
  return documents.value.filter((d) => {
    const parentOk = d.parent_group === activeParentGroup.value;
    const childOk = !activeChildGroup.value || d.child_group === activeChildGroup.value;
    return parentOk && childOk;
  });
});

onMounted(async () => {
  const savedParents = localStorage.getItem("kb-custom-parent-groups");
  const savedChildren = localStorage.getItem("kb-custom-child-groups");
  if (savedParents) customParents.value = JSON.parse(savedParents);
  if (savedChildren) customChildren.value = JSON.parse(savedChildren);
  await refresh();
});

async function refresh() {
  try {
    documents.value = (await listDocuments()).map(normalizeDocumentGroups);
  } catch {
    documents.value = [];
  }
}

function normalizeDocumentGroups(doc: DocumentInfo): DocumentInfo {
  if (doc.parent_group || !doc.group) return doc;
  const [parent, child = ""] = doc.group.split("/").map((part) => part.trim());
  return { ...doc, parent_group: parent, child_group: child };
}

function childOptions(parent: string): string[] {
  if (!parent || parent === "__ungrouped") return [];
  return childGroupsByParent.value[parent] || [];
}

function groupPath(parent: string, child: string): string {
  if (!parent) return "";
  return child ? `${parent} / ${child}` : parent;
}

function addParentGroup() {
  const parent = newParentGroup.value.trim();
  if (!parent) return;
  if (!customParents.value.includes(parent)) {
    customParents.value.push(parent);
    localStorage.setItem("kb-custom-parent-groups", JSON.stringify(customParents.value));
  }
  uploadParentGroup.value = parent;
  uploadChildGroup.value = "";
  newParentGroup.value = "";
}

function addChildGroup() {
  const parent = uploadParentGroup.value;
  const child = newChildGroup.value.trim();
  if (!parent || !child) return;
  const children = customChildren.value[parent] || [];
  if (!children.includes(child)) {
    customChildren.value = { ...customChildren.value, [parent]: [...children, child] };
    localStorage.setItem("kb-custom-child-groups", JSON.stringify(customChildren.value));
  }
  uploadChildGroup.value = child;
  newChildGroup.value = "";
}

async function uploadFiles(fileList: FileList) {
  const files = Array.from(fileList);
  if (!files.length || uploading.value) return;

  uploading.value = true;
  const parent = uploadParentGroup.value;
  const child = uploadChildGroup.value;
  const group = groupPath(parent, child);
  uploadProgress.value = files.map((file) => ({ name: file.name, status: "pending", group }));

  try {
    const resp = await uploadDocuments(files, group, parent, child);
    const jobId = resp.job_id || resp.id;
    if (!jobId) throw new Error("后端未返回上传任务 ID");
    await pollUploadJob(jobId);
  } catch {
    uploadProgress.value.forEach((p) => (p.status = "fail"));
  } finally {
    uploading.value = false;
    if (fileInput.value) fileInput.value.value = "";
    await refresh();
  }
}

async function pollUploadJob(jobId: string) {
  while (true) {
    const job = await getUploadJob(jobId);
    const completed = new Set<string>();

    for (const result of job.results || []) {
      const item = uploadProgress.value.find((p) => p.name === result.filename);
      if (!item) continue;
      item.status = result.status === "ready" ? "ok" : "fail";
      completed.add(result.filename);
    }

    for (const item of uploadProgress.value) {
      if (completed.has(item.name)) continue;
      item.status = job.current === item.name ? "running" : "pending";
    }

    if (["ready", "partial", "failed"].includes(job.status)) return;
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

function handleFiles(e: Event) {
  const files = (e.target as HTMLInputElement).files;
  if (files?.length) uploadFiles(files);
}

function handleDrop(e: DragEvent) {
  dragging.value = false;
  if (e.dataTransfer?.files?.length) uploadFiles(e.dataTransfer.files);
}

async function changeGroup(doc: DocumentInfo, parent: string, child: string) {
  const previous = { group: doc.group, parent: doc.parent_group, child: doc.child_group };
  doc.parent_group = parent;
  doc.child_group = parent ? child : "";
  doc.group = groupPath(doc.parent_group, doc.child_group);
  try {
    await updateDocumentGroup(doc.id, {
      group: doc.group,
      parent_group: doc.parent_group,
      child_group: doc.child_group,
    });
  } catch {
    doc.group = previous.group;
    doc.parent_group = previous.parent;
    doc.child_group = previous.child;
    alert("分组更新失败，请检查后端是否运行");
  }
}

async function confirmDelete(id: string, filename: string) {
  if (!confirm(`确定删除「${filename}」及其所有分块数据？`)) return;
  try {
    await deleteDocument(id);
    documents.value = documents.value.filter((d) => d.id !== id);
  } catch {
    alert("删除失败，请检查后端是否运行");
  }
}

async function doRetrieve() {
  if (!retrieveQ.value.trim()) return;
  try {
    const params = new URLSearchParams({ q: retrieveQ.value, top_k: "5" });
    if (retrieveParentGroup.value) params.set("parent_group", retrieveParentGroup.value);
    if (retrieveChildGroup.value) params.set("child_group", retrieveChildGroup.value);
    const resp = await fetch(`/api/v1/retrieve?${params.toString()}`);
    const data = await resp.json();
    retrieveResults.value = data.results || [];
    retrieveDone.value = true;
  } catch {
    retrieveResults.value = [];
    retrieveDone.value = true;
  }
}
</script>
