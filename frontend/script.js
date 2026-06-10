const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: '',
            isLoading: false,
            activeNav: 'newChat',
            abortController: null,
            sessionId: 'session_' + Date.now(),
            sessions: [],
            showHistorySidebar: false,
            isComposing: false,
            documents: [],
            documentsLoading: false,
            docSearch: '',
            docPage: 1,
            docsPerPage: 10,
            selectedFile: null,
            selectedImage: null,
            queryImagePath: null,
            queryImageBase64: null,
            isUploading: false,
            uploadProgress: '',
            uploadSteps: [],
            uploadProgressCollapsed: false,
            activeUploadJobId: '',
            uploadPollTimer: null,
            deleteJobs: {},
            deletePollTimers: {},
            deleteRemoveTimers: {},
            token: localStorage.getItem('accessToken') || '',
            currentUser: null,
            authMode: 'login',
            authForm: {
                username: '',
                password: '',
                role: 'user',
                admin_code: ''
            },
            authLoading: false,
            theme: localStorage.getItem('theme') || 'dark'
        };
    },
    computed: {
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        },
        isAdmin() {
            return this.currentUser?.role === 'admin';
        },
        sortedDocuments() {
            return [...this.documents].sort((a, b) => a.filename.localeCompare(b.filename));
        },
        filteredDocuments() {
            if (!this.docSearch.trim()) return this.sortedDocuments;
            const q = this.docSearch.toLowerCase();
            return this.sortedDocuments.filter(d => d.filename.toLowerCase().includes(q));
        },
        paginatedDocuments() {
            const start = (this.docPage - 1) * this.docsPerPage;
            return this.filteredDocuments.slice(start, start + this.docsPerPage);
        },
        docTotalPages() {
            return Math.max(1, Math.ceil(this.filteredDocuments.length / this.docsPerPage));
        }
    },
    async mounted() {
        if (this.theme === 'light') document.body.classList.add('light');
        this.configureMarked();
        if (this.token) {
            try {
                await this.fetchMe();
            } catch (_) {
                this.handleLogout();
            }
        }
        this.$nextTick(() => {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.addEventListener('click', (e) => {
                    const citeRef = e.target.closest('.cite-ref');
                    if (!citeRef) return;
                    const msgIndex = citeRef.getAttribute('data-msg-index');
                    const chunkIndex = citeRef.getAttribute('data-chunk-index');
                    if (msgIndex != null && chunkIndex != null) {
                        this.scrollToChunk(Number(msgIndex), Number(chunkIndex));
                    }
                });
            }
        });
    },
    beforeUnmount() {
        this.stopUploadJobPolling();
        this.stopAllDeleteJobPolling();
        Object.values(this.deleteRemoveTimers).forEach(timer => clearTimeout(timer));
    },
    methods: {
        toggleTheme() {
            this.theme = this.theme === 'dark' ? 'light' : 'dark';
            localStorage.setItem('theme', this.theme);
            document.body.classList.toggle('light', this.theme === 'light');
        },
        configureMarked() {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: 'hljs language-',
                breaks: true,
                gfm: true
            });
        },

        parseMarkdown(text, msgIndex) {
            let html = marked.parse(text || '');
            let inCode = false;
            return html.split(/(<[^>]*>)/).map(part => {
                if (part.startsWith('<')) {
                    if (part.startsWith('<code') || part.startsWith('<pre')) inCode = true;
                    if (part.startsWith('</code') || part.startsWith('</pre')) inCode = false;
                    return part;
                }
                if (!inCode && msgIndex != null) {
                    return part.replace(/\[([\d\s,]+)\]/g, (match, p1) => {
                        const numbers = p1.split(',').map(n => n.trim()).filter(n => /^\d+$/.test(n));
                        if (numbers.length === 0) return match;
                        return numbers.map(
                            n => `<sup class="cite-ref" data-msg-index="${msgIndex}" data-chunk-index="${n}">[${n}]</sup>`
                        ).join('');
                    });
                }
                return part;
            }).join('');
        },

        scrollToChunk(msgIndex, chunkIndex) {
            const msgEl = document.querySelectorAll('.message')[msgIndex];
            if (!msgEl) return;
            const details = msgEl.querySelector('details.reasoning-details');
            if (details) details.open = true;
            const chunkEl = document.getElementById(`chunk-${msgIndex}-${chunkIndex}`);
            if (chunkEl) {
                chunkEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
                chunkEl.classList.add('highlight-chunk');
                setTimeout(() => chunkEl.classList.remove('highlight-chunk'), 2000);
            }
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        /**
         * 将单条 rag step 追加到分组列表。
         * 具名 group（子 Agent）：按 group id 合并，支持并行交错到达。
         * 无 group（主流程）：仅追加到最后一个无 group 分组，保留前后两段主流程步骤分离。
         */
        appendRagStepToGroups(prev, step) {
            const groups = prev || [];
            const g = step.group || null;
            if (g) {
                const idx = groups.findIndex(grp => grp.group === g);
                if (idx >= 0) {
                    const existing = groups[idx];
                    const updated = {
                        group: existing.group,
                        label: existing.label,
                        steps: [...existing.steps, step],
                        collapsed: existing.collapsed,
                    };
                    return [...groups.slice(0, idx), updated, ...groups.slice(idx + 1)];
                }
                return [...groups, { group: g, label: g, steps: [step], collapsed: true }];
            }
            const last = groups.length > 0 ? groups[groups.length - 1] : null;
            if (last && last.group === null) {
                const updated = { ...last, steps: [...last.steps, step] };
                return [...groups.slice(0, -1), updated];
            }
            return [...groups, { group: null, label: null, steps: [step], collapsed: false }];
        },

        /**
         * 将 ragSteps 按 group 字段分组（仅用于历史会话加载等一次性场景）。
         * 返回 [{ group: string|null, label: string, steps: [], collapsed: bool }]
         */
        groupRagSteps(steps) {
            if (!steps || !steps.length) return [];
            return steps.reduce((groups, step) => this.appendRagStepToGroups(groups, step), []);
        },

        toggleStepGroup(msgIndex, groupIndex) {
            const msg = this.messages[msgIndex];
            if (!msg || !msg._groupedSteps || !msg._groupedSteps[groupIndex]) return;
            msg._groupedSteps[groupIndex].collapsed = !msg._groupedSteps[groupIndex].collapsed;
        },

        formatCandidateKLabel(trace) {
            if (!trace || trace.candidate_k == null) {
                return '';
            }
            const k = trace.candidate_k;
            if (trace.candidate_k_config_error) {
                return `Milvus 候选池：${k}（${trace.candidate_k_config_error}，已回退倍数计算）`;
            }
            if (trace.candidate_k_source === 'env') {
                return `Milvus 候选池：${k}（环境变量 RETRIEVAL_CANDIDATE_K）`;
            }
            const multiplier = trace.retrieval_candidate_multiplier;
            if (multiplier != null) {
                return `Milvus 候选池：${k}（top_k × ${multiplier}）`;
            }
            return `Milvus 候选池：${k}`;
        },

        formatGradeScore(trace) {
            if (!trace || !trace.grade_score) {
                return '';
            }
            if (trace.grade_score === 'unknown') {
                return trace.grade_error ? '不可用（结构化评分解析失败）' : '不可用';
            }
            return trace.grade_score;
        },

        hasRetrievalFunnel(trace) {
            if (!trace) {
                return false;
            }
            return trace.recall_count != null
                || trace.post_merge_candidate_count != null
                || trace.candidate_count != null;
        },

        hasRagTraceDetails(trace) {
            if (!trace || typeof trace !== 'object') {
                return false;
            }
            return Object.keys(trace).length > 0;
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await fetch(url, opts);
            if (response.status === 401) {
                this.handleLogout();
                throw new Error('登录已过期，请重新登录');
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch('/auth/me');
            if (!response.ok) {
                throw new Error('认证失败');
            }
            this.currentUser = await response.json();
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                alert('用户名和密码不能为空');
                return;
            }

            this.authLoading = true;
            try {
                const endpoint = this.authMode === 'login' ? '/auth/login' : '/auth/register';
                const payload = {
                    username,
                    password
                };
                if (this.authMode === 'register') {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || '认证失败');
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                localStorage.setItem('accessToken', this.token);
                this.authForm.password = '';
                this.authForm.admin_code = '';
                this.messages = [];
                this.sessionId = 'session_' + Date.now();
                this.activeNav = 'newChat';
            } catch (error) {
                alert(error.message);
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout() {
            this.token = '';
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.documents = [];
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
            localStorage.removeItem('accessToken');
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        handleStop() {
            if (this.abortController) {
                this.abortController.abort();
            }
        },

        handleImageSelect(e) {
            const file = e.target.files[0];
            if (file) {
                this.selectedImage = file;
                if (this.queryImagePath) {
                    URL.revokeObjectURL(this.queryImagePath);
                }
                this.queryImagePath = URL.createObjectURL(file);
                const reader = new FileReader();
                reader.onload = () => {
                    this.queryImageBase64 = reader.result;
                };
                reader.readAsDataURL(file);
            }
        },

        clearImage() {
            this.selectedImage = null;
            if (this.queryImagePath) {
                URL.revokeObjectURL(this.queryImagePath);
            }
            this.queryImagePath = null;
            this.queryImageBase64 = null;
            if (this.$refs.imageInput) {
                this.$refs.imageInput.value = '';
            }
        },

        readImageAsDataUrl(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = () => reject(reader.error || new Error('图片读取失败'));
                reader.readAsDataURL(file);
            });
        },

        imageAttachmentStorageKey(sessionId) {
            return `queryImageAttachments:${sessionId}`;
        },

        loadImageAttachments(sessionId) {
            try {
                return JSON.parse(localStorage.getItem(this.imageAttachmentStorageKey(sessionId)) || '{}');
            } catch (_) {
                return {};
            }
        },

        saveImageAttachment(sessionId, messageIndex, attachment) {
            if (!attachment || !attachment.imageUrl) return;
            try {
                const attachments = this.loadImageAttachments(sessionId);
                attachments[String(messageIndex)] = attachment;
                localStorage.setItem(
                    this.imageAttachmentStorageKey(sessionId),
                    JSON.stringify(attachments)
                );
            } catch (_) {
                this.pruneOldImageAttachmentCaches(sessionId);
                try {
                    const attachments = this.loadImageAttachments(sessionId);
                    attachments[String(messageIndex)] = attachment;
                    localStorage.setItem(
                        this.imageAttachmentStorageKey(sessionId),
                        JSON.stringify(attachments)
                    );
                } catch (error) {
                    console.warn('Image preview could not be cached:', error);
                }
            }
        },

        pruneOldImageAttachmentCaches(currentSessionId) {
            const prefix = 'queryImageAttachments:';
            Object.keys(localStorage)
                .filter(key => key.startsWith(prefix) && key !== this.imageAttachmentStorageKey(currentSessionId))
                .forEach(key => localStorage.removeItem(key));
        },

        async handleSend() {
            if (!this.isAuthenticated) {
                alert('请先登录');
                return;
            }

            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing) return;
            let queryImageBase64 = this.queryImageBase64;
            if (this.selectedImage && !queryImageBase64) {
                queryImageBase64 = await this.readImageAsDataUrl(this.selectedImage);
            }
            const messageImageName = this.selectedImage ? this.selectedImage.name : '';
            const messageImageUrl = queryImageBase64 || null;
            const userMsgIdx = this.messages.length;

            this.messages.push({
                text: text,
                isUser: true,
                imageUrl: messageImageUrl,
                imageName: messageImageName
            });
            this.saveImageAttachment(this.sessionId, userMsgIdx, {
                imageUrl: messageImageUrl,
                imageName: messageImageName
            });

            if (this.messages.length === 1) {
                const tempTitle = text.length > 10 ? text.substring(0, 10) + '...' : text;
                const existingSession = this.sessions.find(s => s.session_id === this.sessionId);
                if (!existingSession) {
                    this.sessions.unshift({
                        session_id: this.sessionId,
                        title: tempTitle,
                        message_count: 1,
                        updated_at: new Date().toISOString()
                    });
                }
            }

            this.userInput = '';
            this.clearImage();
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scrollToBottom();
            });

            this.isLoading = true;
            this.messages.push({
                text: '',
                isUser: false,
                isThinking: true,
                ragTrace: null,
                ragSteps: [],
                _groupedSteps: []
            });
            const botMsgIdx = this.messages.length - 1;
            const requestSessionId = this.sessionId;

            this.abortController = new AbortController();

            try {
                const response = await this.authFetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        session_id: this.sessionId,
                        query_image_base64: queryImageBase64 || null,
                    }),
                    signal: this.abortController.signal,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf('\n\n')) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);

                        if (eventStr.startsWith('data: ')) {
                            const dataStr = eventStr.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                const botMsg = this.sessionId === requestSessionId
                                    ? this.messages[botMsgIdx]
                                    : null;
                                if (!botMsg) {
                                    continue;
                                }
                                if (data.type === 'content') {
                                    if (botMsg.isThinking) {
                                        botMsg.isThinking = false;
                                    }
                                    botMsg.text += data.content;
                                } else if (data.type === 'trace') {
                                    botMsg.ragTrace = data.rag_trace;
                                } else if (data.type === 'rag_step') {
                                    if (!botMsg.ragSteps) botMsg.ragSteps = [];
                                    botMsg.ragSteps.push(data.step);
                                    botMsg._groupedSteps = this.appendRagStepToGroups(botMsg._groupedSteps, data.step);
                                } else if (data.type === 'session_title') {
                                    const s = this.sessions.find(
                                        item => item.session_id === data.session_id
                                    );
                                    if (s) {
                                        s.title = data.title;
                                        s.updated_at = new Date().toISOString();
                                        s.message_count = this.messages.length;
                                    } else {
                                        this.sessions.unshift({
                                            session_id: data.session_id,
                                            title: data.title,
                                            message_count: this.messages.length,
                                            updated_at: new Date().toISOString()
                                        });
                                    }
                                } else if (data.type === 'error') {
                                    botMsg.isThinking = false;
                                    botMsg.text += `\n[Error: ${data.content}]`;
                                }
                            } catch (e) {
                                console.warn('SSE parse error:', e);
                            }
                        }
                    }
                    this.$nextTick(() => this.scrollToBottom());
                }

            } catch (error) {
                const botMsg = this.sessionId === requestSessionId
                    ? this.messages[botMsgIdx]
                    : null;
                if (!botMsg) {
                    return;
                }
                if (error.name === 'AbortError') {
                    botMsg.isThinking = false;
                    if (!botMsg.text) {
                        botMsg.text = '(已终止回答)';
                    } else {
                        botMsg.text += '\n\n_(回答已被终止)_';
                    }
                } else {
                    botMsg.isThinking = false;
                    botMsg.text = `错误：${error.message}`;
                }
            } finally {
                this.isLoading = false;
                this.abortController = null;
                this.$nextTick(() => this.scrollToBottom());
            }
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = 'auto';
            textarea.style.height = textarea.scrollHeight + 'px';
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = 'auto';
            }
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        canSwitchConversation() {
            if (!this.isLoading) return true;
            alert('当前回答仍在生成中，请先停止生成后再切换会话。');
            return false;
        },

        handleNewChat() {
            if (!this.isAuthenticated) return;
            if (!this.canSwitchConversation()) return;
            this.messages = [];
            this.sessionId = 'session_' + Date.now();
            this.activeNav = 'newChat';
            this.showHistorySidebar = false;
        },

        handleClearChat() {
            if (confirm('确定要清空当前对话吗？')) {
                this.messages = [];
            }
        },

        async handleHistory() {
            if (!this.isAuthenticated) return;
            this.activeNav = 'history';
            this.showHistorySidebar = true;
            try {
                const response = await this.authFetch('/sessions');
                if (!response.ok) {
                    throw new Error('Failed to load sessions');
                }
                const data = await response.json();
                this.sessions = data.sessions;
            } catch (error) {
                alert('加载历史记录失败：' + error.message);
            }
        },

        async loadSession(sessionId) {
            if (!this.canSwitchConversation()) return;
            this.sessionId = sessionId;
            this.showHistorySidebar = false;
            this.activeNav = 'newChat';

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) {
                    throw new Error('Failed to load session messages');
                }
                const data = await response.json();
                const imageAttachments = this.loadImageAttachments(sessionId);
                this.messages = data.messages.map((msg, idx) => ({
                    text: msg.content,
                    isUser: msg.type === 'human',
                    ragTrace: msg.rag_trace || null,
                    imageUrl: imageAttachments[String(idx)]?.imageUrl || null,
                    imageName: imageAttachments[String(idx)]?.imageName || ''
                }));

                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            } catch (error) {
                alert('加载会话失败：' + error.message);
                this.messages = [];
            }
        },

        async deleteSession(sessionId) {
            const sessionLabel = this.sessions.find(s => s.session_id === sessionId)?.title || sessionId;
            if (!confirm(`确定要删除会话 "${sessionLabel}" 吗？`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                    method: 'DELETE'
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || 'Delete failed');
                }

                this.sessions = this.sessions.filter(s => s.session_id !== sessionId);
                localStorage.removeItem(this.imageAttachmentStorageKey(sessionId));

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = 'session_' + Date.now();
                    this.activeNav = 'newChat';
                }

                if (payload.message) {
                    alert(payload.message);
                }
            } catch (error) {
                alert('删除会话失败：' + error.message);
            }
        },

        handleSettings() {
            if (!this.isAdmin) {
                alert('仅管理员可访问文档管理');
                return;
            }
            this.activeNav = 'settings';
            this.showHistorySidebar = false;
            this.loadDocuments();
        },

        mergeDocumentsWithActiveDeletes(nextDocuments) {
            const merged = Array.isArray(nextDocuments) ? [...nextDocuments] : [];
            Object.keys(this.deleteJobs).forEach(filename => {
                const job = this.deleteJobs[filename];
                if (!job || job.status === 'failed') return;
                const exists = merged.some(doc => doc.filename === filename);
                if (!exists) {
                    const currentDoc = this.documents.find(doc => doc.filename === filename);
                    if (currentDoc) {
                        merged.push(currentDoc);
                    }
                }
            });
            return merged;
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const response = await this.authFetch('/documents');
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || 'Failed to load documents');
                }
                const data = await response.json();
                this.documents = this.mergeDocumentsWithActiveDeletes(data.documents);
                if (this.docPage > this.docTotalPages) {
                    this.docPage = this.docTotalPages;
                }
            } catch (error) {
                alert('加载文档列表失败：' + error.message);
            } finally {
                this.documentsLoading = false;
            }
        },

        handleFileSelect(event) {
            const files = event.target.files;
            if (files && files.length > 0) {
                this.selectedFile = files[0];
                this.uploadProgress = '';
                this.uploadSteps = this.createUploadSteps();
                this.uploadProgressCollapsed = false;
                this.activeUploadJobId = '';
            }
        },

        createUploadSteps() {
            return [
                { key: 'upload', label: '文档上传', percent: 0, status: 'pending', message: '' },
                { key: 'cleanup', label: '清理旧版本', percent: 0, status: 'pending', message: '' },
                { key: 'parse', label: '解析与分块', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '父级分块入库', percent: 0, status: 'pending', message: '' },
                { key: 'vector_store', label: '向量化入库', percent: 0, status: 'pending', message: '' },
            ];
        },

        updateUploadStep(key, percent, status = 'running', message = '') {
            if (!this.uploadSteps.length) {
                this.uploadSteps = this.createUploadSteps();
            }
            const idx = this.uploadSteps.findIndex(step => step.key === key);
            if (idx === -1) return;
            this.uploadSteps[idx] = {
                ...this.uploadSteps[idx],
                percent: Math.max(0, Math.min(100, Math.round(percent || 0))),
                status,
                message
            };
        },

        uploadFileWithProgress(file) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                const formData = new FormData();
                formData.append('file', file);

                xhr.open('POST', '/documents/upload/async');
                const headers = this.authHeaders();
                Object.entries(headers).forEach(([key, value]) => xhr.setRequestHeader(key, value));

                xhr.upload.onprogress = (event) => {
                    if (!event.lengthComputable) return;
                    const percent = Math.round((event.loaded / event.total) * 100);
                    this.updateUploadStep('upload', percent, 'running', `已上传 ${percent}%`);
                };

                xhr.onload = () => {
                    if (xhr.status === 401) {
                        this.handleLogout();
                        reject(new Error('登录已过期，请重新登录'));
                        return;
                    }

                    let data = {};
                    try {
                        data = JSON.parse(xhr.responseText || '{}');
                    } catch (e) {
                        reject(new Error('上传响应解析失败'));
                        return;
                    }

                    if (xhr.status < 200 || xhr.status >= 300) {
                        reject(new Error(data.detail || `HTTP ${xhr.status}`));
                        return;
                    }

                    this.updateUploadStep('upload', 100, 'completed', '文档上传完成');
                    resolve(data);
                };

                xhr.onerror = () => reject(new Error('上传请求失败'));
                xhr.onabort = () => reject(new Error('上传已取消'));
                xhr.send(formData);
            });
        },

        syncUploadJob(job) {
            this.activeUploadJobId = job.job_id;
            this.uploadProgress = job.message || '';
            if (Array.isArray(job.steps)) {
                this.uploadSteps = job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                }));
            }
            // 入库成功后自动收起步骤明细，保留摘要供用户再次展开查看。
            if (job.status === 'completed') {
                this.uploadProgressCollapsed = true;
            }
        },

        toggleUploadProgressCollapsed() {
            this.uploadProgressCollapsed = !this.uploadProgressCollapsed;
        },

        stopUploadJobPolling() {
            if (this.uploadPollTimer) {
                clearInterval(this.uploadPollTimer);
                this.uploadPollTimer = null;
            }
        },

        startUploadJobPolling(jobId) {
            this.stopUploadJobPolling();

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/upload/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load upload job');
                    }

                    const job = await response.json();
                    this.syncUploadJob(job);

                    if (job.status === 'completed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                        this.selectedFile = null;
                        if (this.$refs.fileInput) {
                            this.$refs.fileInput.value = '';
                        }
                        await this.loadDocuments();
                    } else if (job.status === 'failed') {
                        this.stopUploadJobPolling();
                        this.isUploading = false;
                    }
                } catch (error) {
                    this.uploadProgress = '进度查询失败：' + error.message;
                    this.stopUploadJobPolling();
                    this.isUploading = false;
                }
            };

            poll();
            this.uploadPollTimer = setInterval(poll, 1000);
        },

        async uploadDocument() {
            if (!this.selectedFile) {
                alert('请先选择文件');
                return;
            }

            const exists = this.documents.some(doc => doc.filename === this.selectedFile.name);
            if (exists) {
                const confirmed = confirm(`文件 "${this.selectedFile.name}" 已存在，是否覆盖？`);
                if (!confirmed) return;
            }

            this.isUploading = true;
            this.uploadProgress = '正在上传...';
            this.uploadSteps = this.createUploadSteps();
            this.uploadProgressCollapsed = false;
            this.updateUploadStep('upload', 0, 'running', '准备上传');

            try {
                const data = await this.uploadFileWithProgress(this.selectedFile);
                this.uploadProgress = data.message;
                this.activeUploadJobId = data.job_id;
                this.startUploadJobPolling(data.job_id);
            } catch (error) {
                this.updateUploadStep('upload', 100, 'failed', error.message);
                this.uploadProgress = '上传失败：' + error.message;
                this.isUploading = false;
            }
        },

        createDeleteSteps() {
            return [
                { key: 'prepare', label: '准备删除', percent: 0, status: 'pending', message: '' },
                { key: 'bm25', label: '同步 BM25 统计', percent: 0, status: 'pending', message: '' },
                { key: 'milvus', label: '删除向量数据', percent: 0, status: 'pending', message: '' },
                { key: 'parent_store', label: '删除父级分块', percent: 0, status: 'pending', message: '' },
            ];
        },

        isDeletingDocument(filename) {
            const job = this.deleteJobs[filename];
            return job && job.status === 'running';
        },

        isDeleteActionLocked(filename) {
            const job = this.deleteJobs[filename];
            return job && (job.status === 'running' || job.status === 'completed');
        },

        getDeleteButtonIcon(filename) {
            const job = this.deleteJobs[filename];
            if (job?.status === 'running') return 'fas fa-spinner fa-spin';
            if (job?.status === 'completed') return 'fas fa-check';
            return 'fas fa-trash';
        },

        setDeleteJob(filename, nextJob) {
            this.deleteJobs = {
                ...this.deleteJobs,
                [filename]: {
                    ...(this.deleteJobs[filename] || {}),
                    ...nextJob
                }
            };
        },

        syncDeleteJob(filename, job) {
            const current = this.deleteJobs[filename] || {};
            // 后端返回统一的步骤结构，前端只负责同步到当前文档行内卡片。
            this.setDeleteJob(filename, {
                jobId: job.job_id,
                status: job.status,
                message: job.message || '',
                collapsed: job.status === 'completed' ? true : Boolean(current.collapsed),
                steps: Array.isArray(job.steps) ? job.steps.map(step => ({
                    key: step.key,
                    label: step.label,
                    percent: step.percent,
                    status: step.status,
                    message: step.message || ''
                })) : this.createDeleteSteps()
            });
        },

        toggleDeleteJobCollapsed(filename) {
            const job = this.deleteJobs[filename];
            if (!job) return;
            this.setDeleteJob(filename, { collapsed: !job.collapsed });
        },

        stopDeleteJobPolling(filename) {
            const timer = this.deletePollTimers[filename];
            if (!timer) return;
            clearInterval(timer);
            const { [filename]: _removed, ...rest } = this.deletePollTimers;
            this.deletePollTimers = rest;
        },

        stopAllDeleteJobPolling() {
            Object.keys(this.deletePollTimers).forEach(filename => this.stopDeleteJobPolling(filename));
        },

        clearDeleteRemovalTimer(filename) {
            const timer = this.deleteRemoveTimers[filename];
            if (!timer) return;
            clearTimeout(timer);
            const { [filename]: _removed, ...rest } = this.deleteRemoveTimers;
            this.deleteRemoveTimers = rest;
        },

        scheduleDeletedDocumentRemoval(filename) {
            this.clearDeleteRemovalTimer(filename);
            // 删除完成后先保留 3 秒摘要，再从当前列表移除并刷新后端状态。
            const timer = setTimeout(async () => {
                this.documents = this.documents.filter(doc => doc.filename !== filename);
                const { [filename]: _job, ...jobs } = this.deleteJobs;
                const { [filename]: _timer, ...timers } = this.deleteRemoveTimers;
                this.deleteJobs = jobs;
                this.deleteRemoveTimers = timers;
                await this.loadDocuments();
            }, 3000);
            this.deleteRemoveTimers = {
                ...this.deleteRemoveTimers,
                [filename]: timer
            };
        },

        startDeleteJobPolling(filename, jobId) {
            this.stopDeleteJobPolling(filename);

            const poll = async () => {
                try {
                    const response = await this.authFetch(`/documents/delete/jobs/${encodeURIComponent(jobId)}`);
                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to load delete job');
                    }

                    const job = await response.json();
                    this.syncDeleteJob(filename, job);

                    if (job.status === 'completed') {
                        this.stopDeleteJobPolling(filename);
                        this.scheduleDeletedDocumentRemoval(filename);
                    } else if (job.status === 'failed') {
                        this.stopDeleteJobPolling(filename);
                    }
                } catch (error) {
                    this.setDeleteJob(filename, {
                        status: 'failed',
                        message: '删除进度查询失败：' + error.message,
                        collapsed: false,
                        steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                    });
                    this.stopDeleteJobPolling(filename);
                }
            };

            poll();
            this.deletePollTimers = {
                ...this.deletePollTimers,
                [filename]: setInterval(poll, 1000)
            };
        },

        async deleteDocument(filename) {
            if (this.isDeletingDocument(filename)) {
                return;
            }
            if (!confirm(`确定要删除文档 "${filename}" 吗？这将同时删除 Milvus 中的所有相关向量。`)) {
                return;
            }

            this.clearDeleteRemovalTimer(filename);
            this.setDeleteJob(filename, {
                status: 'running',
                message: '正在提交删除任务...',
                collapsed: false,
                steps: this.createDeleteSteps().map(step => (
                    step.key === 'prepare'
                        ? { ...step, percent: 1, status: 'running', message: '正在提交删除任务' }
                        : step
                ))
            });

            try {
                const response = await this.authFetch(`/documents/delete/async/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || 'Delete failed');
                }

                const data = await response.json();
                this.setDeleteJob(filename, {
                    jobId: data.job_id,
                    status: 'running',
                    message: data.message || `正在删除 ${filename}`,
                    collapsed: false
                });
                this.startDeleteJobPolling(filename, data.job_id);

            } catch (error) {
                this.setDeleteJob(filename, {
                    status: 'failed',
                    message: '删除文档失败：' + error.message,
                    collapsed: false,
                    steps: this.deleteJobs[filename]?.steps || this.createDeleteSteps()
                });
            }
        },

        getFileIcon(fileType) {
            const value = String(fileType || '').toLowerCase();
            if (value === 'pdf' || value.endsWith('.pdf')) {
                return 'fas fa-file-pdf';
            } else if (value === 'word' || value.endsWith('.doc') || value.endsWith('.docx')) {
                return 'fas fa-file-word';
            } else if (value === 'excel' || value.endsWith('.xls') || value.endsWith('.xlsx')) {
                return 'fas fa-file-excel';
            } else if (value.endsWith('.html') || value.endsWith('.htm')) {
                return 'fas fa-file-code';
            }
            return 'fas fa-file';
        }
    },
    watch: {
        docTotalPages(newTotal) {
            if (this.docPage > newTotal) {
                this.docPage = newTotal;
            }
        },
        messages: {
            handler() {
                this.$nextTick(() => {
                    this.scrollToBottom();
                });
            },
            deep: true
        }
    }
}).mount('#app');
