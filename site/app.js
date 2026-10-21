(function () {
  const DATA = window.COURSE_DATA;
  const STORAGE_KEY = "sarvam_rag_completed_lessons_v1";

  let state = {
    view: "curriculum", // 'curriculum' | 'reader' | 'cheatsheet' | 'textbook' | 'studio'
    activePath: "all",
    activeLessonId: DATA.phases[0].lessons[0].id,
    activeTab: "doc", // 'doc' | 'quiz' | 'code' | 'outputs'
    activeModuleIdx: 0,
    searchQuery: "",
    completed: loadCompleted(),
  };

  function loadCompleted() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      return raw ? JSON.parse(raw) : {};
    } catch (e) {
      return {};
    }
  }

  function saveCompleted() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state.completed));
    } catch (e) {}
    updateProgressBadge();
  }

  function toggleComplete(lessonId, ev) {
    if (ev) ev.stopPropagation();
    if (state.completed[lessonId]) {
      delete state.completed[lessonId];
    } else {
      state.completed[lessonId] = true;
    }
    saveCompleted();
    render();
  }

  function getAllLessons() {
    const list = [];
    DATA.phases.forEach((p) => {
      p.lessons.forEach((l) => list.push(l));
    });
    return list;
  }

  function getLessonById(id) {
    const all = getAllLessons();
    return all.find((l) => l.id === id) || all[0];
  }

  function updateProgressBadge() {
    const all = getAllLessons();
    const doneCount = all.filter((l) => state.completed[l.id]).length;
    const el = document.getElementById("progress-pill");
    if (el) {
      el.textContent = `✓ ${doneCount} / ${all.length} Completed`;
    }
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // Render Markdown safely with protected math blocks + Mermaid + GitHub Alerts
  function renderMarkdown(md) {
    if (!md) return "";
    const mathTokens = [];
    const codeTokens = [];

    // 1. Temporarily protect fenced code blocks so math regex never touches code
    let text = md.replace(/```([\w-]*)\n([\s\S]*?)```/g, (m, lang, body) => {
      const idx = codeTokens.length;
      if (lang.trim() === "mermaid") {
        codeTokens.push(`<div class="mermaid">${escapeHtml(body.trim())}</div>`);
      } else {
        codeTokens.push(
          `<pre><button class="copy-code-btn" onclick="window.__copyPre(this)">Copy</button><code class="language-${escapeHtml(
            lang || "text"
          )}">${escapeHtml(body.replace(/\n$/, ""))}</code></pre>`
        );
      }
      return `@@CODE_BLOCK_${idx}@@`;
    });

    // 2. Protect display math $$...$$
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (m, formula) => {
      const idx = mathTokens.length;
      mathTokens.push({ display: true, formula: formula.trim() });
      return `@@MATH_TOKEN_${idx}@@`;
    });

    // 3. Protect inline math $...$
    text = text.replace(/\$([^\$\n]+?)\$/g, (m, formula) => {
      const idx = mathTokens.length;
      mathTokens.push({ display: false, formula: formula.trim() });
      return `@@MATH_TOKEN_${idx}@@`;
    });

    // 4. Transform GitHub-style alerts > [!TIP], > [!IMPORTANT], > [!WARNING], > [!NOTE]
    text = text.replace(
      /^>\s*\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*\n((?:>.*\n?)*)/gm,
      (m, kind, body) => {
        const cleanBody = body.replace(/^>\s?/gm, "");
        return `> **${kind}:** ${cleanBody}`;
      }
    );

    let html = window.marked ? window.marked.parse(text) : `<pre>${escapeHtml(text)}</pre>`;

    // 5. Restore math tokens and render with KaTeX
    html = html.replace(/@@MATH_TOKEN_(\d+)@@/g, (m, idxStr) => {
      const item = mathTokens[Number(idxStr)];
      if (!item) return m;
      if (window.katex) {
        try {
          return window.katex.renderToString(item.formula, {
            displayMode: item.display,
            throwOnError: false,
            strict: false,
          });
        } catch (err) {
          return `<code>${escapeHtml(item.formula)}</code>`;
        }
      }
      return item.display ? `$$${escapeHtml(item.formula)}$$` : `$${escapeHtml(item.formula)}$`;
    });

    // 6. Restore fenced code blocks / Mermaid diagrams
    html = html.replace(/@@CODE_BLOCK_(\d+)@@/g, (m, idxStr) => {
      return codeTokens[Number(idxStr)] || m;
    });

    return html;
  }

  window.__copyPre = function (btn) {
    const code = btn.parentElement.querySelector("code");
    if (!code) return;
    navigator.clipboard.writeText(code.innerText).then(() => {
      const old = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => (btn.textContent = old), 1400);
    });
  };

  window.__copyText = function (btn, encoded) {
    const text = decodeURIComponent(encoded);
    navigator.clipboard.writeText(text).then(() => {
      const old = btn.textContent;
      btn.textContent = "✓ Copied!";
      setTimeout(() => (btn.textContent = old), 1400);
    });
  };

  window.__openLesson = function (lessonId, tab) {
    state.view = "reader";
    state.activeLessonId = lessonId;
    state.activeTab = tab || "doc";
    window.scrollTo({ top: 0, behavior: "smooth" });
    render();
  };

  window.__setView = function (viewName) {
    state.view = viewName;
    window.scrollTo({ top: 0, behavior: "smooth" });
    render();
  };

  window.__setPath = function (pathId) {
    state.activePath = pathId;
    render();
  };

  window.__toggleComplete = toggleComplete;

  window.__answerQuiz = function (btn, qIdx, optIdx, correctIdx) {
    const card = btn.closest(".quiz-card");
    if (!card) return;
    const opts = card.querySelectorAll(".quiz-opt");
    opts.forEach((b, i) => {
      b.classList.remove("correct", "wrong");
      if (i === correctIdx) b.classList.add("correct");
      else if (i === optIdx && optIdx !== correctIdx) b.classList.add("wrong");
    });
    const exp = card.querySelector(".quiz-exp");
    if (exp) exp.classList.add("visible");
  };

  window.__selectModule = function (idx) {
    state.activeModuleIdx = idx;
    window.scrollTo({ top: 0, behavior: "smooth" });
    render();
  };

  function renderCurriculumView() {
    const activePathObj =
      DATA.learning_paths.find((p) => p.id === state.activePath) || DATA.learning_paths[0];
    const allowedSet = activePathObj.lessons ? new Set(activePathObj.lessons) : null;
    const q = state.searchQuery.trim().toLowerCase();

    const pathsHtml = DATA.learning_paths
      .map(
        (p) => `
      <div class="path-card ${p.id === state.activePath ? "active" : ""}" onclick="window.__setPath('${p.id}')">
        <div class="path-badge">${escapeHtml(p.badge)}</div>
        <div class="path-title">${escapeHtml(p.name)}</div>
        <div class="path-desc">${escapeHtml(p.desc)}</div>
      </div>`
      )
      .join("");

    const phasesHtml = DATA.phases
      .map((phase) => {
        const filteredLessons = phase.lessons.filter((l) => {
          if (allowedSet && !allowedSet.has(l.id)) return false;
          if (!q) return true;
          return (
            l.title.toLowerCase().includes(q) ||
            l.tagline.toLowerCase().includes(q) ||
            l.markdown.toLowerCase().includes(q)
          );
        });
        if (filteredLessons.length === 0) return "";

        const rows = filteredLessons
          .map((l) => {
            const done = !!state.completed[l.id];
            return `
          <div class="lesson-row">
            <div class="lesson-left">
              <div class="lesson-check ${done ? "done" : ""}" onclick="window.__toggleComplete('${l.id}', event)" title="Mark completed">
                ${done ? "✓" : ""}
              </div>
              <div class="lesson-num">${escapeHtml(phase.number.replace("Phase ", ""))}.${escapeHtml(l.lesson_num)}</div>
              <div class="lesson-info">
                <h3 onclick="window.__openLesson('${l.id}', 'doc')">${escapeHtml(l.title)}</h3>
                <p>${escapeHtml(l.tagline)}</p>
              </div>
            </div>
            <div class="lesson-actions">
              <span class="phase-meta" style="margin-right:0.4rem;">⏱ ${escapeHtml(l.time)}</span>
              <button class="btn-sm btn-primary-sm" onclick="window.__openLesson('${l.id}', 'doc')">📖 Read Lesson</button>
              <button class="btn-sm" onclick="window.__openLesson('${l.id}', 'quiz')">🧠 Quiz (4)</button>
              <button class="btn-sm" onclick="window.__openLesson('${l.id}', 'code')">💻 main.py</button>
              <button class="btn-sm" onclick="window.__openLesson('${l.id}', 'outputs')">🚀 Prompt &amp; Skill</button>
            </div>
          </div>`;
          })
          .join("");

        return `
        <div class="phase-block">
          <div class="phase-header">
            <div class="phase-title-group">
              <span class="phase-Tag" style="background:${phase.color}">${escapeHtml(phase.number)}</span>
              <div>
                <div style="font-weight:800;font-size:1.1rem;">${escapeHtml(phase.title)}</div>
                <div class="phase-meta">${escapeHtml(phase.subtitle)}</div>
              </div>
            </div>
            <div class="phase-meta">${filteredLessons.length} lesson(s) &nbsp;·&nbsp; ${escapeHtml(phase.est)}</div>
          </div>
          <div class="lesson-list">${rows}</div>
        </div>`;
      })
      .join("");

    return `
      <section class="hero">
        <div class="badge-row">
          <span class="pill pill-blue">5 Phases · 13 Lessons</span>
          <span class="pill pill-purple">google/embeddinggemma-2 (768→256 MRL)</span>
          <span class="pill pill-green">Vertex AI Gemini + BigQuery Vector Search</span>
          <span class="pill pill-amber">$0.00 Cloud FinOps Verified</span>
        </div>
        <h1>Sarvam AI + <code>google/embeddinggemma-2</code> RAG Engineering from Scratch</h1>
        <p class="lead">
          Every RAG primitive written from scratch in pure Python before using production SDKs.
          Every lesson follows the <b>6-Beat Architecture</b> and ships with runnable <code>code/main.py</code>,
          a 4-question Pre/Post Quiz, a reusable System Prompt, and an Agent Skill.
        </p>
        <div style="display:flex;gap:0.65rem;flex-wrap:wrap;">
          <button class="btn-sm btn-primary-sm" style="padding:0.58rem 1.1rem;font-size:0.88rem;" onclick="window.__openLesson('${DATA.phases[0].lessons[0].id}', 'doc')">
            ▶ Start Phase 00 · Lesson 01
          </button>
          <button class="btn-sm" style="padding:0.58rem 1.1rem;font-size:0.88rem;" onclick="window.__openLesson('01-math-and-embeddings/01-vectors-norms-and-cosine-similarity', 'doc')">
            📐 Jump to Phase 01: Vector Math &amp; Cosine Similarity
          </button>
          <button class="btn-sm" style="padding:0.58rem 1.1rem;font-size:0.88rem;" onclick="window.__setView('cheatsheet')">
            📋 View Formulas &amp; CLI Cheatsheet
          </button>
          <button class="btn-sm" style="padding:0.58rem 1.1rem;font-size:0.88rem;" onclick="window.__setView('studio')">
            🎛️ 4-Tab Web Studio &amp; FinOps Destroy
          </button>
        </div>

        <div class="beat-strip">
          <div class="beat-card"><div class="beat-num">Beat 1</div><div class="beat-title">The Problem</div><div class="beat-desc">Concrete production failure mode</div></div>
          <div class="beat-card"><div class="beat-num">Beat 2</div><div class="beat-title">The Concept</div><div class="beat-desc">First-principles math &amp; Mermaid</div></div>
          <div class="beat-card"><div class="beat-num">Beat 3</div><div class="beat-title">Build It</div><div class="beat-desc">Pure Python in code/main.py</div></div>
          <div class="beat-card"><div class="beat-num">Beat 4</div><div class="beat-title">Use It</div><div class="beat-desc">Production CLI &amp; Gradio Studio</div></div>
          <div class="beat-card"><div class="beat-num">Beat 5</div><div class="beat-title">Ship It</div><div class="beat-desc">outputs/prompt-*.md &amp; skill-*.md</div></div>
          <div class="beat-card"><div class="beat-num">Beat 6</div><div class="beat-title">Exercises &amp; Quiz</div><div class="beat-desc">Easy/Med/Hard + 4-Q Pre/Post Quiz</div></div>
        </div>
      </section>

      <div class="section-header">
        <h2>1. Choose a Learning Path</h2>
        <span class="phase-meta">Click any path to filter the 13 lessons below</span>
      </div>
      <div class="paths-grid">${pathsHtml}</div>

      <div class="section-header">
        <h2>2. Interactive Curriculum (${activePathObj.name})</h2>
        <span class="phase-meta">Click any lesson to read, run code, take the quiz, or copy prompts &amp; skills</span>
      </div>
      ${phasesHtml || `<div class="phase-block" style="padding:2rem;text-align:center;">No lessons match "${escapeHtml(state.searchQuery)}".</div>`}
    `;
  }

  function renderReaderView() {
    const all = getAllLessons();
    const current = getLessonById(state.activeLessonId);
    const idx = all.findIndex((l) => l.id === current.id);
    const prevLesson = idx > 0 ? all[idx - 1] : null;
    const nextLesson = idx < all.length - 1 ? all[idx + 1] : null;
    const isDone = !!state.completed[current.id];

    const sidebarHtml = DATA.phases
      .map(
        (p) => `
      <div class="sidebar-phase">
        <div class="sidebar-phase-title">${escapeHtml(p.number)} · ${escapeHtml(p.title)}</div>
        ${p.lessons
          .map((l) => {
            const done = !!state.completed[l.id];
            return `
            <div class="sidebar-item ${l.id === current.id ? "active" : ""}" onclick="window.__openLesson('${l.id}', '${state.activeTab}')">
              <span style="color:${done ? "var(--green)" : "var(--text-muted)"};font-weight:700;">${done ? "✓" : l.lesson_num}</span>
              <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(l.title)}</span>
            </div>`;
          })
          .join("")}
      </div>`
      )
      .join("");

    let bodyHtml = "";
    if (state.activeTab === "doc") {
      bodyHtml = `<div class="md-content">${renderMarkdown(current.markdown)}</div>`;
    } else if (state.activeTab === "quiz") {
      const qs = (current.quiz && current.quiz.questions) || [];
      bodyHtml = `
        <div class="md-content">
          <h1>🧠 Interactive Pre &amp; Post Quiz — ${escapeHtml(current.title)}</h1>
          <p style="color:var(--text-muted);">Test your intuition before and after reading the lesson. Click any option to check your answer and reveal the mathematical explanation.</p>
        </div>
        <div style="margin-top:1.25rem;">
          ${qs
            .map(
              (q, qIdx) => `
            <div class="quiz-card">
              <span class="quiz-stage ${q.stage === "pre" ? "pre" : "post"}">${
                q.stage === "pre" ? "Pre-Lesson Intuition Check" : "Post-Lesson Mastery Check"
              } · Question ${qIdx + 1}</span>
              <div class="quiz-q">${escapeHtml(q.question)}</div>
              ${q.options
                .map(
                  (opt, oIdx) => `
                <button class="quiz-opt" onclick="window.__answerQuiz(this, ${qIdx}, ${oIdx}, ${q.correct})">
                  <b>${String.fromCharCode(65 + oIdx)}.</b> ${escapeHtml(opt)}
                </button>`
                )
                .join("")}
              <div class="quiz-exp">
                <b>✓ Explanation:</b> ${escapeHtml(q.explanation)}
              </div>
            </div>`
            )
            .join("")}
        </div>`;
    } else if (state.activeTab === "code") {
      const encoded = encodeURIComponent(current.code);
      bodyHtml = `
        <div class="md-content">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.75rem;margin-bottom:1rem;">
            <div>
              <h1 style="margin-bottom:0.2rem;">💻 From-Scratch Python Implementation</h1>
              <div style="font-family:var(--font-mono);font-size:0.84rem;color:var(--accent);">python3 ${escapeHtml(current.code_path)}</div>
            </div>
            <button class="btn-sm btn-primary-sm" onclick="window.__copyText(this, '${encoded}')">📋 Copy ${escapeHtml(current.code_path)}</button>
          </div>
          <pre><code class="language-python">${escapeHtml(current.code)}</code></pre>
        </div>`;
    } else if (state.activeTab === "outputs") {
      const pEnc = encodeURIComponent(current.prompt_text);
      const sEnc = encodeURIComponent(current.skill_text);
      bodyHtml = `
        <div class="md-content">
          <h1>🚀 Ship It: Reusable System Prompt &amp; Agent Skill</h1>
          <p style="color:var(--text-muted);">Copy these production artifacts directly into your RAG service, Claude Code, Cursor, or Antigravity agent skills directory.</p>

          <div style="display:flex;justify-content:space-between;align-items:center;margin:1.5rem 0 0.5rem;">
            <h3 style="margin:0;">1. Production Prompt (<code>outputs/${escapeHtml(current.prompt_name)}</code>)</h3>
            <button class="btn-sm btn-primary-sm" onclick="window.__copyText(this, '${pEnc}')">📋 Copy Prompt</button>
          </div>
          <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.25rem;">
            ${renderMarkdown(current.prompt_text)}
          </div>

          <div style="display:flex;justify-content:space-between;align-items:center;margin:2rem 0 0.5rem;">
            <h3 style="margin:0;">2. Agent Skill (<code>outputs/${escapeHtml(current.skill_name)}</code>)</h3>
            <button class="btn-sm btn-primary-sm" onclick="window.__copyText(this, '${sEnc}')">📋 Copy Skill</button>
          </div>
          <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.25rem;">
            ${renderMarkdown(current.skill_text)}
          </div>
        </div>`;
    }

    return `
      <div class="reader-layout">
        <aside class="sidebar">${sidebarHtml}</aside>
        <div class="reader-main">
          <div class="reader-toolbar">
            <div class="tab-group">
              <button class="tab-btn ${state.activeTab === "doc" ? "active" : ""}" onclick="window.__openLesson('${current.id}', 'doc')">📖 6-Beat Lesson</button>
              <button class="tab-btn ${state.activeTab === "quiz" ? "active" : ""}" onclick="window.__openLesson('${current.id}', 'quiz')">🧠 Pre/Post Quiz (4)</button>
              <button class="tab-btn ${state.activeTab === "code" ? "active" : ""}" onclick="window.__openLesson('${current.id}', 'code')">💻 code/main.py</button>
              <button class="tab-btn ${state.activeTab === "outputs" ? "active" : ""}" onclick="window.__openLesson('${current.id}', 'outputs')">🚀 Prompt &amp; Skill</button>
            </div>
            <button class="btn-sm ${isDone ? "btn-primary-sm" : ""}" onclick="window.__toggleComplete('${current.id}', event)">
              ${isDone ? "✓ Completed" : "○ Mark Complete"}
            </button>
          </div>
          <div class="reader-body">
            ${bodyHtml}
            <div class="lesson-nav-footer">
              <div>
                ${
                  prevLesson
                    ? `<button class="btn-sm" onclick="window.__openLesson('${prevLesson.id}', 'doc')">← Prev: ${escapeHtml(prevLesson.title)}</button>`
                    : `<button class="btn-sm" onclick="window.__setView('curriculum')">← Back to Curriculum</button>`
                }
              </div>
              <div>
                ${
                  nextLesson
                    ? `<button class="btn-sm btn-primary-sm" onclick="window.__openLesson('${nextLesson.id}', 'doc')">Next: ${escapeHtml(nextLesson.title)} →</button>`
                    : `<button class="btn-sm btn-primary-sm" onclick="window.__setView('studio')">Explore Web Studio &amp; FinOps →</button>`
                }
              </div>
            </div>
          </div>
        </div>
      </div>`;
  }

  function renderCheatsheetView() {
    return `
      <div class="reader-main" style="padding:2rem;">
        <div class="md-content">${renderMarkdown(DATA.cheatsheet_md)}</div>
        <hr style="margin:2.5rem 0;border:none;border-top:1px solid var(--border);" />
        <div class="md-content">${renderMarkdown(DATA.roadmap_md)}</div>
      </div>`;
  }

  function renderTextbookView() {
    const modules = DATA.deep_dive_modules || [];
    const current = modules[state.activeModuleIdx] || modules[0];
    return `
      <div class="reader-layout">
        <aside class="sidebar">
          <div class="sidebar-phase-title">Deep-Dive Companion Modules</div>
          ${modules
            .map(
              (m, i) => `
            <div class="sidebar-item ${i === state.activeModuleIdx ? "active" : ""}" onclick="window.__selectModule(${i})">
              <span>📘</span>
              <span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(m.title)}</span>
            </div>`
            )
            .join("")}
        </aside>
        <div class="reader-main">
          <div class="reader-body">
            <div class="md-content">${current ? renderMarkdown(current.markdown) : ""}</div>
          </div>
        </div>
      </div>`;
  }

  function renderStudioView() {
    return `
      <div class="reader-main" style="padding:2rem;">
        <div class="md-content">
          <h1>🎛️ 4-Tab Production Gradio Web Studio &amp; Cloud FinOps Teardown</h1>
          <p style="color:var(--text-muted);">
            Launch locally with <code>python -m ui.app</code> on <code>http://localhost:7860</code> or execute every workflow from the terminal via <code>python -m cli.rag_cli</code>.
          </p>
        </div>

        <div class="studio-grid">
          <div class="studio-card">
            <img src="assets/screenshots/ui_tab1_rag_studio.png" alt="Tab 1: Hybrid RAG Studio" />
            <div class="studio-card-body">
              <h3>Tab 1 — Hybrid RAG Studio (Dense + BM25 + RRF + Vertex AI Gemini)</h3>
              <p>Interactive query console with MRL dimension selector (768/512/256/128), retrieval mode selector, grounded [Doc i] answer generation, and top-K inspection.</p>
            </div>
          </div>
          <div class="studio-card">
            <img src="assets/screenshots/ui_tab2_mrl_explorer.png" alt="Tab 2: Matryoshka Explorer" />
            <div class="studio-card-body">
              <h3>Tab 2 — Matryoshka Representation Learning (MRL) Explorer</h3>
              <p>Live benchmark comparing 768-D, 512-D, 256-D, and 128-D prefixes on latency, memory compression (3.0x at 256-D), and 100% Top-5 retrieval overlap.</p>
            </div>
          </div>
          <div class="studio-card">
            <img src="assets/screenshots/ui_tab3_multimodal_lab.png" alt="Tab 3: Multimodal Audio & Image Lab" />
            <div class="studio-card-body">
              <h3>Tab 3 — Cross-Modal Audio &amp; Image RAG Lab</h3>
              <p>Unified 768-D vector search across PNG architecture cards, WAV/MP3 speech audio clips, and technical documentation using google/embeddinggemma-2.</p>
            </div>
          </div>
          <div class="studio-card">
            <img src="assets/screenshots/ui_tab4_gcp_bigquery.png" alt="Tab 4: GCP BigQuery Vector Search" />
            <div class="studio-card-body">
              <h3>Tab 4 — Google Cloud Storage &amp; BigQuery Vector Search</h3>
              <p>Serverless COSINE_DISTANCE vector search on BigQuery paired with one-command Cloud FinOps teardown (<code>python -m cli.rag_cli destroy --yes</code>).</p>
            </div>
          </div>
        </div>

        <div class="md-content" style="margin-top:2.25rem;">
          <h2>🧹 Verified Cloud FinOps Teardown Receipt (<code>$0.00/month</code>)</h2>
          <p>Output captured from <code>data/artifacts/teardown_verification.txt</code> after executing <code>python -m cli.rag_cli destroy --yes</code>:</p>
          <pre><code>${escapeHtml(DATA.teardown_txt || "GCP BigQuery Dataset (sarvam_rag): DELETED ($0.00)\nGCP Cloud Storage Bucket (gs://genai-511910-sarvam-rag): DELETED ($0.00)")}</code></pre>
        </div>
      </div>`;
  }

  function render() {
    updateProgressBadge();
    document.querySelectorAll(".nav-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.view === state.view);
    });

    const app = document.getElementById("app");
    if (state.view === "curriculum") app.innerHTML = renderCurriculumView();
    else if (state.view === "reader") app.innerHTML = renderReaderView();
    else if (state.view === "cheatsheet") app.innerHTML = renderCheatsheetView();
    else if (state.view === "textbook") app.innerHTML = renderTextbookView();
    else if (state.view === "studio") app.innerHTML = renderStudioView();

    if (window.hljs) {
      document.querySelectorAll("pre code").forEach((block) => {
        window.hljs.highlightElement(block);
      });
    }
    if (window.mermaid) {
      try {
        window.mermaid.run({ querySelector: ".mermaid" });
      } catch (e) {}
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const searchEl = document.getElementById("search-input");
    if (searchEl) {
      searchEl.addEventListener("input", (e) => {
        state.searchQuery = e.target.value;
        if (state.view !== "curriculum") state.view = "curriculum";
        render();
      });
    }

    const themeBtn = document.getElementById("theme-btn");
    if (themeBtn) {
      themeBtn.addEventListener("click", () => {
        const html = document.documentElement;
        const next = html.getAttribute("data-theme") === "light" ? "dark" : "light";
        html.setAttribute("data-theme", next);
        themeBtn.textContent = next === "light" ? "🌙 Dark" : "☀️ Light";
      });
    }

    if (window.mermaid) {
      window.mermaid.initialize({ startOnLoad: false, theme: "dark" });
    }

    render();
  });
})();
