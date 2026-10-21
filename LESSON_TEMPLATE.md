# Lesson Template

Every lesson in this curriculum follows a 6-beat structure. Copy this template when creating a new lesson.

```markdown
# [Lesson Title]

> [One-line motto — the core idea that sticks]

**Type:** Build | Learn
**Languages:** Python
**Prerequisites:** [Lesson links or "None"]
**Time:** ~[30-90] minutes

---

## Learning Objectives

By the end of this lesson, you will be able to:

- [Specific, measurable skill 1]
- [Specific, measurable skill 2]
- [Specific, measurable skill 3]

---

## The Problem

[2-3 paragraphs. What breaks without this concept? Use a concrete scenario from Sarvam AI's multilingual voice/text developer documentation or production RAG workloads. Make the reader feel the pain before offering the solution.]

---

## The Concept

[The explanation. Rules:]
- Intuition first, math second — always
- Every formula gets a plain-English translation right below it
- At least one Mermaid diagram per concept (architecture, data flow, or geometry)
- Comparison tables when choosing between approaches
- Real numbers from our `google/embeddinggemma-2` + `sarvam-docs` benchmark, not hand-waving

---

## Build It

[Step-by-step from-scratch implementation in `code/main.py`. Rules:]
- Build from scratch first (pure Python + NumPy) so every operation is transparent
- Every code block is runnable and matches `code/main.py`
- Explain WHY before showing HOW
- Show expected output after each step

### Step 1: [First step]

[Explanation + code + expected output]

### Step 2: [Second step]

[Explanation + code + expected output]

---

## Use It

[Real-world application on the production Sarvam AI + `google/embeddinggemma-2` + GCP stack. Rules:]
- Show both the **CLI workflow** (`python -m cli.rag_cli ...`) and the **Interactive Web Studio workflow** (`http://localhost:8090`)
- Connect the from-scratch `code/main.py` implementation to the production module in `src/`
- Include expected metrics or terminal/UI verification steps

---

## Ship It

[What the learner walks away with. Every lesson produces at least two reusable artifacts:]
- **Prompt** (`outputs/prompt-*.md`) — A battle-tested system prompt or diagnostic prompt they can use immediately
- **Skill** (`outputs/skill-*.md`) — An AI agent skill they can install into Claude Code, Cursor, or Antigravity via `python3 scripts/install_skills.py`
- **Cloud FinOps / Teardown Check** — How to verify or clean up any cloud resources (`python -m cli.rag_cli destroy`)

---

## Exercises

1. **[Easy]** [Modify something from the lesson — 10-15 min]
2. **[Medium]** [Extend the implementation — 20-30 min]
3. **[Hard]** [Combine with a previous lesson or optimize for production — 30-60 min]

---

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| [Term 1] | [Jargon version] | [Plain English version] |
| [Term 2] | [Jargon version] | [Plain English version] |

---

## Further Reading

- [Paper, documentation, or production module reference — max 3-5 links]
```
