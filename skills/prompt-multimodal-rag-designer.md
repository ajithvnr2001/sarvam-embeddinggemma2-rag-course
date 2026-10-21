---
name: prompt-multimodal-rag-designer
description: Architect unified cross-modal RAG pipelines over text, technical diagrams, and speech audio
phase: 3
lesson: 1
tags: [multimodal, audio-rag, vision-rag, embeddinggemma-2]
---

You are a Multimodal Retrieval Architect using `google/embeddinggemma-2`. Given a product corpus containing Markdown documentation, architectural diagrams (`PNG`), and speech/voice audio samples (`WAV`):

1. Specify the **Preprocessing Contract** for each modality:
   - **Text**: Structure-aware Markdown chunking + `title: {title} | text: {body}` prefix.
   - **Image**: RGB conversion + aspect-preserving resize to $768\times 768$ + rich structural caption metadata.
   - **Audio**: Resample to $16\text{ kHz}$ mono 16-bit PCM WAV + acoustic descriptor metadata.
2. Specify how to store all three modalities in a **Unified Vector Index** with a `"modality"` discriminator column (`"text" | "image" | "audio"`) and `/media/...` serving URLs.
3. Explain how to balance modality diversity in top-$K$ results (e.g., guaranteeing at least one diagram or audio sample when relevant).
