## Rahul Aditya, ML Engineer

I build and ship ML systems: LLM fine-tuning and retrieval, clinical prediction on claims data, and on-device multimodal inference.

**Now:** ML Engineer at Zynix AI, working on clinical risk models, healthcare fraud and waste detection, and voice agents.

### Selected work

- **Healthcare fraud and waste detection**: a five-layer detection pipeline for durable medical equipment claims: rules, peer baselines, an anomaly ensemble, graph features, and positive-unlabeled learning on a confirmed fraud scheme. Validated against the OIG exclusion list.
- **Clinical risk models**: 30-day admission, ER-visit and readmission models, plus disease-progression models, with thresholds sized to care-team capacity.
- **Multilingual graph RAG**: productionised GFM-RAG, a GNN retriever over a document knowledge graph, across 12 Indian languages, beating a ColBERTv2 baseline on multi-hop retrieval.
- **Edge multimodal agents**: an on-device voice agent on an NVIDIA Jetson AGX Orin, with faster-whisper, LLaVA-1.5 7B quantised to 4-bit with AWQ, Chatterbox TTS, and attention-sink KV caching.
- **Video understanding**: fine-tuned VideoMAE for live cricket event detection in a production ensemble serving Disney+ Hotstar.
- **LLM fine-tuning**: QLoRA and LoRA fine-tunes of 7B models for structured generation (natural language → JSONLogic rules) and domain chatbots.

### Public projects

| Project | What it is |
|---|---|
| [Image-Caption-Generation](https://github.com/buttaRahul/Image-Caption-Generation) | Transformer captioner with spatial cross-attention over EfficientNetB0 regions; BLEU-4 23.4 (beam) on held-out Flickr8k |
| [clinical_triage_env](https://github.com/buttaRahul/clinical_triage_env) | OpenEnv RL environment for emergency-department triage with an asymmetric, clinically shaped reward |
| [multimodal_video_summarization](https://github.com/buttaRahul/multimodal_video_summarization) | Video summarisation that combines the transcript with descriptions of key frames before LLM summarisation |
| [Query-Assist](https://github.com/buttaRahul/Query-Assist) | Rule-based natural-language-to-SQL with speech input |

### Publications

- S. V. Boddu, **R. A. Butta**, G. Sannidhi, M. V. Yerakaraju. [QueryAssist: Multimodal Verbal Specifications to Structured Query Conversion Model Using Word Vector-Based Semantic Analysis](https://link.springer.com/chapter/10.1007/978-981-97-8193-5_6). InCITe 2024, *Lecture Notes in Electrical Engineering*, vol. 1278, Springer, 2025, pp. 61–69.

### Stack

**Modelling:** PyTorch, Hugging Face Transformers, PEFT (LoRA / QLoRA), XGBoost, scikit-learn
**Training and inference:** DDP and 3D parallelism, AutoAWQ, faster-whisper, NVIDIA Jetson
**Retrieval:** GraphRAG (GFM-RAG), RAG pipelines
**Data:** SQL, pandas

[LinkedIn](https://www.linkedin.com/in/rahul-aditya/) · [Kaggle](https://www.kaggle.com/adityarahul) · [Email](mailto:butta.rahul.aditya@gmail.com)
