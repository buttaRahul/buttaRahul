## Rahul Aditya, ML Engineer

I build ML systems end to end: LLM fine-tuning and retrieval, GPU kernels and training infrastructure, and multimodal models.

### Public projects

| Project | What it is |
|---|---|
| [fused-rmsnorm-triton](https://github.com/buttaRahul/fused-rmsnorm-triton) | Fused residual-add + RMSNorm in Triton, forward and backward: 85–89% of peak memory bandwidth on an RTX 4060, 5–12× faster than eager PyTorch, verified end to end on Qwen3-0.6B |
| [Image-Caption-Generation](https://github.com/buttaRahul/Image-Caption-Generation) | Transformer captioner with spatial cross-attention over EfficientNetB0 regions; BLEU-4 23.4 (beam) on held-out Flickr8k |
| [clinical_triage_env](https://github.com/buttaRahul/clinical_triage_env) | OpenEnv RL environment for emergency-department triage with an asymmetric, clinically shaped reward |
| [multimodal_video_summarization](https://github.com/buttaRahul/multimodal_video_summarization) | Video summarisation that combines the transcript with descriptions of key frames before LLM summarisation |
| [Query-Assist](https://github.com/buttaRahul/Query-Assist) | Rule-based natural-language-to-SQL with speech input |

### Publications

- S. V. Boddu, **R. A. Butta**, G. Sannidhi, M. V. Yerakaraju. [QueryAssist: Multimodal Verbal Specifications to Structured Query Conversion Model Using Word Vector-Based Semantic Analysis](https://link.springer.com/chapter/10.1007/978-981-97-8193-5_6). InCITe 2024, *Lecture Notes in Electrical Engineering*, vol. 1278, Springer, 2025, pp. 61–69. Best Paper Award.

### Stack

**Modelling:** PyTorch, Hugging Face Transformers, PEFT (LoRA / QLoRA), XGBoost, scikit-learn
**Training and inference:** Triton, distributed data-parallel training, AWQ quantisation, NVIDIA Jetson
**Retrieval:** GraphRAG, RAG pipelines
**Data:** SQL, pandas

[LinkedIn](https://www.linkedin.com/in/rahul-aditya/) · [Kaggle](https://www.kaggle.com/adityarahul) · [Email](mailto:butta.rahul.aditya@gmail.com)
