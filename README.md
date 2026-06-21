# Latent and Explicit Narrative Representations for Multilingual Narrative Similarity

Master thesis · Ana Ciobanu · Alexandru Ioan Cuza University, Iași · 2026  
Supervisor: Diana Trandabăț

---

## Repository Structure

```
narrative_nlp/
├── dataset/
│   ├── romanian_narrative_similarity_dataset/   # Romanian MT dataset + translation cache
│   ├── aspects_cache_v1.json                    # Extracted aspects – verbose prose
│   ├── aspects_cache_v2.json                    # Extracted aspects – role-label steps
│   ├── aspects_cache_v3.json                    # Extracted aspects – compact phrases
│   ├── merged_aspects_cache_d_t_t.json          # Merged cache used by training notebook
│   ├── merged_aspects_cache.json                # Merged cache (alternative)
│   ├── synthetic_data_new.jsonl                 # Additional LLM-generated triples (1,097)
│   ├── tell_me_again_triplets.jsonl             # Tell-Me-Again! auxiliary triples (1,200)
│   └── G2_condition_predictions/               # Model predictions and embeddings
├── aspect_informativeness_analysis.ipynb
├── aspect_informativeness_vs_gold_aspect_labels.ipynb
├── aspect_informativeness_vs_gold_aspect_labels_bge_m.ipynb
├── aspect-aware-residual-solution-deterministic.ipynb
├── build-tell-me-again-training-data-redesigned.ipynb
├── clean_track_b_embedding_ensemble.ipynb
├── extract-aspects-with-llama-v1.py
├── extract-aspects-with-llama-v2.ipynb
├── extract-aspects-with-llama-v3.ipynb
├── generate_synth_new.ipynb
├── latent_explicit_narrative_similarity_conditions.ipynb  # Main training notebook
├── merge_aspects_cache_multiple_files.py
├── merge-aspects-cache.ipynb
├── multilingual-g2-en-ro-comparison.ipynb
├── qwen3-zero-shot-track-b-embeddings.ipynb
├── translate-to-romanian.ipynb
└── thesis_app/backend/
    ├── app.py                                   # Streamlit application
    └── extract_aspects_app.py                   # Aspect extraction as separate application
```

The SemEval-2026 Task 4 official data (`dev_track_a.jsonl`, `test_track_a.jsonl`,
`test_track_b.jsonl`, and organiser synthetic triples) is not included.
It can be obtained from: [https://github.com/narrative-similarity-task](https://github.com/narrative-similarity-task)

---

## Data Created in This Thesis

| File | Description |
|------|-------------|
| `synthetic_data_new.jsonl` | 1,097 triplets generated with `llama3.1:8b` |
| `tell_me_again_triplets.jsonl` | 1,200 hard-negative triplets from Tell-Me-Again! |
| `aspects_cache_v1/v2/v3.json` | Narrative aspects extracted with Llama 3.1 8B |
| `romanian_narrative_similarity_dataset/` | Romanian MT of all SemEval files via NLLB-200 |

---

## Main Notebooks

| Notebook | Purpose |
|----------|---------|
| `latent_explicit_narrative_similarity_conditions.ipynb` | All model conditions (Baseline, G2, aspect variants) |
| `multilingual-g2-en-ro-comparison.ipynb` | English vs. Romanian MT comparison |
| `clean_track_b_embedding_ensemble.ipynb` | G2 + Qwen3 embedding ensemble |
| `extract-aspects-with-llama-v1/v2/v3` | Aspect extraction pipelines |
| `translate-to-romanian.ipynb` | NLLB-200 translation pipeline |
| `build-tell-me-again-training-data-redesigned.ipynb` | Tell-Me-Again! triplet construction |
| `generate_synth_new.ipynb` | Additional synthetic data generation |
| `aspect_informativeness_analysis.ipynb` | Aspect informativeness analysis |
| `qwen3-zero-shot-track-b-embeddings.ipynb` | Qwen3 zero-shot Track B embeddings |

---

## Streamlit Application

```bash
pip install streamlit plotly pandas numpy scikit-learn scipy sentence-transformers
streamlit run thesis_app/backend/app.py
```

For the Live Aspect Extraction page, Ollama must be running locally:

```bash
ollama pull llama3.1:8b
ollama serve
```

---

## Reproducibility

All training experiments use seed 42. Before running the training notebook, set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

In a Kaggle/Colab notebook:

```python
%env CUBLAS_WORKSPACE_CONFIG=:4096:8
```

---

## Notes
Romanian dataset files are derived from SemEval-2026 Task 4 data and subject to the original task terms.