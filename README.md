# Russian LLM Pretraining and Supervised Fine-Tuning

<img width="1100" height="783" alt="image" src="https://github.com/user-attachments/assets/7ff98fa6-cd3c-4bd0-a8fc-002ed0c76c15" />


This repository contains a compact, reproducible study of two different stages in the language-model lifecycle. The accompanying notebook first trains a small decoder-only model from random initialization on Russian literary texts. It then separately fine-tunes the released **Qwen2.5-0.5B base model** on a Russian instruction-following dataset. The aim is to make the experimental choices, boundaries, and limitations explicit rather than to present a general-purpose Russian LLM.

## What the notebook does

The **pretraining section** clones the upstream `JoannaBy/RussianNovels` collection at run time, detects exact duplicate documents, filters non-Cyrillic and separator-like lines, normalizes whitespace and excessive repeated punctuation, and creates fixed-size text chunks. It trains a 3,000-token ByteLevel BPE tokenizer and a Llama-style decoder-only model with roughly 132 million parameters. A fixed 95/5 chunk-level split supplies validation loss and perplexity during training.

The **SFT section** starts from `Qwen/Qwen2.5-0.5B`, which is a base model rather than a ready-made chat assistant. It converts examples from `d0rj/alpaca-cleaned-ru` into system/user/assistant conversations, selects a fixed shuffled subset of 5,000 examples, and splits it 90/10 into training and validation data. It compares qualitative answers generated from the same chat-template prompt before and after fine-tuning, while logging held-out SFT loss.

## Scope and limitations

This is an **educational training experiment**, not a benchmark, a production assistant, or a claim that the model has learned reliable factual knowledge. The literary corpus source itself describes the collection as material for stylometric experiments and not as a balanced benchmark [1]. The pretraining validation split is randomly drawn from the same source collection. The SFT validation split is drawn from one instruction dataset. Neither result supports conclusions about general language ability, factual accuracy, safety, or real-world assistant quality.

The original workflow was developed in Google Colab with a high-memory GPU. The full pretraining configuration trains a roughly 132M-parameter model for 10 epochs at a 512-token sequence limit; it is therefore compute-intensive. The notebook includes a smoke-test switch for pipeline validation, but a smoke test is not meaningful training.

## Data and model access

No text corpus, instruction data, model weights, checkpoints, or Hugging Face cache is committed to this repository. The notebook retrieves upstream resources programmatically:

- `JoannaBy/RussianNovels` is used only by cloning its upstream repository at run time. This project does not redistribute it. Review the upstream source, its terms, and the copyright status of individual works before use beyond this experiment [1].
- `d0rj/alpaca-cleaned-ru` is loaded from the Hugging Face Hub. Its dataset card identifies it as a Russian translation of `yahma/alpaca-cleaned`, reports 51,760 rows, and lists the CC-BY-4.0 license [2].
- `Qwen/Qwen2.5-0.5B` is loaded from the Hugging Face Hub. The model card identifies it as a 0.5B-parameter causal base model under Apache-2.0 and advises post-training before conversational use [3].

## Repository structure

```text
.
├── russian_llm_pretraining_and_sft.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the notebook

Create a clean Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `notebooks/russian_llm_pretraining_and_sft.ipynb` and run the cells in order. The project root should be the Jupyter server's working directory. On Google Colab, clone this repository first and change into the cloned directory before opening the notebook.

Before a full run, choose one of the two settings in the pretraining cell:

```python
RUN_FULL_PRETRAINING = True   # Full documented configuration: 10 epochs
# RUN_FULL_PRETRAINING = False  # Short smoke test only
```

Training artifacts are written to `artifacts/` and are intentionally ignored by Git. Do not commit downloaded corpora, checkpoints, or model weights.

## Suggested GitHub description

> Reproducible study of Russian LLM pretraining from scratch and supervised fine-tuning of Qwen2.5-0.5B, with explicit data provenance, held-out loss monitoring, and qualitative generation analysis.
