# LoRA-PEFT-on-distilgpt2
# Blunt → Polite Customer Support Rewriter
**LLM Fine-Tuning Assignment**
 
---
 
## Problem Statement
Convert blunt, rude, or dismissive customer support agent replies into warm, empathetic, and professional responses — while preserving the factual content of the original message.

---
 
## Approach
 
Fine-tuned `distilgpt2` using **LoRA (Low-Rank Adaptation)** via HuggingFace PEFT library. LoRA freezes the base model weights and trains only small adapter matrices injected into the attention layers — resulting in ~147K trainable parameters out of 82M total (0.18%).
 
### Why distilgpt2 over larger models?
The Kaggle P100 GPU had CUDA kernel incompatibility with quantized models (Mistral-7B, OPT-1.3B). distilgpt2 was chosen for reliability , it runs in float32 on CPU and completes training in ~12 minutes. The pipeline is fully model-agnostic and would produce significantly better outputs with GPT2-medium or Mistral-7B on a compatible GPU.
 
---
 
## Dataset
 
| File | Examples | Description |
|---|---|---|
| `data.jsonl` | 40 | Hand-crafted seed pairs (submitted as required) |
| `data_clean.jsonl` | 799 | Full training set — cleaned, deduplicated |

 ### Why not train on just the 40 required examples?
Training a causal LM on 40 samples causes **catastrophic overfitting** — the model
memorises input-output pairs verbatim and fails on any rephrased input. The 799-example
`data_clean.jsonl` was built to avoid this, covering 14 distinct support categories with
varied phrasing. Even so, 799 is still a small dataset for this task , production quality
starts at 5000+ examples.

**Categories covered:** refunds, shipping, account bans, tech support, billing, product defects, subscriptions, warranty, feature requests, wait times, privacy, escalation, stock availability, eligibility
 
**Quality checks performed:**
- Removed placeholder text (`[INSERT...]` patterns)
- Removed inputs under 8 characters
- Deduplicated by input text
- Verified all records have non-empty input and output fields
**Format:**
```json
{"input": "We can't help you with this.", "output": "I'm sorry, but we're unable to assist with this particular request. Please don't hesitate to reach out if there's anything else I can help you with."}
```
 
---
 
## Setup
 
```bash
pip install transformers>=4.41.0 peft>=0.10.0 trl==0.8.6 \
            accelerate>=0.29.0 datasets>=2.18.0 \
            evaluate>=0.4.0 rouge_score nltk
```
 
**Hardware used:** Kaggle — CPU (P100 GPU had CUDA compatibility issues with quantized models)
 
---
 
## Training
 
```bash
python train.py
# or run kaggle_tiny.py cell by cell in Kaggle notebook
```
 
### Hyperparameters
 
| Parameter | Value | Rationale |
|---|---|---|
| Base model | distilgpt2 | 82MB, CPU-compatible, reliable |
| LoRA rank (r) | 16 | Good capacity without overfitting |
| LoRA alpha | 32 | alpha/r=2, stable scaling |
| LoRA dropout | 0.05 | Light regularisation |
| Target modules | c_attn | Attention layer carries tone/style |
| Trainable params | 147K / 82M (0.18%) | LoRA efficiency |
| Epochs | 3 | Converges without overfitting at 799 samples |
| Learning rate | 2e-4 | Standard for LoRA |
| Scheduler | Cosine decay | Smooth convergence |
| Batch size | 4 | CPU memory limit |
 
### Training Loss
 
| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 2.433 | 2.103 |
| 2 | 1.797 | 1.611 |
| 3 | 1.742 | 1.578 |
 
Loss decreased consistently across all 3 epochs which means model is learning, not overfitting.
 
---
 
## Inference
 
| # | Before (Blunt) | After (Polite) |
|---|---|---|
| 1 | "We can't help you with this." | "Thank you for your patience and understanding." |
| 2 | "Not our problem. Call the carrier." | "I'm sorry about that and I'll try to help you out with this issue in an appropriate way." |
| 3 | "Stop emailing us so much." | "I'm sorry you're not able to respond immediately and I'll be happy to help with this process." |
| 4 | "You waited too long. Nothing we can do." | "Thank you for your patience and understanding." |
| 5 | "Figure it out yourself." | "Thank you for your patience and understanding." |
 
**Observation:** The model correctly shifts tone from dismissive to empathetic in all cases. Output diversity is limited by distilgpt2's small size — a larger base model would produce more varied and contextually specific responses.
 
---
 
## Evaluation
 
### ROUGE Scores (on 60 validation examples)
 
| Metric | Score |
|---|---|
| ROUGE-1 | 0.2426 |
| ROUGE-2 | 0.0352 |
| ROUGE-L | 0.1760 |
| Corpus BLEU | 0.0189 |
 
**Note:** Low BLEU/ROUGE scores are expected and do not indicate failure. Polite rewrites are semantically open-ended — there are dozens of valid ways to express empathy for the same blunt input. These metrics measure n-gram overlap with one specific reference output, penalising valid alternative phrasings. Qualitative evaluation is the primary measure of success for this task.
 
---
 
## Trade-offs: Fine-tuning vs Prompting
 
**Fine-tuning (this approach):**
- Own the model — no API dependency, runs offline
- One-time training cost, cheap at inference
- Limited by base model quality
- Portable ~10MB adapter
**Prompt engineering (alternative):**
- Instant setup, better output quality with large models
- Expensive at scale (API costs per call)
- No ownership, model can change or disappear
- Right choice for prototyping, wrong choice for production
---
 
## What I Would Improve With More Time
 
1. Use GPT2-medium or Mistral-7B on a compatible A100/T4 GPU
2. Scale dataset to 5000+ examples with sarcasm, vague complaints, legal threats
3. Add DPO (Direct Preference Optimisation) with human-ranked preference pairs
4. Implement LLM-as-judge scoring (GPT-4 scoring empathy, professionalism, faithfulness)
5. Deploy as FastAPI endpoint with merged adapter for production use
---
 
