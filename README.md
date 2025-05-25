# GenAI24‑NIST Text‑to‑Text Pipeline 📝

Comprehensive solution for the **NIST GenAI24 Text‑to‑Text** task.

---

## 📁 Repository Layout

```
repo/
├── updated_a2c.py                # Generator 1 – Actor‑Critic RL
├── generation_updated.ipynb      # Generator 2 – Sampling & fluency
│
├── Active Learning.ipynb         # Discriminator 1 – GPT‑assisted reviewer
├── gen.py                        # Discriminator 2 – RoBERTa train / test
├── pred.py                       # Helper for gen.py inference
│
├── xml_cleanup.ipynb             # Utility: clean NIST XML (optional)
└── outputs/
    ├── summaries.xml             # Final XML submission
    └── results.csv               # AI/Human predictions
```

---

## ✨ At a Glance

| Category        | Module(s)                           | Highlights                                   |
|-----------------|-------------------------------------|----------------------------------------------|
| **Generator 1** | `updated_a2c.py`                    | Actor‑Critic RL, human‑like enhancements, XML writer |
| **Generator 2** | `generation_updated.ipynb`          | Top‑k / Top‑p / Temp sampling exploration    |
| **Discriminator 1** | `Active Learning.ipynb`         | GPT‑4 review & manual filtering              |
| **Discriminator 2** | `gen.py` (+ `pred.py`)          | RoBERTa classifier (train & inference)       |

---

## 1 · Generators

### 🔹 1.1 `updated_a2c.py` — Actor‑Critic RL Generator

- Loads SGML topics & article files  
- Trains GPT‑2 policy/value networks with **A2C**  
- Adds rhetorical devices, back‑translation, paraphrasing  
- Uses internal RoBERTa reward for style compliance  
- Writes **`summaries.xml`** and **`results.csv`**

```bash
export OPENAI_API_KEY=sk-...
python updated_a2c.py
```

Key output files:

| File            | Purpose                |
|-----------------|------------------------|
| `summaries.xml` | NIST‑formatted summaries |
| `results.csv`   | AI/Human flag per topic |
| `training.log`  | RL reward & loss        |

---

### 🔹 1.2 `generation_updated.ipynb` — Interactive Generator Notebook

This Jupyter notebook is an **interactive wrapper** around the core pipeline.  
It lets you:

- Load the same article/topic dataset used by `updated_a2c.py`
- Tweak generation parameters (e.g., max tokens, number of chunks, paraphrasing flags)
- Manually inspect intermediate chunks and summaries
- Export a `summaries.xml` preview without rerunning full RL training

Open it in JupyterLab / VS Code, run the cells sequentially, and adjust the cells marked **“🔧 Parameters”** to experiment with different settings.

Use it for **exploratory runs, debugging and quick demos**, rather than head‑less batch generation.
---

## 2 · Discriminators

### 🔸 2.1 `Active Learning.ipynb` — GPT‑Assisted Reviewer

A notebook that:

1. Loads generator outputs  
2. Calls **GPT‑4** for coherence, redundancy, toxicity scores  
3. Flags low‑quality summaries for human attention  
4. Optionally augments training data

Acts as a **semi‑automatic discriminator**.

---

### 🔸 2.2 `gen.py` & `pred.py` — RoBERTa Classifier

#### Train

```bash
python gen.py   --mode train   --data_path ./train_data.csv   --model_save_path ./Roberta
```

`train_data.csv` columns: `text,label` (0 = Human, 1 = AI)

#### Infer

```bash
python gen.py   --mode test   --model_save_path ./Roberta   --input_directory ./txts   --results_file ./results.csv
```

Produces **`results.csv`** with confidence scores.

---

## ⚙️ Environment Setup

```bash
pip install torch transformers datasets evaluate scikit-learn openai             spacy nltk beautifulsoup4 rouge-score
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
```

---

## 📤 Final Outputs

| File                 | Description                        |
|----------------------|------------------------------------|
| `outputs/summaries.xml` | Submission for NIST evaluation  |
| `outputs/results.csv`   | AI/Human discrimination results |
| `training.log`          | RL training trace               |
