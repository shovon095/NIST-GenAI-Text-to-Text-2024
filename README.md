# GenAI24‑NIST Text‑to‑Text Pipeline 📝

Comprehensive solution for the **NIST GenAI24 Text‑to‑Text** task.

---

## 📁 Repository Layout

```
repo/Generator
├── updated_a2c.py               # Generator 1 – Actor-Critic RL
├── generation_updated.ipynb     # Generator 2 – Paraphrasing, Role-aware Back-Style
├── xml_cleanup.ipynb            # Cleans and verifies NIST XML output
└── outputs/
    └── summaries.xml            # Final XML-formatted summaries for NIST

repo/Discriminator
├── Active Learning.ipynb        # Discriminator 1 – Self-training with human in the loop using BERT and RoBERTa
├── gen.py                       # Discriminator 2 – RoBERTa trainer & evaluator
├── pred.py                      # Inference helper for gen.py
├── discriminator_format.ipynb  # Visualization and format testing for discrimination outputs
└── outputs/
    └── results.csv              # AI/Human predictions
```


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

| Technique                        | What it does                                                                                                                            |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Back-translation pipeline**    | Sends each draft summary through an intermediate language (e.g. French → English) to introduce natural word order changes and idioms.   |
| **GPT-based paraphrasing**       | Calls `gpt-3.5-turbo` to rephrase sentences with configurable temperature and top-p.                                                    |
| **Role-aware rewriting**         | Extracts key **Subjects / Verbs / Objects** with spaCy, then injects them into prompts so the paraphrased text preserves factual roles. |
| **Rhetorical & hedge additions** | Adds analogies, hedge words (“perhaps”, “might”) and temporal markers to mimic human narrative flow.                                    |


Open it in JupyterLab / VS Code, run the cells sequentially, and adjust the cells marked **“🔧 Parameters”** to experiment with different settings.

Use it for **exploratory runs, debugging and quick demos**, rather than head‑less batch generation.
---

## 2 · Discriminators

### 🔸 2.1 `Self training.ipynb` 

### 🔸 2.1 `Active Learning.ipynb` — Manual Discriminator Feedback Interface

This notebook implements a **manual feedback loop** for reviewing and improving AI/Human classification results from multiple models (e.g., BERT and RoBERTa).

Features:

- Loads and aligns predictions from two different models  
- Displays **confidence scores and labels** for each summary  
- Highlights disagreements or low-confidence predictions  
- Provides an interface to **manually label or confirm** the final decision  
- Optionally records decisions to improve future training data

This tool helps simulate an **active learning process** where human reviewers guide the refinement of discriminator labels.

> No model training or inference is done here — it is a **human-in-the-loop evaluator** powered by precomputed model outputs.


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
