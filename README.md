# Scientific LLM Fine-Tuning with LoRA SFT and GRPO

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#run-the-local-demo)
[![Base model](https://img.shields.io/badge/Base-SmolLM2--135M--Instruct-orange)](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/Miladsaeedi70/scientific-multitask-instructions)
[![SFT adapter](https://img.shields.io/badge/Hugging%20Face-SFT%20adapter-yellow)](https://huggingface.co/Miladsaeedi70/smollm2-135m-scientific-sft-lora)
[![GRPO adapter](https://img.shields.io/badge/Hugging%20Face-GRPO%20adapter-yellow)](https://huggingface.co/Miladsaeedi70/smollm2-135m-scientific-grpo-lora)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

A reproducible scientific language-model project that adapts
`HuggingFaceTB/SmolLM2-135M-Instruct` in two stages:

1. **LoRA supervised fine-tuning (SFT)** on 1,576 multi-task scientific examples.
2. **Group Relative Policy Optimization (GRPO)** using a frozen, groupwise
   LLM ranking judge.

The central technical contribution is a groupwise reward design that prevents
the zero-advantage problem caused by identical independent judge scores.

> The selected checkpoint is the **25-step GRPO pilot**. The 100-step
> checkpoint was effectively tied under direct pairwise judging, while the
> 25-step model achieved slightly stronger overall ROUGE-L.

## Key results

Weighted over the untouched 158-example test split:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Mean words |
|---|---:|---:|---:|---:|---:|
| Base | 0.2220 | 0.0428 | 0.1501 | 0.1734 | 82.0 |
| SFT | 0.2858 | 0.0627 | 0.2048 | 0.2270 | 39.3 |
| **GRPO pilot (25 steps)** | **0.2885** | **0.0643** | **0.2088** | **0.2323** | 39.2 |
| GRPO final (100 steps) | 0.2832 | 0.0620 | 0.2075 | 0.2301 | 37.5 |

The 25-step and 100-step checkpoints received weighted direct judge scores of
**0.4968** and **0.5032**, respectively. Their difference was below the 0.01
model-selection threshold.

![Overall test-set comparison](assets/overall_rouge_comparison.png)

## Method

```mermaid
flowchart LR
    A[Scientific multi-task dataset] --> B[LoRA SFT]
    B --> C[SFT adapter]
    C --> D[Generate 4 candidates per prompt]
    D --> E[Frozen groupwise LLM judge]
    E --> F[Strict relative ranking rewards]
    F --> G[GRPO LoRA optimization]
    G --> H[25-step and 100-step checkpoints]
    H --> I[ROUGE + reversed-order pairwise evaluation]
    I --> J[Select 25-step checkpoint]
```

### Why groupwise ranking?

Independent numerical judging frequently assigned the same score to all four
candidate responses. GRPO then produced zero relative advantages and no useful
learning signal.

The replacement reward presents all four candidates to the judge together and
requires a strict ranking. For four candidates, ranks are mapped to:

```text
1.00, 0.67, 0.33, 0.00
```

This produced a stable reward mean of 0.5 and nonzero within-group variance.

## Local demo

The repository includes a Gradio demo that runs locally on the user's computer.
It does not require deployment to a Hugging Face Space.

The first run downloads:

- The public base model:
  `HuggingFaceTB/SmolLM2-135M-Instruct`
- The public selected GRPO LoRA adapter:
  `Miladsaeedi70/smollm2-135m-scientific-grpo-lora`

The downloaded files are cached by Hugging Face for later runs.

> **Required before sharing this repository:** the selected GRPO adapter must be
> public. If the adapter is private, other users will not be able to run the
> demo without authentication.

## Run the local demo

### 1. Clone the repository

Replace `YOUR_GITHUB_USERNAME` with the GitHub username that owns this project.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/scientific-llm-finetuning.git
cd scientific-llm-finetuning
```

### 2. Create a virtual environment

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the demo

```bash
python app.py
```

The browser should open automatically at:

```text
http://127.0.0.1:7860
```

Stop the demo with `Ctrl+C` in the terminal.

### Run with a different adapter

The application reads `BASE_MODEL_ID` and `ADAPTER_ID` from environment
variables when they are provided.

#### macOS or Linux

```bash
ADAPTER_ID=/path/to/local/adapter python app.py
```

or:

```bash
ADAPTER_ID=your-account/your-public-adapter python app.py
```

#### Windows PowerShell

```powershell
$env:ADAPTER_ID="C:\path\to\local\adapter"
python app.py
```

This allows the same demo to run with a local adapter without changing
`app.py`.

## Dataset

The dataset is publicly available on the Hugging Face Hub:

**[Miladsaeedi70/scientific-multitask-instructions](https://huggingface.co/datasets/Miladsaeedi70/scientific-multitask-instructions)**

It contains **1,576 scientific instruction-following examples** across eight
tasks:

- Scientific question answering
- Summarization
- Concept explanation
- Method comparison
- Technical simplification
- Bullet generation
- Data analysis
- Code generation

### Dataset splits

| Split | Examples |
|---|---:|
| Train | 1,260 |
| Validation | 158 |
| Test | 158 |
| **Total** | **1,576** |

### Dataset fields

Each example contains:

- `id`: unique example identifier
- `category`: scientific subject category
- `task`: instruction task
- `difficulty`: difficulty level
- `messages`: system, user, and assistant messages in conversational format

### Load the dataset

```python
from datasets import load_dataset

dataset = load_dataset(
    "Miladsaeedi70/scientific-multitask-instructions"
)

print(dataset)
print(dataset["train"][0])
```

The dataset uses grouped train, validation, and test splitting to reduce prompt
leakage across splits.

## Repository structure

```text
scientific-llm-finetuning/
├── assets/
├── data/
├── notebooks/
│   ├── 01_dataset_creation.ipynb
│   ├── 02_tokenizer_analysis.ipynb
│   ├── 03_lora_finetuning_and_evaluation.ipynb
│   └── 04_grpo_experiment.ipynb
├── results/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

### Main files

- `app.py`: loads the base model and selected LoRA adapter and starts the local
  Gradio interface.
- `requirements.txt`: lists the packages needed to run the demo.
- `notebooks/`: contains the dataset, tokenizer, SFT, GRPO, and evaluation
  workflow.
- `results/`: contains compact evaluation summaries.
- `assets/`: contains figures used in this README.

Model weights are not stored in GitHub. The demo downloads them from the
Hugging Face Hub, keeping the repository lightweight.

## Reproduce the experiments

1. Install the notebook dependencies used in the relevant notebook.
2. Load the public dataset:

   ```python
   from datasets import load_dataset

   dataset = load_dataset(
       "Miladsaeedi70/scientific-multitask-instructions"
   )
   ```

3. Run `notebooks/02_tokenizer_analysis.ipynb`.
4. Run `notebooks/03_lora_finetuning_and_evaluation.ipynb`.
5. Run `notebooks/04_grpo_experiment.ipynb`.
6. Verify positive `reward_std` and low `frac_reward_zero_std`.
7. Compare SFT, 25-step GRPO, and 100-step GRPO on the untouched test split.

## Troubleshooting

### The adapter repository cannot be found

Confirm that this repository exists and is public:

```text
Miladsaeedi70/smollm2-135m-scientific-grpo-lora
```

Alternatively, set `ADAPTER_ID` to a local adapter folder containing at least:

```text
adapter_config.json
adapter_model.safetensors
```

### The demo is slow

The model can run on CPU, but generation may be slower. The app automatically
uses CUDA or Apple MPS when available and otherwise falls back to CPU.

### Port 7860 is already in use

Stop the other Gradio process or change the launch configuration in `app.py`.

### Installation problems

Use Python 3.10, 3.11, or 3.12 in a clean virtual environment, then reinstall:

```bash
python -m pip install --upgrade pip
pip install --force-reinstall -r requirements.txt
```

## Evaluation cautions

- ROUGE measures reference overlap, not factual correctness.
- The same judge family contributed to training and evaluation, so pairwise
  results are not fully independent.
- The base model has only 135M parameters.
- Task-level performance varies; code generation and technical simplification
  did not improve consistently.
- Outputs must be independently checked for high-stakes scientific use.

## Roadmap

- [x] Create the multi-task scientific dataset
- [x] Publish the dataset on Hugging Face
- [x] Complete LoRA SFT
- [x] Complete groupwise-judge GRPO experiments
- [x] Add a locally runnable Gradio demo
- [ ] Make the selected GRPO adapter public
- [ ] Add evaluation with a second independent judge model
- [ ] Add bootstrap confidence intervals
- [ ] Add human review of a stratified response sample
