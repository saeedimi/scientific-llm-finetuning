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

The central technical contribution is a groupwise reward design that addresses
the zero-advantage problem caused by identical independent judge scores.

> **Selected model:** the 25-step GRPO pilot. The 100-step checkpoint was
> effectively tied under direct pairwise judging, while the 25-step checkpoint
> achieved slightly stronger overall ROUGE-L.

## Main finding

Supervised fine-tuning produced the largest improvement over the base model.
GRPO then provided a smaller additional gain, with the strongest checkpoint
appearing after only 25 optimization steps.

Extending GRPO to 100 steps did not produce a meaningful overall improvement.
For this model and dataset, longer preference optimization was not
automatically better.

## Key results

Weighted over the untouched 158-example test split:

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | Mean words |
|---|---:|---:|---:|---:|---:|
| Base | 0.2220 | 0.0428 | 0.1501 | 0.1734 | 82.0 |
| SFT | 0.2858 | 0.0627 | 0.2048 | 0.2270 | 39.3 |
| **GRPO pilot (25 steps)** | **0.2885** | **0.0643** | **0.2088** | **0.2323** | 39.2 |
| GRPO final (100 steps) | 0.2832 | 0.0620 | 0.2075 | 0.2301 | 37.5 |

The 25-step and 100-step checkpoints received weighted direct judge scores of
**0.4968** and **0.5032**, respectively. Their difference of **0.0063** was
below the predefined **0.01** model-selection threshold.

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

## LLM-as-a-Judge reward design

GRPO used the frozen model `Qwen/Qwen2.5-0.5B-Instruct` as a groupwise
LLM judge.

For each training prompt:

1. The policy generated four candidate responses.
2. The judge received:
   - the task category,
   - the original prompt,
   - the reference answer,
   - and all four candidate responses.
3. The judge returned a strict best-to-worst ranking with no ties.
4. The ranking was converted to relative rewards:

| Rank | Reward |
|---:|---:|
| 1 | 1.00 |
| 2 | 0.67 |
| 3 | 0.33 |
| 4 | 0.00 |

Candidate presentation order was shuffled to reduce position bias. If the
judge returned an invalid ranking, all candidates received a neutral reward
of `0.5`.

The judge model remained frozen throughout training. Only the policy's LoRA
parameters were updated.

### Why independent scoring failed

The first reward implementation evaluated each candidate independently using
a numerical score. The judge often assigned the same score to every response
within a generation group.

Because GRPO learns from relative performance inside each group, identical
rewards produced zero reward variance and therefore zero useful advantage.

The groupwise ranking formulation forced relative differentiation among the
four candidates and substantially reduced the zero-advantage problem. During
the validated pilot run, reward variance remained nonzero for nearly all
training groups.

## Experiment configuration

| Component | Configuration |
|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Training dataset | `Miladsaeedi70/scientific-multitask-instructions` |
| SFT method | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| LoRA target modules | All linear layers |
| Maximum SFT sequence length | 512 |
| SFT epochs | 3 |
| Effective SFT batch size | 16 |
| SFT optimizer steps | 237 |
| GRPO starting policy | Scientific SFT LoRA adapter |
| Judge model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Candidates per prompt | 4 |
| Maximum prompt length | 384 |
| Maximum completion length | 128 |
| GRPO pilot | 25 steps, KL beta `0.0` |
| Longer GRPO run | 100 steps, KL beta `0.001` |
| Random seed | 42 |

## Checkpoint evaluation and selection

The Base, SFT, 25-step GRPO, and 100-step GRPO checkpoints were evaluated on
the untouched 158-example test split.

Evaluation included:

- ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum
- Mean response length
- Task-level performance
- Pairwise LLM judging

For pairwise evaluation, candidate order was reversed and evaluated again to
reduce A/B position bias.

The 25-step and 100-step checkpoints received judge scores of `0.4968` and
`0.5032`. The difference of `0.0063` was below the predefined `0.01`
selection threshold, so ROUGE-L was used as the tiebreaker.

The 25-step checkpoint was selected because its ROUGE-L score was `0.2088`,
compared with `0.2075` for the 100-step checkpoint.

## Public artifacts

| Artifact | Repository |
|---|---|
| Dataset | [`Miladsaeedi70/scientific-multitask-instructions`](https://huggingface.co/datasets/Miladsaeedi70/scientific-multitask-instructions) |
| SFT adapter | [`Miladsaeedi70/smollm2-135m-scientific-sft-lora`](https://huggingface.co/Miladsaeedi70/smollm2-135m-scientific-sft-lora) |
| Selected GRPO adapter | [`Miladsaeedi70/smollm2-135m-scientific-grpo-lora`](https://huggingface.co/Miladsaeedi70/smollm2-135m-scientific-grpo-lora) |

## Local demo

The repository includes a Gradio demo that runs locally on the user's
computer. It does not require deployment to a Hugging Face Space.

The first run downloads:

- The public base model:
  `HuggingFaceTB/SmolLM2-135M-Instruct`
- The public selected GRPO LoRA adapter:
  `Miladsaeedi70/smollm2-135m-scientific-grpo-lora`

The downloaded files are cached by Hugging Face for later runs.

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

## Limitations

- The base model contains only 135M parameters.
- Model outputs may contain scientific inaccuracies or unsupported claims.
- ROUGE measures lexical overlap and does not directly measure factuality.
- An LLM judge can introduce position bias, verbosity bias, and model-family
  preferences.
- The reference answer may influence judge rankings even when another valid
  response uses different wording.
- Reversed candidate ordering reduces but does not eliminate judge bias.
- The judge used for reward generation is not a substitute for expert human
  evaluation.
- Results are based primarily on one random seed.
- Performance varies by task; code generation and technical simplification
  did not consistently improve after GRPO.
- The model should not be used as the sole source for high-stakes scientific,
  medical, legal, environmental, or engineering decisions.

Important scientific claims should be independently verified.

## Project status

- [x] Create and validate the scientific instruction dataset
- [x] Publish the dataset on Hugging Face
- [x] Complete tokenizer analysis
- [x] Complete LoRA supervised fine-tuning
- [x] Complete 25-step and 100-step GRPO experiments
- [x] Select the best checkpoint
- [x] Publish the selected GRPO adapter
- [x] Add a locally runnable Gradio demo
- [ ] Add evaluation with an independent judge family
- [ ] Add human evaluation
- [ ] Add bootstrap confidence intervals
- [ ] Repeat GRPO with multiple random seeds

## License

This project is released under the Apache 2.0 license. See [LICENSE](LICENSE).

The base model is distributed separately under its own license.
