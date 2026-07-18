## Dataset

The dataset is publicly available on the Hugging Face Hub:

**[Miladsaeedi70/scientific-multitask-instructions](https://huggingface.co/datasets/Miladsaeedi70/scientific-multitask-instructions)**

It contains **1,576 scientific instruction-following examples** across eight tasks, including scientific question answering, summarization, concept explanation, method comparison, technical simplification, bullet generation, data analysis, and code generation.

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