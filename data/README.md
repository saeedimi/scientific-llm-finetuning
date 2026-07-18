# Dataset

The training dataset is intentionally not included in this repository.

Expected Hugging Face `DatasetDict` structure:

```text
train:      1,260 examples
validation:   158 examples
test:         158 examples
```

Expected columns:

```text
id
category
task
difficulty
messages
```

Before publishing the dataset, verify:

- No private or identifying information
- No API keys or secrets
- No restricted copyrighted passages
- Clear documentation of synthetic and human-written content
- Duplicate and near-duplicate checks
- Group-aware splitting to prevent prompt leakage
- A dataset card with provenance, license, limitations, and intended use
