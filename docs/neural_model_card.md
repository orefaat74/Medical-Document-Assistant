# Neural model card — Phase 3 TextCNN

- Architecture: learned word embedding + padding-aware multi-window 1D-CNN.
- Checkpoint: `models/phase3/text_cnn.pt` (state dictionaries and primitive metadata).
- Selected trial / parameters: `trial_08` / `{'learning_rate': 0.003, 'batch_size': 32, 'dropout': 0.5}`.
- Validation macro F1: **0.9972**.
- Test accuracy / macro F1: **0.9854 / 0.9876**.
- Final fit: **2129** train/validation documents, **23** preselected epochs.
- Vocabulary: **20000** tokens; PAD=0, UNK=1.
- Input budget: eight contiguous windows of 256 tokens, up to 2,048 tokens per label.
- Initial runtime: PyTorch `2.8.0+cpu`, `cpu`.

## Use

```powershell
.\.venv\Scripts\python.exe -m medical_doc_assistant classify-neural --file "data/examples/phase2_demo_leaflet.txt"
```

```python
from pathlib import Path
from medical_doc_assistant.neural_model import classify_neural_text

text = Path("data/examples/phase2_demo_leaflet.txt").read_text(encoding="utf-8")
result = classify_neural_text(Path("models/phase3/text_cnn.pt"), text)
print(result["predicted_class"])
```

The checkpoint includes architecture parameters, label order, the token vocabulary,
encoding settings, and source/split provenance. Loading reconstructs the model
from its state dictionary with `weights_only=True`; inference uses evaluation
mode with dropout disabled.

## Scope and comparison

Outputs are the six EPC-derived course categories: antibacterial, anti_inflammatory_nsaid, corticosteroid, antifungal, acid_reducer, antihypertensive.
This is a closed-set English medicine-label classifier. It does not provide an
expert-confirmed indication, diagnosis, medical recommendation, or calibrated
confidence. Other document types, unknown categories, and Arabic are later extensions.

Training-only vocabulary fitting and group-separated validation govern tuning.
Final vocabulary/model fitting use train + validation, excluding test. The same
record split as Phase 2 is used for evaluation. TF-IDF uses full text whereas the
CNN samples a bounded context; coverage and actual results are recorded in
`reports/phase_3_review.md`. Source labels are automatic metadata targets.

Data provided by the U.S. Food and Drug Administration (<https://open.fda.gov>).

This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.

As stated in the project plan, the project is informational only and is not
medical advice.
