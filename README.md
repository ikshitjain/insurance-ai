# Insurance AI - Quality Assurance System

## Overview

This Insurance AI system now has comprehensive **Quality Assurance (QA)** checks built into every component to ensure data integrity, model reliability, and system health.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QA Configuration Layer                    │
│                      (qa_config.py)                          │
│  - DocumentQA    - EmbeddingQA    - VectorStoreQA            │
│  - ModelQA       - RAGQA          - Validation Utils         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│  create_index.py  │                     │      app.py       │
│  (Index Creation) │                     │   (Streamlit UI)  │
│                   │                     │                   │
│  ✓ Input QA       │                     │  ✓ Health Checks  │
│  ✓ Document QA    │                     │  ✓ Input Validation│
│  ✓ Chunk QA       │                     │  ✓ Output QA      │
│  ✓ Embedding QA   │                     │  ✓ Error Handling │
│  ✓ VectorStore QA │                     │                   │
└───────────────────┘                     └───────────────────┘
```

## QA Modules

### 1. DocumentQA (`qa_config.py`)
Validates PDF documents and text chunking:

| Check | Threshold |
|-------|-----------|
| File extensions | `.pdf` only |
| File size | 10 KB - 50 MB |
| Pages | 1 - 500 |
| Chunk size | 100 - 1000 chars |
| Chunk overlap | 10 - 200 chars |
| Min chunks | 1 |
| Max empty chunks | < 5% |
| Min avg chunk length | 200 chars |

### 2. EmbeddingQA (`qa_config.py`)
Validates embedding models and vectors:

| Check | Threshold |
|-------|-----------|
| Supported models | `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, etc. |
| Vector dimension | 384 - 1024 |
| Invalid values | No NaN/Inf allowed |
| Magnitude | 0.5 - 1.5 (normalized) |

### 3. VectorStoreQA (`qa_config.py`)
Validates FAISS vector store:

| Check | Threshold |
|-------|-----------|
| Min vectors | 1 |
| Max vectors | 1,000,000 |
| Consistency | Vector count = chunk count |
| Retrieval test | Similarity search must work |

### 4. ModelQA (`qa_config.py`)
Validates ML model inputs and predictions:

| Feature | Valid Range |
|---------|-------------|
| SEX | 0 - 1 |
| INSURED_VALUE | 0 - 100,000,000 |
| PREMIUM | 0 - 1,000,000 |
| PROD_YEAR | 1990 - 2030 |
| SEATS_NUM | 1 - 50 |
| CARRYING_CAPACITY | 0 - 50,000 |
| CCM_TON | 0 - 20,000 |
| Max reasonable claim | ₹100,000,000 |

### 5. RAGQA (`qa_config.py`)
Validates LLM/RAG operations:

| Check | Threshold |
|-------|-----------|
| Supported models | `phi3`, `tinydolphin`, `llama2`, `mistral` |
| Query length | 3 - 1000 chars |
| Response length | Max 5000 chars |
| Timeout | 60 seconds |

## Usage

### Creating Index with QA

```bash
python create_index.py
```

The script will:
1. ✓ Validate input file and parameters
2. ✓ Load and validate PDF document
3. ✓ Create and validate text chunks
4. ✓ Generate and validate embeddings
5. ✓ Build and validate FAISS vector store
6. ✓ Save with confirmation

**Exit codes:**
- `0`: All QA checks passed
- `1`: One or more QA checks failed

### Running the App with QA

```bash
streamlit run app.py
```

The app performs:
- **System Health Dashboard** (sidebar): Real-time health status of ML Model, Vector DB, and LLM
- **Input Validation**: All user inputs validated before processing
- **Prediction QA**: Feature ranges checked, prediction values validated
- **Query QA**: Query length validated, response quality checked
- **Error Handling**: Graceful degradation with informative error messages

## Validation Status Types

| Status | Icon | Meaning |
|--------|------|---------|
| `PASS` | ✓ | Check passed successfully |
| `WARNING` | ⚠️ | Non-critical issue, proceed with caution |
| `FAIL` | ✗ | Critical issue, operation blocked |

## Health Check Levels

| Level | Status | Description |
|-------|--------|-------------|
| Healthy | ✅ | All checks passed |
| Degraded | ⚠️ | Some warnings, system functional |
| Unhealthy | ❌ | Critical failure, system not functional |

## Adding New QA Checks

To add a new validation rule:

1. **Add to `qa_config.py`**:
```python
class YourComponentQA:
    @staticmethod
    def validate_your_check(value) -> ValidationResult:
        if not valid:
            return ValidationResult(
                ValidationStatus.FAIL,
                "Description of failure",
                "field_name",
                value
            )
        return ValidationResult(
            ValidationStatus.PASS,
            "Description of success",
            "field_name"
        )
```

2. **Use in your code**:
```python
from qa_config import YourComponentQA

result = YourComponentQA.validate_your_check(some_value)
if result.status == ValidationStatus.FAIL:
    # Handle failure
    pass
```

## Troubleshooting

### Common QA Failures

| Error | Solution |
|-------|----------|
| "File not found" | Check `data/PCIPolicyWording.pdf` exists |
| "No chunks created" | PDF may be image-only; try OCR |
| "Empty embedding vector" | Check embedding model downloaded |
| "Vector store is empty" | Re-run `create_index.py` |
| "LLM not available" | Start Ollama: `ollama serve` |

### System Health Issues

If sidebar shows ❌ for any component:

1. **ML Model ❌**: Ensure `insurance_model.pkl` exists
2. **Vector DB ❌**: Run `python create_index.py` to create `faiss_index/`
3. **LLM ❌**: Start Ollama and download a model:
   ```bash
   ollama serve
   ollama pull phi3
   ```

## Best Practices

1. **Always check health dashboard** before making predictions
2. **Review warnings** even if predictions succeed
3. **Keep QA thresholds** updated as data patterns change
4. **Log QA failures** for debugging and improvement
5. **Test with edge cases** to validate QA rules

## Future Enhancements

- [ ] Confidence scores for predictions
- [ ] Automated QA report generation
- [ ] Performance benchmarks
- [ ] Anomaly detection for inputs
- [ ] A/B testing framework for models
