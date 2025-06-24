# Task 4: Model Comparison & Selection

## Objective
Compare multiple NER models for extracting product, price, and location entities from Amharic Telegram messages.

## Models Compared
- **XLM-Roberta** (xlm-roberta-base)
- **DistilBERT** (distilbert-base-multilingual-cased)

## Evaluation Summary

| Model         | Eval Loss | Size  | Training Speed | Notes |
|---------------|-----------|-------|----------------|-------|
| XLM-Roberta   | ~0.12     | ~3GB  | Slower         | Better accuracy, but heavy |
| DistilBERT    | 0.085     | ~500MB| Faster         | Chosen model ✅ |

## Selected Model
**DistilBERT** was selected for deployment based on its low eval loss, faster training, and smaller size.

Model saved to: `models/ner-distilbert`
