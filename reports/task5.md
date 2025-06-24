# Task 5: Model Interpretability

## Objective

Enhance the transparency of the DistilBERT-based Amharic Named Entity Recognition (NER) model by applying model interpretability techniques. The goal is to understand how the model makes decisions and identify areas for further improvement.

---

## Tools Used

- **SHAP (SHapley Additive Explanations):** To compute per-token contributions and visualize which tokens influenced entity predictions.
- **LIME (Local Interpretable Model-agnostic Explanations):** (Optional) Intended for interpreting predictions in local contexts but not fully integrated due to time constraints.

---

## Approach

1. **Sample Input Selection:**
   - Used representative product advertisement text from Telegram such as:
     ```
     "Imitation Volcano Humidifier 1400 ብር"
     ```

2. **Token-Level Explanation with SHAP:**
   - Loaded the fine-tuned `distilbert-base-multilingual-cased` model.
   - Used a forward function to generate token-wise logits.
   - Applied SHAP to explain token contributions.
   - Visualized SHAP values using `shap.plots.text()`.

3. **Difficult Case Analysis:**
   - Observed that certain tokens (especially numeric or Amharic script) resulted in `[UNK]` (unknown tokens), reducing interpretability.
   - The model had difficulty classifying overlapping or adjacent entities (e.g., "1400 ብር" as price).

---

## Observations

| Text Span                         | Entity       | SHAP Insight                                           |
|----------------------------------|--------------|--------------------------------------------------------|
| `Imitation Volcano Humidifier`   | Product      | SHAP values were low due to token fragmentation        |
| `1400 ብር`                         | Price        | SHAP did not clearly differentiate Amharic numerals    |
| `[UNK]` tokens                   | -            | Hurt explainability; caused null or default outputs    |

---

## Limitations

- SHAP and LIME are better suited for classification or regression outputs than for sequence labeling tasks like NER.
- Tokenization introduces subword splits (`##ifier`, `##cano`) which dilute interpretability.
- `[UNK]` tokens hinder transparency especially for unseen Amharic or mixed script inputs.

---

## Recommendations

- **Improve Tokenizer Coverage:** Consider domain-specific tokenizers or byte-level models (e.g., byte-level BPE).
- **Use Integrated Gradients or Attention-based Visualization:** These techniques are more NER-suited.
- **Amharic-Specific Tools:** Explore tokenizers and models trained natively on Amharic for better feature attribution.

---

## Conclusion

Model interpretability using SHAP gave partial insights into how the DistilBERT-based NER model identifies entities. Although effective in visualizing token contributions, limitations with multilingual tokenization and `[UNK]` degradation highlight the need for better preprocessing and token-level evaluation tools.

> 📁 Outputs saved to: `interpretability/`  
> 🧠 Primary model explained: `models/ner-distilbert`
