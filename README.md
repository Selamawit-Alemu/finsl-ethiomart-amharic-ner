
# EthioMart Amharic NER System

This project builds a Named Entity Recognition (NER) pipeline for extracting product, price, and location information from Amharic e-commerce messages posted on Telegram channels.

---

## 📌 Project Goals

- Scrape real-time messages from Ethiopian e-commerce Telegram channels
- Preprocess and normalize Amharic text data
- Manually label a subset of data in CoNLL format for NER training
- Fine-tune transformer models for Amharic entity extraction

---

## 📁 Folder Structure

    ├── data/
    │ ├── raw/ # Raw input files (e.g., Excel with channels)
    │ ├── clean/ # Preprocessed & labeled data
    ├── outputs/ # Model outputs, logs
    ├── photos/ # Downloaded media (images from Telegram)
    ├── models/
    │ └── ner-distilbert/
    │ └── WeightedTokenClassification.py
    ├── notebooks/
    │ ├── 01_preprocessing.ipynb
    │ ├── FinTech Vendor Scorecard for Micro-Lending.ipynb
    │ ├── interpreting_distilbert.ipynb
    │ └── model_evaluation.ipynb
    ├── reports/
    │ ├── task4_model_comparison.md
    │ └── task5.md
    ├── scripts/
    │ ├── Parse_labeled_conll.py
    │ ├── auto_label_unlabeled.py
    │ ├── prepare_for_label_studio.py
    │ ├── preprocess_telegram_data.py
    │ ├── real_time_ingest.py
    │ ├── telegram_scrapper.py
    │ ├── train_distilbert.py
    │ └── train_ner_model.py
    ├── requirements.txt
    └── README.md

---

---

## ✅ Tasks Completed

### **Task 1: Data Collection**
- Scraped 5 active Telegram vendors using `telethon` API.
- Saved messages, metadata (timestamps, views), and media paths into CSV.

### **Task 2: Data Annotation**
- Manually labeled ~500 messages in **CoNLL** format using `label-studio`.
- Focused on 3 entity types: `Product`, `Price`, `Location`.

### **Task 3: NER Model Training**
- Preprocessed Amharic Telegram messages.
- Fine-tuned multilingual transformers:
  - ✅ `DistilBERT` (final choice)
  - `XLM-Roberta` (tested but heavier)
- Used `class-weighted loss` to tackle class imbalance (many "O" tokens).
- Achieved eval loss ~0.085 with DistilBERT.

### **Task 4: Model Comparison**
- Compared models based on loss, speed, size.
- DistilBERT selected due to:
  - Smaller size (~500MB)
  - Lower loss
  - Faster training
- See: `reports/task4_model_comparison.md`

### **Task 5: Model Interpretability**
- Applied **SHAP** and **LIME** to visualize token-level predictions.
- Investigated ambiguous predictions and edge cases.
- See: `reports/task5.md` and `notebooks/interpreting_distilbert.ipynb`

### **Task 6: Vendor Scorecard**
- Built an engine to aggregate:
  - Posting frequency
  - Average views per post
  - Average listed price
  - Top performing post
- Computed a custom **Lending Score**:
Lending Score = 0.5 * AvgViews + 0.5 * PostsPerWeek

- Final scorecard presented in: `notebooks/FinTech Vendor Scorecard for Micro-Lending.ipynb`

---

## 📊 Sample Vendor Scorecard Output

| Vendor            | Avg. Views/Post | Posts/Week | Avg. Price (ETB) | Lending Score |
|------------------|------------------|------------|------------------|----------------|
| @ZemenExpress    | 1220             | 5.2        | 1600             | 861.0          |
| @sinayelj        | 800              | 2.5        | 1200             | 512.5          |
| @Leyueqa         | 1440             | 6.0        | 2100             | 990.0          |

---

## 📈 Tools & Technologies

| Area                    | Tools / Libraries                                                                 |
|-------------------------|------------------------------------------------------------------------------------|
| Data Ingestion          | `telethon`, `pandas`, `openpyxl`, `dotenv`                                        |
| Annotation              | `label-studio`, `regex`                                                           |
| Modeling                | `transformers`, `datasets`, `torch`, `seqeval`, `evaluate`                        |
| Interpretability        | `SHAP`, `LIME`                                                                    |
| Tracking & DevOps       | `flake8`, `pytest`, `jupyter`, `accelerate`, `tqdm`                               |
| Visualizations          | `matplotlib`, `seaborn`                                                           |

---

## 🔧 Setup Instructions

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/ethiomart-amharic-ner.git
cd ethiomart-amharic-ner

## 🛠️ Setup Instructions

1. **Clone the repo & install dependencies**
   ```bash
   git clone https://github.com/<your-username>/ethiomart-amharic-ner.git
   cd ethiomart-amharic-ner
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt


Configure your Telegram API credentials

Create a .env file in the root directory:

TG_API_ID=your_api_id
TG_API_HASH=your_api_hash
phone=+2519xxxxxxx

🚀 How to Run the System

🧲 Step 1: Scrape Telegram Channels

Edit data/raw/5_channels_to_crawl.xlsx to include 5+ target channels (one per line). Then run:

python scripts/telegram_scrapper.py

This will:

Log in to your Telegram account

Download up to 10,000 messages per channel

Save messages and metadata to telegram_data.csv

Download media to /photos

🧹 Step 2: Preprocess the Data

python scripts/preprocess_telegram_data.py

This will:

Clean Amharic/English text (punctuation, emojis, formatting)

Tokenize each message

Export structured data to processed_telegram_data.csv

Prepare unlabeled_conll.txt for manual labeling

🏷️ Step 3: Label in CoNLL Format

Manually label a subset (30–50 messages) using entity tags:

B-Product, I-Product

B-PRICE, I-PRICE

B-LOC, I-LOC

O for non-entities

Save your labels to data/clean/labeled_conll.txt.
Preprocess + Train:

    python scripts/preprocess_telegram_data.py
    python scripts/train_distilbert.py
    Run Jupyter Notebooks:

jupyter notebook
# Open model_evaluation.ipynb or FinTech Vendor Scorecard.ipynb


 Business Impact

    Entity extraction enables automation: We can now track pricing trends and vendor activity at scale.

    Vendor Scorecard bridges fintech and real-time engagement, giving lenders evidence for offering loans.

    Transparent AI via SHAP/LIME builds trust for future AI-driven financial decisions in Ethiopia.


## 🚀 **Setup & Run Instructions**

1. **Clone repo:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/ethiomart-amharic-ner.git
    cd ethiomart-amharic-ner
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Scrape Telegram data:**
    ```bash
    python scripts/telegram_scrapper.py
    ```

4. **Preprocess and train model:**
    ```bash
    python scripts/preprocess_telegram_data.py
    python scripts/train_distilbert.py
    ```

5. **Launch Jupyter notebooks:**
    ```bash
    jupyter notebook
    ```
    - Open and run:
      - `model_evaluation.ipynb`
      - `FinTech Vendor Scorecard for Micro-Lending.ipynb`
      - `interpreting_distilbert.ipynb`
