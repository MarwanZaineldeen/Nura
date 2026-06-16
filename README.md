# RAG-Based Mental Health Support Chatbot

This repository contains the module work for the NLP final project.

## Module 1: Language Detection

The language detector is implemented with traditional NLP:

- Vectorizer: character-level TF-IDF with `char_wb` n-grams from 2 to 4 characters
- Classifier: Multinomial Naive Bayes
- Dataset: `papluca/language-identification`
- Supported languages: Arabic, Bulgarian, German, Greek, English, Spanish, French, Hindi, Italian, Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish, Urdu, Vietnamese, Chinese

### Train and Evaluate

```bash
.\.venv\Scripts\python.exe src\models\language_classifier.py
```

This trains the model, saves it to `src/models/saved_lang_model.pkl`, and writes reports to:

```text
reports/module_1_language_detection/
```

### Run the UI

```bash
.\.venv\Scripts\python.exe src\models\language_detector_ui.py
```

The UI returns the detected language, confidence, and whether the prediction passed the confidence threshold.

### Install Dependencies

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
