# Detecting Patronising and Condescending Language (PCL)

This repository contains our approach to detecting **patronising and condescending language (PCL)** using both traditional baselines and transformer-based models. The project was developed for the Natural Language Processing course at Imperial College London.

We tackle this nuanced classification task by combining **DeBERTa-v3-Small**, **data augmentation**, **class imbalance handling**, and **error analysis**, outperforming baseline models like TF-IDF and Bag-of-Words.


## Project Structure

```bash
.
├── data_analysis.ipynb            # Data exploration and label/category analysis
├── tf-idf_baseline.ipynb          # TF-IDF + Logistic Regression baseline
├── model_bow.ipynb                # Bag-of-Words + Naive Bayes baseline
├── deberta.ipynb                  # Final DeBERTa-v3 training notebook
├── albert.ipynb                   # ALBERT (initial test model)
├── hyperparameter_tuning.ipynb    # DeBERTa tuning experiments
├── data_augmentation.py           # Back-translation + synonym replacement
├── dev.txt                        # Development set (for evaluation)
├── test.txt                       # Test set (no labels)
├── requirements.txt               # Python dependencies
├── data/                          # Training, dev and test .csv/.tsv files
├── Dont_Patronize_Me_Trainingset/ # Raw dataset (from official task)
├── figures/                       # Figures and plots for report/analysis
├── report.pdf                     # Final report
└── README.md
```

## Task Description

The goal is to **predict whether a given text contains patronising and condescending language**. This challenge was originally part of **SemEval 2022 - Task 4**. The main challenges include:

- Subtle and subjective phrasing  
- Dataset class imbalance  
- Biased framing across social categories  

## Models and Techniques

We experimented with several approaches:

### Statistical Baselines

- **TF-IDF + Logistic Regression**  
- **Bag-of-Words + Naive Bayes**

These models provided limited performance due to their inability to capture context or tone.  
**Best F1-score**: `0.31`

### Transformers

- **ALBERT-v2** – Lightweight and efficient, but underperformed on nuance  
- **DeBERTa-v3-Small** – Final model used for its strong performance-to-complexity ratio

## Improvements Applied

- **Data Augmentation** via back-translation and synonym replacement  
- **Class-weighted loss** to address imbalance in PCL labels  
- **Early stopping** and **cosine learning rate scheduler**  
- **Hyperparameter tuning**: dropout, learning rate, batch size  
- **Error analysis** for failure cases and ambiguous texts  

## Results

| Model                  | F1 Score (Dev) |
|------------------------|----------------|
| Bag-of-Words (NB)      | 0.22           |
| TF-IDF (LogReg)        | 0.31           |
| ALBERT-v2              | 0.47           |
| **DeBERTa-v3-Small**   | **0.52**       |

The DeBERTa model showed clear improvements in capturing tone and subtlety, outperforming all traditional baselines.

## Analysis Highlights

- Accuracy improves with **more explicit condescension**, and drops for subtle cases  
- **Longer texts** tend to be more error-prone due to token truncation  
- Performance varies by **topic category** (e.g. homelessness, refugees), likely due to framing biases

## Dataset

We used the **Don’t Patronize Me!** dataset from SemEval 2022:

- Task: Binary classification (PCL vs non-PCL)  
- Significant class imbalance (~15% PCL)  
- Texts labeled by category (e.g. women, refugees, homeless)

**Dataset links:**
- [Official GitHub](https://github.com/Perez-AlmendrosC/dontpatronizeme)  
- [SemEval Paper](https://aclanthology.org/2022.nlp4pi-1.15/)


## Authors

- Cesare Bergossi  
- Lisa Faloughi  
- Oliver Shakespeare  
