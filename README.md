# infant-gut-microbiome-transition-ml

>Amplicon-based infant gut microbiome transition across 0–48 months using nested CV + ML models.
>Predictive biomarker analysis age specific.
>"Window of opportunity" INFANT GUT MICROBION.
>Repository contains the code for the infant gut microbiome transition.

## Methods



The analysis uses a 75/25 train-test split followed by nested cross-validation on the training set.



Models evaluated:



\- Random Forest

\- Extra Trees

\- LightGBM

\- XGBoost

\- RBF-SVM

\- Multinomial Logistic Regression

\- Multilayer Perceptron



\## Repository Structure



```text

.

├── README.md

├── requirements.txt

├── scripts/

│   └── Infant\_microbiome\_transition.py

├── data/

│   └── README.md

└── results/

&#x20;   └── README.md

