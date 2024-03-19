# Project Title

A BERT-based Transfer Learning Approach for Predicting the Minimum Inhibitory Concentrations of Antimicrobial Peptides for Escherichia coli and Staphylococcus aureus

## Description

Antimicrobial peptides (AMPs) are a promising alternative for combating bacterial drug resistance. While current computer prediction models excel in binary classification of AMPs from sequences, 
there is a lack of regression methods to accurately quantify AMP activity against specific bacteria, making the identification of highly potent AMPs a challenge. In this study, we proposed a 
deep learning model based on the fine-tuned Bidirectional Encoder Representations from Transformers (BERT) architecture to extract embedding features from input sequences and predict minimum inhibitory concentrations (MICs) for target bacterial species. Using the transfer learning strategy, we built re gression models for Escherichia coli (EC) and Staphylococcus aureus (SA) using data curated from DBAASP.

## Getting Started

### Python packages

* torch==2.0.1+cu118
* biopython==1.81
* transformers==4.28.1
* tokenizers==0.13.3

### Executing program (take EC as an example)

* run EC/bert_finetuen/train_test.py to build the regression model
* run EC/bert_finetune/reproduce.py to reproduce experimental results
* run EC/ml_base/ml_train_test.py to build traditional machine learning models

### Predict sequences (receive a fasta file and output a csv file)
* change the variables (fasta_path, csv_path) to your own filename, run predict/predict.py to predict pMIC values for input sequences
## Acknowledgments

Inspiration, code snippets, etc.
* Lei, Thomas MT, Stanley CW Ng, and Shirley WI Siu. "Application of ANN, XGBoost, and other ml methods to forecast air quality in Macau." Sustainability 15, no. 6 (2023): 5341.
