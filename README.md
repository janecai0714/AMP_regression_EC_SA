# A BERT-based Transfer Learning Approach for Predicting the Minimum Inhibitory Concentrations of Antimicrobial Peptides for Escherichia coli and Staphylococcus aureus

## Description

Antimicrobial peptides (AMPs) are a promising alternative for combating bacterial drug resistance. While current computer prediction models excel in binary classification of AMPs from sequences, 
there is a lack of regression methods to accurately quantify AMP activity against specific bacteria, making the identification of highly potent AMPs a challenge. In this study, we proposed a 
deep learning model based on the fine-tuned Bidirectional Encoder Representations from Transformers (BERT) architecture to extract embedding features from input sequences and predict minimum inhibitory concentrations (MICs) for target bacterial species. Using the transfer learning strategy, we built re gression models for Escherichia coli (EC) and Staphylococcus aureus (SA) using data curated from DBAASP.

### Dataset
data is curated from DBAASP, it includes sequences only with 5-60 AA in length. The activity values of the peptides were converted to pMIC (-log10 MIC), where the unit of MIC is µM. This dataset was used to construct regression models for Escherichia coli and Staphylococcus aureus.
* Dataset for Escherichia coli has a median MIC value of 13.49 µM (corresponding to a pMIC of −1.13): 4042 sequences. （/data/EC.csv)
  Train dataset for Escherichia coli: 3638 sequences. (/data/train-EC.csv); test dataset for Escherichia coli: 404 sequences. (/data/test-EC.csv)

* Dataset for Staphylococcus aureus has a median MIC value of 16.22 µM (corresponding to a pMIC of −1.21): 3275 sequences. (/data/SA.csv)
  Train dataset for Staphylococcus aureus: 2947 sequences. (/data/train-SA.csv); test dataset for Staphylococcus aureus: 328 sequences. (/data/test-SA.csv)

## Getting Started

### Python packages

* torch==2.0.1+cu118
* biopython==1.81
* transformers==4.28.1
* tokenizers==0.13.3

### Executing program (take EC as an example)

* run /EC/bert_finetuen/train_test.py to build the regression model
* run /EC/bert_finetune/reproduce.py to reproduce experimental results
* run /EC/ml_base/ml_train_test.py to build traditional machine learning models

### Predict sequences (receive a fasta file and output a csv file)
* in the /predict/predict.py, change the variables (fasta_path, csv_path) to your own filename, run /predict/predict.py to predict pMIC values for input sequences
## Acknowledgments

Inspiration, code snippets, etc.
* Elnaggar, Ahmed, Michael Heinzinger, Christian Dallago, Ghalia Rehawi, Yu Wang, Llion Jones, Tom Gibbs et al. "Prottrans: Toward understanding the language of life through self-supervised learning." IEEE transactions on pattern analysis and machine intelligence 44, no. 10 (2021): 7112-7127.
