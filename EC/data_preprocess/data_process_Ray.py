import pandas as pd
import numpy as np
import os
import re
from pyteomics import mass
from Bio.SeqUtils.ProtParam import ProteinAnalysis # biopython

# Read the raw csv file
root_dir = os.path.dirname(os.path.abspath("data_process.py"))
data_file_name = 'peptides-complete.csv'
data_file_path = os.path.join(root_dir, data_file_name)
data_raw_df = pd.read_csv(data_file_path, low_memory=False, encoding='utf_8_sig')
# Remove all-empty columns
data_raw_df = data_raw_df.dropna(axis=1, how='all')
# Remove Unrelated columns
data_raw_df = data_raw_df[['ID', 'COMPLEXITY', 'NAME',
                           'N TERMINUS', 'SEQUENCE', 'C TERMINUS', 'SYNTHESIS TYPE',
                           'TARGET GROUP', 'TARGET OBJECT',
                           'UNUSUAL OR MODIFIED AMINO ACID - POSITION', 'UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE', 'UNUSUAL OR MODIFIED AMINO ACID - BEFORE MODIFICATION',
                           'TARGET ACTIVITY - TARGET SPECIES',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - TARGET CELL',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - LYSIS VALUE',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - UNIT',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - NOTE',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - REFERENCE']]
# Remove ID-empty records
data_raw_df.dropna(subset=['ID'], axis=0, inplace=True)
# Remove SEQUENCE-empty records
data_raw_df.dropna(subset=['SEQUENCE'], axis=0, inplace=True)
# Remove non-MIC ACTIVITY MEASURE records
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - LYSIS VALUE'] == '50% Hemolysis')].index), inplace=True)

# Remove non-monomer records
# complexity column only contains {'Multi-Peptide', 'Monomer', 'Multimer'}
# Remove 'Multi-Peptide' and 'Multimer' records
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['COMPLEXITY'] == 'Monomer')].index), inplace=True)

# Remove all records with unusual modification in the sequence
# Based on column "UNUSUAL OR MODIFIED AMINO ACID - POSITION" and
# 'UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE' is/isn't Null
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['UNUSUAL OR MODIFIED AMINO ACID - POSITION'].isnull())].index), inplace=True)
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE'].isnull())].index), inplace=True)

# Remove all records containing non-standard AA
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['SEQUENCE'].str.isalpha())].index), inplace=True)
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['SEQUENCE'].str.isupper())].index), inplace=True)
data_raw_df.drop(index=(data_raw_df.loc[(data_raw_df['SEQUENCE'].str.contains('B|J|O|U|X|Z'))].index), inplace=True)

# Remove all records without toxicity value
data_raw_df.drop(index=(data_raw_df.loc[data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'].isnull()].index), inplace=True)

# PROCESS toxicity VALUES
# Remove toxicity symbol  >, <, =, >=, <= ----> consider >500 as 500
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'].str.strip()
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'].str.strip('>')
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'].str.strip('<')
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'].str.strip('=')
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'].str.strip('?')
# Remove ± ----> consider 3.3±1.1 as 3.3
data_raw_df = pd.concat([data_raw_df, data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION'].str.split('±', expand=True)], axis = 1)
if 0 in data_raw_df.columns: data_raw_df.rename(columns={0: 'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'}, inplace=True)
if 1 in data_raw_df.columns: data_raw_df.drop([1], axis=1, inplace=True)
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'].str.strip()
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'].str.strip('\\')
# Remove other toxicity values with unstandard symbol
data_raw_df.drop(index=(data_raw_df.loc[(data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'].str.contains(' |,|\n|\t|–>|–'))].index), inplace=True)

# Remove range values "-" and "->" ----> consider activity range "10-20" as 10+(20-10)/2
def value_split_process(sample_split):
    sample_split[0] = sample_split[0].strip()
    sample_split[0] = sample_split[0].strip('>')
    sample_split[0] = sample_split[0].strip('<')
    sample_split[0] = sample_split[0].strip('=')
    sample_split[0] = sample_split[0].strip('-')
    sample_split[0] = sample_split[0].strip('-')
    sample_split[0] = sample_split[0].strip('?')
    sample_split[0] = sample_split[0].strip('\\t')
    sample_split[1] = sample_split[1].strip()
    sample_split[1] = sample_split[1].strip('>')
    sample_split[1] = sample_split[1].strip('<')
    sample_split[1] = sample_split[1].strip('=')
    sample_split[1] = sample_split[1].strip('-')
    sample_split[1] = sample_split[1].strip('?')
    sample_split[1] = sample_split[1].strip('\\t')
    return sample_split
def range_value_process(sample):
    sample = sample.replace(' ', '')
    sample = sample.replace('\n', '')
    sample = sample.replace('\t', '')
    if '-' in sample:
        sample_split = sample.split('-')
        sample_split_process = value_split_process(sample_split)
        result = float(sample_split_process[0]) + (float(sample_split_process[1]) - float(sample_split_process[0])) / 2
    else:
        sample = sample.strip()
        sample = sample.strip('>')
        sample = sample.strip('<')
        sample = sample.strip('=')
        sample = sample.strip('-')
        sample = sample.strip('\\t')
        sample = sample.strip('?')
        result = float(sample)
    return result
data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'] = data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'].map(lambda x: range_value_process(x))

# Remove samples with empty unit in "HEMOLITIC CYTOTOXIC ACTIVITY - UNIT" column
data_raw_df.drop(index=(data_raw_df.loc[data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - UNIT'].isnull()].index), inplace=True)

# pip index that need to convert µg/ml into µM
convert_idx = data_raw_df.loc[data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - UNIT'] == 'µg/ml'].index.tolist()
for pip_idx in convert_idx:
    pip_mass = ProteinAnalysis(data_raw_df.loc[pip_idx, 'SEQUENCE']).molecular_weight()
    data_raw_df.loc[pip_idx, 'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'] = (data_raw_df.loc[pip_idx, 'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'] * 1000) / pip_mass

data_raw_df['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED - UNIT'] = ['µM']*data_raw_df.shape[0]
# Remove duplicate records #1
data_raw_df.drop_duplicates(inplace=True)
data_raw_df.reset_index(inplace=True, drop=True)

# Remove duplicate records #2
data_raw_df.drop_duplicates(inplace=True)
data_raw_df.reset_index(inplace=True, drop=True)
data_raw_df.to_csv("Hemolysis50.csv")

# remove ID with multiple 50% hemolysis records (222 records)
# example ID360, 50% Hemolysis > 1000 on reference 19, but 50% Hemolysis 19±1.0 on reference 2

# keep duplicate records for one ID and use the average
data_df_ave = data_raw_df.groupby(['ID', 'SEQUENCE'])['HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION - PROCED'].mean().reset_index()
data_df_ave.to_csv("Hemolysis50_ave.csv", index=False)
print("smart")