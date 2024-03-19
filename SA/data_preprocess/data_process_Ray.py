# encoding: utf-8
# Requirements
# This is the complete peptide dataset from DBAASP,
# need to generate a separate dataset file for each specie.
# The dataset file should contain the ID, Sequence, MIC value (converted into uM).

# Ignore these sequences:
# 1) non-monomer (i.e. other multimer, and we take only monomer)
# 2) there are unusal modification in the sequence
# (check the column "UNUSUAL.OR.MODIFIED.AMINO.ACID...POSITIO")
# 3) sequence contains "X" unknown amino acid
# 4) without MIC measure

# For the valid entries, please do
# 5) convert MIC value into uM unit
# 6) compute averaged MIC value for the same sequence with multiple MIC values
# 7) save each specie data into a file "specie_name.csv"

# Library Requirement
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
#print(data_raw_df.shape)
# Dim: (139117, 79)

# Remove all-empty columns
data_raw_df = data_raw_df.dropna(axis=1, how='all')
#print(data_raw_df.shape)
# Dim: (139117, 78)

# Remove Unrelated columns
data_raw_df = data_raw_df[['ID', 'COMPLEXITY', 'NAME',
                           'N TERMINUS', 'SEQUENCE', 'C TERMINUS', 'SYNTHESIS TYPE',
                           'TARGET GROUP', 'TARGET OBJECT',
                           'UNUSUAL OR MODIFIED AMINO ACID - POSITION', 'UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE', 'UNUSUAL OR MODIFIED AMINO ACID - BEFORE MODIFICATION',
                           'TARGET ACTIVITY - TARGET SPECIES',
                           'TARGET ACTIVITY - ACTIVITY MEASURE VALUE',
                           'TARGET ACTIVITY - CONCENTRATION',
                           'TARGET ACTIVITY - UNIT',
                           'TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION',
                           'HEMOLITIC CYTOTOXIC ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)']]
#print(data_raw_df.shape)
# Dim: (139117, 17)

# Remove ID-empty records
data_raw_df.dropna(subset=['ID'], axis=0, inplace=True)
#print(data_raw_df.shape)
# Dim: (139117, 17)

# Remove SEQUENCE-empty records
data_raw_df.dropna(subset=['SEQUENCE'], axis=0, inplace=True)
#print(data_raw_df.shape)
# Dim: (139109, 17)

# Remove non-MIC ACTIVITY MEASURE records
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['TARGET ACTIVITY - ACTIVITY MEASURE VALUE'] == 'MIC')].index), inplace=True)
#print(data_raw_df.shape)
# Dim: (93101, 17)

# Remove non-monomer records
# complexity column only contains {'Multi-Peptide', 'Monomer', 'Multimer'}
# Remove 'Multi-Peptide' and 'Multimer' records
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['COMPLEXITY'] == 'Monomer')].index), inplace=True)
#print(data_raw_df.shape)
# Dim: (90961, 17)

# Remove all records with unusual modification in the sequence
# Based on column "UNUSUAL OR MODIFIED AMINO ACID - POSITION" and
# 'UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE' is/isn't Null
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['UNUSUAL OR MODIFIED AMINO ACID - POSITION'].isnull())].index), inplace=True)
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE'].isnull())].index), inplace=True)
#print(data_raw_df.shape)
# Dim: (81654, 17)

# Remove all records containing non-standard AA
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['SEQUENCE'].str.isalpha())].index), inplace=True)
data_raw_df.drop(index=(data_raw_df.loc[~(data_raw_df['SEQUENCE'].str.isupper())].index), inplace=True)
data_raw_df.drop(index=(data_raw_df.loc[(data_raw_df['SEQUENCE'].str.contains('B|J|O|U|X|Z'))].index), inplace=True)
#print(data_raw_df.shape)
# Dim: (70288, 17)

# Remove all records without MIC value
data_raw_df.drop(index=(data_raw_df.loc[data_raw_df['TARGET ACTIVITY - CONCENTRATION'].isnull()].index), inplace=True)
#print(data_raw_df.shape)
# Dim: (70287, 17)

# PROCESS MIC VALUES
# Remove MIC symbol  >, <, =, >=, <= ----> consider >500 as 500
data_raw_df['TARGET ACTIVITY - CONCENTRATION'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION'].str.strip()
data_raw_df['TARGET ACTIVITY - CONCENTRATION'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION'].str.strip('>')
data_raw_df['TARGET ACTIVITY - CONCENTRATION'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION'].str.strip('<')
data_raw_df['TARGET ACTIVITY - CONCENTRATION'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION'].str.strip('=')
data_raw_df['TARGET ACTIVITY - CONCENTRATION'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION'].str.strip('?')
# Remove ± ----> consider 3.3±1.1 as 3.3
data_raw_df = pd.concat([data_raw_df, data_raw_df['TARGET ACTIVITY - CONCENTRATION'].str.split('±', expand=True)], axis = 1)
if 0 in data_raw_df.columns: data_raw_df.rename(columns={0: 'TARGET ACTIVITY - CONCENTRATION - PROCED'}, inplace=True)
if 1 in data_raw_df.columns: data_raw_df.drop([1], axis=1, inplace=True)
data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].str.strip()
data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].str.strip('\\')
# Remove other MIC values with unstandard symbol
data_raw_df.drop(index=(data_raw_df.loc[(data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].str.contains(' |,|\n|\t|–>|–'))].index), inplace=True)
#print(data_raw_df.shape)
# Dim: (70263, 18)

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
data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED'] = data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].map(lambda x: range_value_process(x))
#print(data_raw_df.shape)
# Dim: (70263, 18)

# Remove samples with empty unit in "TARGET ACTIVITY - UNIT" column
data_raw_df.drop(index=(data_raw_df.loc[data_raw_df['TARGET ACTIVITY - UNIT'].isnull()].index), inplace=True)
#print(data_raw_df.shape)
# Dim: (70262, 18)

# convert µg/ml into µM
convert_idx = data_raw_df.loc[data_raw_df['TARGET ACTIVITY - UNIT'] == 'µg/ml'].index.tolist()
for pip_idx in convert_idx:

    # mass calculate library 1
    #pip_mass = mass.calculate_mass(sequence=data_raw_df.loc[pip_idx, 'SEQUENCE'])

    # mass calculate library 2
    pip_mass = ProteinAnalysis(data_raw_df.loc[pip_idx, 'SEQUENCE']).molecular_weight()

    data_raw_df.loc[pip_idx, 'TARGET ACTIVITY - CONCENTRATION - PROCED'] = (data_raw_df.loc[pip_idx, 'TARGET ACTIVITY - CONCENTRATION - PROCED'] * 1000) / pip_mass

data_raw_df['TARGET ACTIVITY - CONCENTRATION - PROCED - UNIT'] = ['µM']*data_raw_df.shape[0]

# Remove duplicate records #1
data_raw_df.drop_duplicates(inplace=True)
data_raw_df.reset_index(inplace=True, drop=True)
#print(data_raw_df.shape)
# Dim: (69043, 19)

# Process "TARGET ACTIVITY - TARGET SPECIES" column
# shorter TARGET SPECIES name, first 2 words
def specie_name_cut(x):
    x.strip(x)
    x_split = x.split(' ')
    if len(x_split) >= 2: x = ' '.join([x_split[0], x_split[1]])
    return x
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x:specie_name_cut(x))
# transform all specie name into lower case
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x: x.lower())
# remove "sp.", " ", "ATCC" from some specie names
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x: x.strip(' '))
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x: re.sub(r'sp.$|^sp.', '', x))
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x: x.strip(' '))
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x: re.sub(r'atcc$|^atcc', '', x))
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x: x.strip(' '))
# Convert ligature case
# Ref. https://stackoverflow.com/questions/31553324/fluphenazine-read-as-xef-xac-x82uphenazine
ligatures = {0xFB00: u'ff', 0xFB01: u'fi', 0xFB02: u'fl'}
data_raw_df['TARGET ACTIVITY - TARGET SPECIES'] = data_raw_df['TARGET ACTIVITY - TARGET SPECIES'].apply(lambda x: x.translate(ligatures))

# Remove duplicate records #2
data_raw_df.drop_duplicates(inplace=True)
data_raw_df.reset_index(inplace=True, drop=True)
#print(data_raw_df.shape)
# Dim: (62328, 19)'''

# average activity values for same TARGET ACTIVITY - TARGET SPECIES and same SEQUENCES
def dup_safe_remove(x):
    unique_prop = pd.unique(x)
    flag = unique_prop.shape[0]
    if flag==1: return unique_prop
    else: return unique_prop[0]
def none_safe_remove(x):
    if x == 'None':
        return np.NaN
    else: return x

# Count unique sequences
pip_seq_list = set(data_raw_df['SEQUENCE'].tolist())
#print(len(pip_seq_list))
# 9216 unique sequences'''

# Code: only consider SEQUENCE uniqueness, ignore N and C modification
'''count_point = 0
unique_data_csv = pd.DataFrame(columns=data_raw_df.columns)
for pip_seq in pip_seq_list:
    count_point += 1
    pip_sample = data_raw_df.loc[data_raw_df['SEQUENCE'] == pip_seq]
    if pip_sample.shape[0] > 1:
        if len(set(pip_sample['TARGET ACTIVITY - TARGET SPECIES'])) < pip_sample.shape[0]:
            unique_pip_sample = pip_sample.groupby('TARGET ACTIVITY - TARGET SPECIES').agg({
                      'ID': lambda x: dup_safe_remove(x),
                      'COMPLEXITY': lambda x: dup_safe_remove(x),
                      'NAME': lambda x: dup_safe_remove(x),
                      'N TERMINUS': lambda x: dup_safe_remove(x),
                      'SEQUENCE': lambda x: dup_safe_remove(x),
                      'C TERMINUS': lambda x: dup_safe_remove(x),
                      'SYNTHESIS TYPE': lambda x: dup_safe_remove(x),
                      'TARGET GROUP': lambda x: dup_safe_remove(x),
                      'TARGET OBJECT': lambda x: dup_safe_remove(x),
                      'UNUSUAL OR MODIFIED AMINO ACID - POSITION': lambda x: dup_safe_remove(x),
                      'UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE': lambda x: dup_safe_remove(x),
                      'UNUSUAL OR MODIFIED AMINO ACID - BEFORE MODIFICATION': lambda x: dup_safe_remove(x),
                      'TARGET ACTIVITY - ACTIVITY MEASURE VALUE': lambda x: dup_safe_remove(x),
                      'TARGET ACTIVITY - CONCENTRATION': lambda x: dup_safe_remove(x),
                      'TARGET ACTIVITY - UNIT': lambda x: dup_safe_remove(x),
                      'TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)': lambda x: dup_safe_remove(x),
                      'TARGET ACTIVITY - CONCENTRATION - PROCED': 'mean',
                      'TARGET ACTIVITY - CONCENTRATION - PROCED - UNIT': lambda x: dup_safe_remove(x)})
            unique_pip_sample['TARGET ACTIVITY - TARGET SPECIES'] = unique_pip_sample.index.tolist()
            unique_data_csv = unique_data_csv.append(unique_pip_sample, ignore_index=True, sort=True)
        else:
            unique_data_csv = unique_data_csv.append(pip_sample, ignore_index=True, sort=True)
    else:
        unique_data_csv = unique_data_csv.append(pip_sample, ignore_index=True, sort=True)
    print('PROCESSING SEQ: ', count_point)
print('----------')
print('TOTAL SEQ: ', len(pip_seq_list))
#print(unique_data_csv.shape)
# Dim: (47655, 20)
'''

# Code: consider both SEQUENCE, N/C modification
data_raw_df[['N TERMINUS', 'C TERMINUS']] = data_raw_df[['N TERMINUS', 'C TERMINUS']].fillna('None')
count_point = 0
unique_data_csv = pd.DataFrame(columns=data_raw_df.columns)
for pip_seq in pip_seq_list:
    print(pip_seq)
    count_point += 1
    pip_sample = data_raw_df.loc[data_raw_df['SEQUENCE'] == pip_seq]
    if pip_sample.shape[0] > 1:
        if len(set(pip_sample['TARGET ACTIVITY - TARGET SPECIES'])) < pip_sample.shape[0]:
            unique_pip_sample = pip_sample.groupby(['N TERMINUS', 'C TERMINUS', 'TARGET ACTIVITY - TARGET SPECIES']).agg({
                'ID': lambda x: dup_safe_remove(x),
                'COMPLEXITY': lambda x: dup_safe_remove(x),
                'NAME': lambda x: dup_safe_remove(x),
                'N TERMINUS': lambda x: dup_safe_remove(x),
                'SEQUENCE': lambda x: dup_safe_remove(x),
                'C TERMINUS': lambda x: dup_safe_remove(x),
                'SYNTHESIS TYPE': lambda x: dup_safe_remove(x),
                'TARGET GROUP': lambda x: dup_safe_remove(x),
                'TARGET OBJECT': lambda x: dup_safe_remove(x),
                'UNUSUAL OR MODIFIED AMINO ACID - POSITION': lambda x: dup_safe_remove(x),
                'UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE': lambda x: dup_safe_remove(x),
                'UNUSUAL OR MODIFIED AMINO ACID - BEFORE MODIFICATION': lambda x: dup_safe_remove(x),
                'TARGET ACTIVITY - ACTIVITY MEASURE VALUE': lambda x: dup_safe_remove(x),
                'TARGET ACTIVITY - CONCENTRATION': lambda x: dup_safe_remove(x),
                'TARGET ACTIVITY - UNIT': lambda x: dup_safe_remove(x),
                'TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)': lambda x: dup_safe_remove(x),
                'TARGET ACTIVITY - CONCENTRATION - PROCED': 'mean',
                'TARGET ACTIVITY - CONCENTRATION - PROCED - UNIT': lambda x: dup_safe_remove(x),
                'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION': lambda x: dup_safe_remove(x),
                'HEMOLITIC CYTOTOXIC ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)': lambda x: dup_safe_remove(x)})
            unique_pip_sample['TARGET ACTIVITY - TARGET SPECIES'] = np.array(unique_pip_sample.index.tolist())[:, 2]
            unique_data_csv = unique_data_csv.append(unique_pip_sample, ignore_index=True, sort=True)
        else:
            unique_data_csv = unique_data_csv.append(pip_sample, ignore_index=True, sort=True)
    else:
        unique_data_csv = unique_data_csv.append(pip_sample, ignore_index=True, sort=True)
    print('PROCESSING SEQ: ', count_point)

unique_data_csv['N TERMINUS'] = unique_data_csv['N TERMINUS'].apply(lambda x: none_safe_remove(x))
unique_data_csv['C TERMINUS'] = unique_data_csv['C TERMINUS'].apply(lambda x: none_safe_remove(x))
print('----------')
print('TOTAL SEQ: ', len(pip_seq_list))
#print(unique_data_csv.shape)
# Dim: (50941, 19)


# Sort columns
# Remove useless columns
# 'UNUSUAL OR MODIFIED AMINO ACID - POSITION',
# 'UNUSUAL OR MODIFIED AMINO ACID - MODIFICATION TYPE',
# 'UNUSUAL OR MODIFIED AMINO ACID - BEFORE MODIFICATION',
# 'TARGET ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)'
# 'TARGET ACTIVITY - CONCENTRATION',
# 'TARGET ACTIVITY - UNIT',
unique_data_csv = unique_data_csv[['ID', 'COMPLEXITY', 'NAME', 'N TERMINUS', 'SEQUENCE', 'C TERMINUS',
                                   'SYNTHESIS TYPE', 'TARGET GROUP', 'TARGET OBJECT',
                                   'TARGET ACTIVITY - TARGET SPECIES',
                                   'TARGET ACTIVITY - ACTIVITY MEASURE VALUE',
                                   'TARGET ACTIVITY - CONCENTRATION - PROCED',
                                   'TARGET ACTIVITY - CONCENTRATION - PROCED - UNIT',
                                   'HEMOLITIC CYTOTOXIC ACTIVITY - CONCENTRATION',
                                   'HEMOLITIC CYTOTOXIC ACTIVITY - ACTIVITY (μg/ml) (Calculated By DBAASP)'
                                   ]]
#print(unique_data_csv.shape)
# Dim: (50942, 13)

# Generate processed CSV file with total records
unique_data_csv.to_csv('./peptides-complete_processed.csv', index=False, encoding='utf_8_sig')

# Generate TARGET SPECIES index-count csv file
species_set = set(unique_data_csv['TARGET ACTIVITY - TARGET SPECIES'].tolist())
print('unique species type after processing: ', len(species_set))
species_index_df = pd.DataFrame({'name': list(species_set),
                                 'count': [len(unique_data_csv.loc[unique_data_csv['TARGET ACTIVITY - TARGET SPECIES'] == i].index.tolist()) for i in list(species_set)]})
species_index_df.to_csv('./peptides-complete_processed_species_index.csv', index=False, encoding='utf_8_sig')

# Split records - different species into different CSV files
split_data_save_folder = './split_data'
if not os.path.isdir(split_data_save_folder):
    os.mkdir(split_data_save_folder)
dict = {}
for specie_name in list(species_set):
    split_df = unique_data_csv[unique_data_csv['TARGET ACTIVITY - TARGET SPECIES'] == specie_name]
    split_df.to_csv(split_data_save_folder+'/specie_' + specie_name.replace(' ', '-') + '.csv', index=False, encoding='utf_8_sig')
    nter = sum(pd.isna(split_df["N TERMINUS"]))
    cter = sum(pd.isna(split_df["C TERMINUS"]))
    dict[specie_name] = [len(split_df), {"n terminus of nan": nter, "c terminus of nan": cter}]

# count N TERMINUS == nan
nter = sum(pd.isna(unique_data_csv["N TERMINUS"])) # 46719
cter = sum(pd.isna(unique_data_csv["C TERMINUS"])) # 25619
dict["total n terminus of nan"] = nter
dict["total c terminus of nan"] = cter
import json
with open('./species_number.json', 'w') as f:
    json.dump(dict, f)
# import matplotlib.pyplot as plt
# def plotSpecieConcentration(specie_name, unique_data_csv=unique_data_csv):
#     unit = unique_data_csv['TARGET ACTIVITY - CONCENTRATION - PROCED - UNIT'][0]
#     specie_df = unique_data_csv[unique_data_csv['TARGET ACTIVITY - TARGET SPECIES'] == specie_name]
#     mic_list = specie_df['TARGET ACTIVITY - CONCENTRATION - PROCED'].to_list()
#     plt.plot(mic_list)
#     plt.title("MIC value of %s" % specie_name)
#     plt.xlabel("The i-th sample of the specie")
#     plt.ylabel("The MIC value (%s)" % unit)
#     plt.savefig("%s_sample_value.png" % specie_name.replace(' ', '_'))
#     plt.close()
#     plt.hist(mic_list, 1000, density=True)
#     plt.title("MIC density of %s" % specie_name)
#     plt.xlabel("MIC Value (%s)" % unit)
#     plt.ylabel("Density")
#     plt.savefig("%s_density.png" % specie_name.replace(' ', '_'))
# vip_species = ["escherichia coli", "staphylococcus aureus"]
# plotSpecieConcentration(vip_species[0], unique_data_csv)
# vip_species = ["escherichia coli", "staphylococcus aureus"]
# escherichia_coli = unique_data_csv[unique_data_csv['TARGET ACTIVITY - TARGET SPECIES'] == vip_species[0]]
# staphylococcus_aureus = unique_data_csv[unique_data_csv['TARGET ACTIVITY - TARGET SPECIES'] == vip_species[1]]
print("smart")