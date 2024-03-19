import pandas as pd

column = ["ID", "SEQUENCE", "LOG_MIC"]
csv_path = "/home/jianxiu/Documents/amp_regression/ecoli_stl/in_data_novel/test_ecoli75.csv"
seq_data = pd.read_csv(csv_path, usecols= column)
sentences = []
for i in range(len(seq_data.SEQUENCE)):
    sentence = " ".join(seq_data.SEQUENCE[i])
    sentences.append(sentence)
seq_data['with_space'] = sentences
seq_data.to_csv(csv_path)
print("smart")