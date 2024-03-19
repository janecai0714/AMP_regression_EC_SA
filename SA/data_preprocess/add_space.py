import pandas as pd

csv_path = "/home/jianxiu/Documents/SA/data/SA.csv"
seq_data = pd.read_csv(csv_path)
sentences = []
for i in range(len(seq_data.SEQUENCE)):
    sentence = " ".join(seq_data.SEQUENCE[i])
    sentences.append(sentence)
seq_data['with_space'] = sentences
seq_data.to_csv(csv_path)
print("smart")