#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
from collections import Counter

def AAC(fastas, **kw):
	AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
	#AA = 'ARNDCQEGHILKMFPSTWYV'
	encodings = []
	header = ['#']
	for i in AA:
		header.append(i)
	encodings.append(header)

	for i in fastas:
		name, sequence = i[0], re.sub('-', '', i[1])
		count = Counter(sequence)
		for key in count:
			count[key] = count[key]/len(sequence)
		code = [name]
		for aa in AA:
			code.append(count[aa])
		encodings.append(code)
	return encodings

if __name__ == "__main__":
	from readFasta import readFasta
	path = "/home/mb95537/acp-design/ion_channels/uniprot-preprocessed-dataset/calcium.fasta"
	fastas = readFasta(path)
	encdn = AAC(fastas, order = None)
	import pandas as pd
	df = pd.DataFrame(encdn)
	df.index = df.iloc[:, 0]
	df.columns = df.iloc[0]
	df.drop(["#"], axis=1, inplace=True)
	df.drop(["#"], axis=0, inplace=True)
	print("%s's feature number: %d" % (AAC.__name__, len(df.columns)) )
	# add class
	df["default"] = [1] * len(fastas)