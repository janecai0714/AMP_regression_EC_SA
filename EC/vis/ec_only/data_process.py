import pandas as pd

ec = pd.read_csv("/home/jianxiu/Documents/EC/vis/ec_only/EC.csv")
sa = pd.read_csv("/home/jianxiu/Documents/EC/vis/ec_only/SA.csv")

ec_only = ec.loc[pd.merge(ec, sa, on=['ID','SEQUENCE'], how='left', indicator=True)['_merge'] == 'left_only']
sa_only = sa.loc[pd.merge(sa, ec, on=['ID','SEQUENCE'], how='left', indicator=True)['_merge'] == 'left_only']
print("smart")