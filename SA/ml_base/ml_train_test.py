from ifeature.codes import *
from ifeature.PseKRAAC import *
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr, kendalltau
from sklearn.model_selection import GridSearchCV

def isDataFrame(obj):
    if type(obj) == pd.core.frame.DataFrame:
        return True
    else:
        return False

def getTrueFastas(fastas):
    if isDataFrame(fastas):
        fts = fastas[["ID", "SEQUENCE"]]
        return fts.to_numpy()
    else:
        return fastas

codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'y']
def one_hot_encode(seq, max_len=60):
    # o = list(set(codes) - set(seq))
    s = pd.DataFrame(list(seq))
    l = len(s)
    if max_len < l:
        max_len = l
    x = pd.DataFrame(np.zeros((max_len, 20), dtype=int), columns=codes)
    a = s[0].str.get_dummies(sep=',')
    # a = a.join(x)
    # a = a.sort_index(axis=1)
    b = x + a
    b = b.replace(np.nan, 0)
    b = b.astype(dtype=int)
    # e = a.values.flatten()
    return b

def oneHot(fastas, max_len=60, class_val=None):
    fastas = getTrueFastas(fastas)
    fts = []
    names = fastas[:,0]
    for seq in fastas[:,1]:
        if len(seq) > max_len:
            continue
        # print("seq: ", seq)
        e = one_hot_encode(seq, max_len=max_len)
        e = e.values.flatten()
        fts.append(e)
    df = pd.DataFrame(fts)
    df.index = names
    # df.columns = df.iloc[0]
    print("Gene oneHot:")
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

#subtype = {'g-gap': 0, 'lambda-correlation': 4}
#subtype = 'g-gap' or 'lambda-correlation'
def genePsekraac(fastas, ft_name="type1", raactype=2, subtype='lambda-correlation', ktuple=2, gap_lambda=1, class_val=None):
    # fastas = readFasta.readFasta(path)
    # gap_lambda = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    fastas = getTrueFastas(fastas)
    #type1(fastas, subtype, raactype, ktuple, glValue)
    eval_func = "%s.type1(fastas, subtype=subtype, raactype=raactype, ktuple=ktuple, glValue=gap_lambda)" % (ft_name)
    print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    print("feature number of PseKRAAC.%s(%s, raac_type=%d, ktuple=%d, gap_lambda=%d): %d" %
          (ft_name, subtype, raactype, ktuple, gap_lambda, len(df.columns)))
    ft_whole_name = "%sraac%s" % (ft_name, raactype)
    print("Gene %s :" % (ft_whole_name))
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def GeneIfeature(fastas, ft_name="AAC", gap=0, nlag=4, lambdaValue=4, class_val=None):
    # fastas = readFasta.readFasta(path)
    # CKSAAP: gap = 0, 1, 2, 3 (3 = min sequence length - 2)
    # SOCNumber QSOrder PAAC APAAC: lambdaValue = 0, 1, 2, 3, 4 (4 = min sequence length - 1)
    # NMBroto: nlag= 2, 3, 4
    #fastas = getTrueFastas(fastas)
    fastas = readFasta.readFasta(fastas)
    eval_func = "%s.%s(fastas, gap=%d, order=None, nlag=%d, lambdaValue=%d)" % (ft_name, ft_name, gap, nlag, lambdaValue)
    print(eval_func)
    encdn = eval(eval_func)
    df = pd.DataFrame(encdn)
    df.index = df.iloc[:, 0]
    df.columns = df.iloc[0]
    df.drop(["#"], axis=1, inplace=True)
    df.drop(["#"], axis=0, inplace=True)
    # print("%s's feature number: %d" % (ft_name, len(df.columns)))
    print("Gene %s :" % (ft_name))
    row, col = df.shape
    print("\t\t its class value: %s" % (str(class_val)))
    print("\t\t its input No.: %s" % row)
    print("\t\t its feature No.: %s" % col)
    # add class
    if class_val != None:
        df["default"] = [class_val] * len(fastas)
    return df

def svm_para(x_train, y_train):
    param_grid = {'C': [0.1, 1, 10, 100],
                  'gamma': [1,0.1,0.01, 0.001],
                  'kernel': ['rbf', 'poly', 'linear', 'sigmoid']}
    grid = GridSearchCV(SVR(), param_grid,scoring='neg_mean_squared_error',cv=2,verbose=3)
    grid.fit(x_train, y_train)
    return grid.best_estimator_
def rf_para(x_train, y_train):
    param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9]}
    grid = GridSearchCV(RandomForestRegressor(), param_grid,scoring='neg_mean_squared_error',cv=2,verbose=3)
    grid.fit(x_train, y_train)
    return grid.best_estimator_

def df_para(x_train, y_train):
    param_grid = {"splitter":["best","random"],
            "max_depth" : [1,3,5,7,9,11],
           "min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
           "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5],
           "max_features":["auto","log2","sqrt",None],
           "max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90] }
    grid = GridSearchCV(DecisionTreeRegressor(), param_grid,scoring='neg_mean_squared_error',cv=2,verbose=3)
    grid.fit(x_train, y_train)
    return grid.best_estimator_
def knn_para(x_train, y_train):
    param_grid = { 'n_neighbors' : [3, 5,7,9,11,13,15],
               'weights' : ['uniform','distance'],
               'metric' : ['minkowski','euclidean','manhattan']}
    grid = GridSearchCV(KNeighborsRegressor(), param_grid,scoring='neg_mean_squared_error',cv=2,verbose=3)
    grid.fit(x_train, y_train)
    return grid.best_estimator_
def ml_train_test(x_train, y_train, x_test, y_test, regressor_name):
    # if regressor_name == "SVM":
    #     regressor = svm_para(x_train, y_train)
    # elif regressor_name == "RF":
    #     regressor = rf_para(x_train, y_train)
    # elif regressor_name == "DT":
    #     regressor = df_para(x_train, y_train)
    # elif regressor_name == "KNN":
    #     regressor = knn_para(x_train, y_train)

    if regressor_name == "SVM":
        regressor = SVR()
    elif regressor_name == "RF":
        regressor = RandomForestRegressor()
    elif regressor_name == "DT":
        regressor = DecisionTreeRegressor()
    elif regressor_name == "KNN":
        regressor = KNeighborsRegressor()
    regressor.fit(x_train, y_train)
    preds = regressor.predict(x_test)
    reg_mse = mean_squared_error(y_test, preds)
    reg_r2 = r2_score(y_test, preds)
    reg_pcc = pearsonr(y_test, preds)[0]
    reg_ktc = kendalltau(y_test, preds)[0]
    print("smart")
    return reg_mse, reg_r2, reg_pcc, reg_ktc

def csv2fasta_func(csv_path, fasta_path):
    # get .csv info
    seq_data = pd.read_csv(csv_path)
    # .csv to .fasta
    fast_file = open(fasta_path, "w")
    for i in range(len(seq_data.SEQUENCE)):
        fast_file.write(">" + str(seq_data.ID[i]) + "\n")
        fast_file.write(seq_data.SEQUENCE[i] + "\n")
    fast_file.close()


if __name__ == "__main__":
    max_num_split = 5
    feature_name_list = ["CTDD"] #"QSOrder", "CTDC", "CTDT"
    regressor_name_list = ["SVM", "RF", "DT", "KNN"] #, "RF", "DT", "KNN"

    data_list, encoding_list, regressor_list, mse_list, r2_list, pcc_list, ktc_list = [], [], [], [], [], [], []
    for i in range(max_num_split):
        train_csv = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC.csv"
        train_fasta = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC.fasta"
        test_csv = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC.csv"
        test_fasta = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC.fasta"

        csv2fasta_func(train_csv, train_fasta)
        csv2fasta_func(test_csv, test_fasta)

        train_df = pd.read_csv(train_csv)
        y_train = train_df["EC_pMIC"].values
        test_df = pd.read_csv(test_csv)
        y_test = test_df["EC_pMIC"].values
        for feature_name in feature_name_list:
            # train_ft_df = GeneIfeature(train_fasta, ft_name=feature_name)
            # test_ft_df = GeneIfeature(test_fasta, ft_name=feature_name)
            train_ft_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/train-EC_" + feature_name + ".pkl"
            test_ft_path = "/home/jianxiu/Documents/EC/data/" + str(i) + "/test-EC_" + feature_name + ".pkl"

            # train_ft_df.to_pickle(train_ft_path)
            # test_ft_df.to_pickle(test_ft_path)

            train_ft = pd.read_pickle(train_ft_path)
            test_ft = pd.read_pickle(test_ft_path)
            for regressor_name in regressor_name_list:
                reg_mse, reg_r2, reg_pcc, reg_ktc = ml_train_test(train_ft, y_train, test_ft, y_test, regressor_name)
                data_list.append(str(i))
                encoding_list.append(feature_name)
                regressor_list.append(regressor_name)

                mse_list.append(reg_mse)
                r2_list.append(reg_r2)
                pcc_list.append(reg_pcc)
                ktc_list.append(reg_ktc)
        print("smart")
    ml_path = "/home/jianxiu/Documents/EC/mse_loss_func/ml_result_withoutCTDD.csv"
    ml_result_dict = {'data': data_list, "encoding": encoding_list, "regressor": regressor_list,
                        "mse": mse_list, "r2": r2_list, "pcc": pcc_list, "ktc":ktc_list}
    ml_result_df = pd.DataFrame(ml_result_dict)
    ml_result_df.to_csv(ml_path)
print("smart")
