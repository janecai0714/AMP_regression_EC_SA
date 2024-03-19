import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# pMIC: -3.95 to 2.16, set bias as 4.0 (min pMIC+bias to 0), set scale to 1/6, so the range to be (0, 1)
# min-max normalization
largest_MIC = 10000; def_bias = 4.0; def_scale = 1/6;
def getRootDir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = "/".join(root_dir.split("/")[:-1])
    return root_dir

def descale(output, bias=def_bias, scale=def_scale):
    new_out = output / scale - bias
    return new_out
class GetOriFolder():
    def __init__(self):
        self.root_dir = getRootDir()
        self.data_dir = os.path.join(self.root_dir, 'split_data')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        # self.mrg_dir = os.path.join(self.data_dir, 'merge')
        # if not os.path.isdir(self.mrg_dir):
        #     os.mkdir(self.mrg_dir)
        self.rs_dir = os.path.join(self.root_dir, 'result')
        if not os.path.isdir(self.rs_dir):
            os.mkdir(self.rs_dir)
        self.mdl_dir = os.path.join(self.root_dir, 'model')
        if not os.path.isdir(self.mdl_dir):
            os.mkdir(self.mdl_dir)
        self.pic_dir = os.path.join(self.root_dir, 'pics')
        if not os.path.isdir(self.pic_dir):
            os.mkdir(self.pic_dir)
def getOriSepcieLocation(specie_name):
    fld = GetOriFolder()
    sepcie_path = os.path.join(fld.data_dir,  "specie_%s.csv" % specie_name.replace(' ', '-'))
    return sepcie_path

def abbreviation(string):
    words = string.split(" ")
    abb = ""
    for word in words:
        first_char = word[0]
        abb = abb + first_char.capitalize()
    return abb

def controlSeqsLengthRange(infos, shorest_len=5, largerest_len=60):
    infos = infos.copy()
    seqs = infos["SEQUENCE"].to_list()
    for seq in seqs:
        l = len(seq)
        if l < shorest_len or l > largerest_len:
            infos = infos[infos["SEQUENCE"] != seq]
    return infos

def controlMICvalue(infos, largest_MIC=largest_MIC):
    infos = infos.copy()
    infos = infos[infos['TARGET ACTIVITY - CONCENTRATION - PROCED'] < largest_MIC]
    return infos

def printDataInfo(first_line, path, data, colPrint=False):
    print(first_line)
    print("\tPath: %s" % path)
    print("\tSample No.: %d" % len(data))
    if colPrint:
        print("\tFeature No.: %d" % data.shape[-1])
    return
class GetOriSpecieInfo():
    def __init__(self, specie_name="escherichia coli", control_seqLens=True, control_MIC=True, largest_MIC=largest_MIC):
        self.specie_path = getOriSepcieLocation(specie_name)
        self.specie_abb = abbreviation(specie_name)
        specie_info = pd.read_csv(self.specie_path)
        # only extract seqs whose N and C terminus are np.nan
        ind = pd.isna(specie_info["N TERMINUS"]) & pd.isna(specie_info["C TERMINUS"])
        specie_info = specie_info[ind]
        if control_seqLens:
            specie_info = controlSeqsLengthRange(specie_info, shorest_len=5, largerest_len=60)
        if control_MIC:
            specie_info = controlMICvalue(specie_info, largest_MIC=largest_MIC)
        self.specie_info = specie_info
        # add space to aa in sequences
        self.SEQUENCE_space = [ " ".join(ele) for ele in self.specie_info["SEQUENCE"]]
        self.specie_info['SEQUENCE_space'] = self.SEQUENCE_space

        self.pMIC_name = '%s_pMIC' % self.specie_abb
        self.pMIC_scale_name = '%s_pMIC_scale' % self.specie_abb
        self.MIC_name = '%s_MIC' % self.specie_abb

        self.pMIC = -np.log10(self.specie_info['TARGET ACTIVITY - CONCENTRATION - PROCED'])
        self.pMIC_scale = (self.pMIC + def_bias) * def_scale

        self.specie_info[self.pMIC_name] = self.pMIC
        self.specie_info[self.pMIC_scale_name] = self.pMIC_scale
        self.mic = self.specie_info['TARGET ACTIVITY - CONCENTRATION - PROCED'].to_list()
        self.specie_info[self.MIC_name] = self.mic
        fastas = self.specie_info[["ID", "SEQUENCE", "SEQUENCE_space", self.MIC_name, self.pMIC_name, self.pMIC_scale_name]]
        self.fastas = fastas
        self.colnames = fastas.columns.to_list()
        printDataInfo("%s Info: " % specie_name, self.specie_path, self.specie_info)
class GetFolder(GetOriFolder):
    def __init__(self):
        GetOriFolder.__init__(self)
        self.data_dir = os.path.join(self.root_dir, 'data')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        self.log_dir = os.path.join(self.root_dir, 'log')
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)


def getSpeciePath(specie_name):
    fld = GetFolder()
    abb = abbreviation(specie_name)
    sepcie_path = os.path.join(fld.data_dir,  "%s.csv" % abb)
    return sepcie_path, abb

def createFolder(folder, slient=True):
    if not os.path.isdir(folder):
        os.makedirs(folder)
        print("created folder: \n\t %s" % folder)
    else:
        if not slient:
            print("%s existed." % folder)
    return

class handleInfos():
    def __init__(self, infos1, infos2):
        self.infos1, self.infos2 = infos1, infos2

    def infos1DelInfos2(self):
        infos1, infos2 = self.infos1, self.infos2
        seqs1 = infos1["SEQUENCE"].to_list()
        seqs2 = infos2["SEQUENCE"].to_list()
        infos = infos1.copy()
        for seq in seqs2:
            if seq in seqs1:
                infos = infos[infos["SEQUENCE"] != seq]
        return infos
def splitTestFromAll(file_path, x=10, rep=0, valDoNotRep=True):
    spl_path = file_path.split("/")
    if valDoNotRep:
        rep_dir = os.path.join("/".join(spl_path[:-1]), str(rep))
    else:
        rep_dir = os.path.join("/".join(spl_path[:-1]), "test%d" % rep)
    createFolder(rep_dir)
    train_path = os.path.join(rep_dir, "train-%s" % (spl_path[-1]))
    test_path = os.path.join(rep_dir, "test-%s" % (spl_path[-1]))
    df = pd.read_csv(file_path)
    test = df.sample(frac=x/100, replace=False).copy()
    train = handleInfos(df, test).infos1DelInfos2()
    test.to_csv(test_path)
    print("all sample: ", len(df), "\ntrain sample: ", len(train), "\ntest sample: ", len(test))
    print("saved %d percent samples of trainset (%s): \n\t%s" % (x, spl_path[-1], test_path))
    train.to_csv(train_path)
    print("saved %d percent samples of trainset (%s): \n\t%s" % (100-x, spl_path[-1], train_path))
    return train_path, test_path

def splitValidationFromTrain(file_path, x=10, rep=0, valDoNotRep=True):
    spl_path = file_path.split("/")
    if not valDoNotRep:
        rep_dir = os.path.join("/".join(spl_path[:-1]), "val%d" % rep)
    else:
        rep_dir = os.path.join("/".join(spl_path[:-1]))
    createFolder(rep_dir)
    tra_path = os.path.join(rep_dir, "tra-%s" % (spl_path[-1]))
    val_path = os.path.join(rep_dir, "val-%s" % (spl_path[-1]))
    df = pd.read_csv(file_path)
    val = df.sample(frac=x/100, replace=False).copy()
    train = handleInfos(df, val).infos1DelInfos2()
    val.to_csv(val_path)
    print("train sample: ", len(df), "\ntra train sample: ", len(train), "\nval train sample: ", len(val))
    print("saved %d percent samples of trainset (%s): \n\t%s" % (x, spl_path[-1], val_path))
    train.to_csv(tra_path)
    print("saved %d percent samples of trainset (%s): \n\t%s" % (100-x, spl_path[-1], tra_path))
    return tra_path, val_path

def PreProcessedSingleSpecies(specie="escherichia coli", x=10, reps=5, valDoNotRep=True):
    """
    control the sequence length in [5,60], MIC value < 1000 μM;
    split the single data to train (tra-train/val-train) / test datasets ratio = 9:1 randomly
    :param specie: specie name
    :return:
    """
    # read original host specie info and save fastas to data folder
    paths = {}
    spe = GetOriSpecieInfo(specie)
    path, abb = getSpeciePath(specie)
    dir = os.path.dirname(path)
    createFolder(dir)
    spe.fastas.to_csv(path, index=False)
    paths[abb] = path
    print("saved %s fastas to: \n\t%s" % (specie, path))
    if valDoNotRep:
        for i in range(reps):
            train_path, test_path = splitTestFromAll(path, x=x, rep=i, valDoNotRep=valDoNotRep)
            tra_path, val_path = splitValidationFromTrain(train_path, x=x, rep=i, valDoNotRep=valDoNotRep)
            paths["%s-tra-train%d" % (abb, i)] = tra_path
            paths["%s-val-train%d" % (abb, i)] = val_path
            paths["%s-train%d" % (abb, i)] = train_path
            paths["%s-test%d" % (abb, i)] = test_path
    else:
        for i in range(reps):
            train_path, test_path = splitTestFromAll(path, x=x, rep=i, valDoNotRep=valDoNotRep)
            for j in range(reps):
                tra_path, val_path = splitValidationFromTrain(train_path, x=x, rep=j, valDoNotRep=valDoNotRep)
                paths["%s-tra%d-train%d" % (abb, j, i)] = tra_path
                paths["%s-val%d-train%d" % (abb, j, i)] = val_path
            paths["%s-train%d" % (abb, i)] = train_path
            paths["%s-test%d" % (abb, i)] = test_path
    return paths

specie = "escherichia coli"
paths = PreProcessedSingleSpecies(specie=specie)
print("smart")