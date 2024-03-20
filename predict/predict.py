from model_def import REG
from seq_dataloader import *
from seq_dataloader import _get_train_data_loader, _get_test_data_loader, freeze
import collections
from onedrivedownloader import download
import os
import warnings
warnings.filterwarnings('ignore')

def read_fasta_file(fasta_path, csv_path):
    f = open(fasta_path, "r")
    seq = collections.OrderedDict()
    for line in f:
        if line.startswith(">"):
            name = line.split()[0]
            seq[name] = ''
        else:
            seq[name] += line.replace("\n", '').strip()
    f.close()
    seq_df = pd.DataFrame(seq.items(), columns=['ID', 'SEQUENCE'])
    seq_df["SEQUENCE_space"] = [" ".join(ele) for ele in seq_df["SEQUENCE"]]
    seq_df.to_csv(csv_path)
    return seq_df
def predict(ec_model, sa_model, fasta_path, csv_path):
    batch_size = 500
    seq = read_fasta_file(fasta_path, csv_path)
    test_loader = _get_test_data_loader(batch_size, csv_path)
    ec_predict_list, sa_predict_list = [],[]
    ec_model.eval()
    sa_model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids']
            b_input_mask = batch['attention_mask']
            ec_predict_pMIC, _ = ec_model(b_input_ids, attention_mask=b_input_mask)
            ec_predict_list.extend(ec_predict_pMIC.data.numpy())

            sa_predict_pMIC, _ = sa_model(b_input_ids, attention_mask=b_input_mask)
            sa_predict_list.extend(sa_predict_pMIC.data.numpy())
    ec_predict_list = [item for sublist in ec_predict_list for item in sublist]
    sa_predict_list = [item for sublist in sa_predict_list for item in sublist]
    seq["ec_predicted_pMIC"] = ec_predict_list
    seq["sa_predicted_pMIC"] = sa_predict_list
    seq.to_csv(csv_path, index=False)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ec_url = "https://ipmedumo-my.sharepoint.com/:u:/g/personal/p2214906_mpu_edu_mo/EQNDcFHVJ9VAll_hT1hvrYkBcTV_WHX8nrBphv_xf_-FXw?e=Dgc9PB"
    ec_model_path = "ec_prot_bert_finetune_reproduce.pkl"
    if not os.path.exists(ec_model_path):
        download(ec_url, ec_model_path)
    ec_model = REG()
    ec_model.load_state_dict(torch.load(ec_model_path))

    sa_url ="https://ipmedumo-my.sharepoint.com/:u:/g/personal/p2214906_mpu_edu_mo/EUS21YkRlMBGuGJuaxAFqt0BFDcItYnD9RmvUlGL3LkhGA?e=FD2UUK"
    sa_model_path = "sa_prot_bert_finetune_reproduce.pkl"
    if not os.path.exists(sa_model_path):
        download(sa_url, sa_model_path)
    sa_model = REG()
    sa_model.load_state_dict(torch.load(sa_model_path))

    fasta_path = "train_po.fasta"
    csv_path = "train_po.csv"
    predict(ec_model, sa_model, fasta_path, csv_path)
    print('smart')
