import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from skfuzzsys import *
from train_test import *

# hyper-parameter
train_size = 0.9
learning_rate = 0.2
num_fuzzy_set = 3
max_iter = 300
batch_ratio = 0.1
tau_fea_sel_ratio = 0.5
tau_rul_ext_ratio = 0.01

# load dataset
dataset_name = r'Wine'
dataset = torch.load(fr'datasets/{dataset_name}.pt')
sam, label = dataset.sample, dataset.target

# one-hot the label
label = torch.LongTensor(preprocessing.OneHotEncoder().fit_transform(label).toarray())

# split train-test samples
tra_sam, test_sam, tra_tar, test_tar = train_test_split(sam, label, train_size=train_size)

# preprocessing, linearly normalize the training and test samples into the interval [0,1]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
tra_sam = torch.Tensor(min_max_scaler.fit_transform(tra_sam))
test_sam = torch.Tensor(min_max_scaler.transform(test_sam))

# the number of samples, features, and classes
num_tra_sam, num_fea = tra_sam.shape
num_class = tra_tar.shape[1]

# init the model
DG_TSK = TSK(num_fea, num_class, num_fuzzy_set, mf='Gaussian', tnorm='prod',
             fea_sel=True, gate_fea='gate_m', rul_ext=True, gate_rule='gate_m')
DG_TSK.reinit_system_points(tra_sam, tra_tar, spread='std_mean', max_num_rule=300)

# == training, FS and RE
DG_TSK.trained_param(tra_param='gatesTHEN')
train_full_batch(tra_sam, DG_TSK, tra_tar, learning_rate, max_iter)

# select features, extract rules
selected_fea_ind = DG_TSK.select_fea(tau_fea_sel_ratio)
extracted_rule_ind = DG_TSK.extract_rule(tau_rul_ext_ratio)
DG_TSK.prune_structure(selected_fea_ind, extracted_rule_ind)
DG_TSK.antecedent.fea_sel = False
DG_TSK.consequent.rul_ext = False

# == fine tuning
DG_TSK.consequent.con_param.data = torch.zeros_like(DG_TSK.consequent.con_param.data)
DG_TSK.trained_param(tra_param='IF_THEN')
train_full_batch(tra_sam[:, selected_fea_ind], DG_TSK, tra_tar, learning_rate, max_iter)

# == test
tra_loss, tra_acc = test(tra_sam[:, selected_fea_ind], DG_TSK, tra_tar)
test_loss, test_acc = test(test_sam[:, selected_fea_ind], DG_TSK, test_tar)
print(fr'{dataset_name} dataset, training acc: {tra_acc:.4f}, test acc: {test_acc:.4f}')
print(fr'No. of selected features: {len(selected_fea_ind)}, No. of the extracted rules: {len(extracted_rule_ind)}')
