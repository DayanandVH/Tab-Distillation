import time
import os
import numpy as np
import random

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_curve, roc_auc_score, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import average_precision_score

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP

seed= 42


# Convert Categorical features to one-hot encoding and Numerical Features to Normalization
def preprocess_mixdata(train_data, test_data, trainy, testy, cat_features, num_features):
    all_cat_data = pd.concat([train_data[cat_features], test_data[cat_features]])
    all_cat_data_transformed = pd.get_dummies(all_cat_data.astype(str))
    cat_train_transformed = all_cat_data_transformed.iloc[:len(train_data)]
    cat_train_transformed.reset_index(drop=True)
    cat_test_transformed = all_cat_data_transformed.iloc[len(train_data):]
    cat_test_transformed.reset_index(drop=True)
    data_trainX = pd.concat([cat_train_transformed, train_data[num_features]], axis=1)
    data_testX = pd.concat([cat_test_transformed, test_data[num_features]], axis=1)
    scaler = StandardScaler()
    scaler.fit(data_trainX[num_features])
    num_trainX_transformed = scaler.transform(data_trainX[num_features])
    num_trainX_transformed = pd.DataFrame(num_trainX_transformed, columns=num_features)
    num_testX_transformed = scaler.transform(data_testX[num_features])
    num_testX_transformed = pd.DataFrame(num_testX_transformed, columns=num_features)
    cat_trainX = data_trainX.drop(num_features, axis=1, inplace=False)
    cat_trainX = cat_trainX.reset_index(drop=True)
    cat_testX = data_testX.drop(num_features, axis=1, inplace=False)
    cat_testX = cat_testX.reset_index(drop=True)

    trainX = pd.concat([cat_trainX, num_trainX_transformed], axis=1)
    testX = pd.concat([cat_testX, num_testX_transformed], axis=1)

    return trainX, testX, trainy, testy

# Numerical Features to Normalization
def preprocess_numdata(train_data, test_data, trainy, testy):
    scaler = StandardScaler()
    scaler.fit(train_data)
    trainX = scaler.transform(train_data)
    testX = scaler.transform(test_data)

    return trainX, testX, trainy, testy


# Read the original dataset for distillation, preprocess it, do a train-test split and also get other information like number of classes and class labels. Also read the Random Selection
def get_tabular_dataset(dataset, run_no, type='Original', class_balance=False, no_samples=None):

    train_set = pd.read_csv('data/'+str(dataset)+'/run'+str(run_no)+'/'+str(dataset)+'_train_set.csv')
    test_set = pd.read_csv('data/'+str(dataset)+'/run'+str(run_no)+'/'+str(dataset)+'_test_set.csv')
    cat_uniqval_count = []
    if dataset == 'Credit_Default' or dataset == 'Credit_Fraud' or dataset == 'Census_Income' or dataset == 'Adult_Data' or dataset == 'Bank_Marketing' or dataset == 'KDD_Cup' or dataset == 'IEEE_Fraud':
        num_classes = 2
        if dataset == 'Credit_Default':
            num_features = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'LIMIT_BAL', 'default payment next month']
            cat_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        if dataset == 'Credit_Fraud':
            num_features = list(train_set.columns)
            cat_features = []
        if dataset == 'Census_Income':
            num_features = ['age_n', 'attr6_n', 'attr17_n', 'attr18_n', 'attr19_n', 'attr25_n', 'attr31_n', 'attr40_n', 'class']
            cat_features = ['att2_c', 'att3_c', 'att4_c', 'att5_c', 'att7_c', 'att8_c', 'att9_c', 'att10_c', 'att11_c', 'att12_c', 'att13_c', 'att14_c', 'att15_c', 'att16_c', 'att20_c', 'att21_c', 'att22_c', 'att23_c', 'att24_c', 'att26_c', 'att27_c', 'att28_c', 'att29_c', 'att30_c', 'att32_c', 'att33_c', 'att34_c', 'att35_c', 'att36_c', 'att37_c', 'att38_c', 'att39_c', 'att41_c']
        if dataset == 'Adult_Data':
            num_features = ['age', 'fnlwgt', 'Education-num', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Class']
            cat_features = ['workclass', 'education', 'Marital-status', 'occupation', 'relationship', 'race', 'sex', 'Native-country']
        if dataset == 'Bank_Marketing':
            num_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y']
            cat_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
        if dataset == 'KDD_Cup':
            # Dropping or Not including 'is_host_login' feature name in categorical featues as it contains only one categorical value '0'
            num_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                            'num_compromised', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                            'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                            'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                            'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'outcome']
            cat_features = ['protocol_type', 'service', 'flag', 'land', 'logged_in', 'root_shell', 'su_attempted', 'is_guest_login']
        if dataset == 'IEEE_Fraud':
            num_features = ['V' + str(i) for i in range(1, 340)] + ['isFraud']
            cat_features = ['id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4', 'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']

        train_set = train_set[cat_features + num_features]
        for i in num_features: train_set[i] = train_set[i].astype('float64')
        for i in cat_features: train_set[i] = train_set[i].astype('str')

        test_set = test_set[cat_features + num_features]
        for i in num_features: test_set[i] = test_set[i].astype('float64')
        for i in cat_features: test_set[i] = test_set[i].astype('str')

        if dataset == 'Credit_Fraud': cat_uniqval_count = []
        else: cat_uniqval_count = [len(set(list(train_set[i].values))) for i in cat_features]

        data_trainX = train_set.iloc[:, :-1]
        trainy = train_set.iloc[:, -1]
        data_testX = test_set.iloc[:, :-1]
        testy = test_set.iloc[:, -1]

        # Transform Numerical features in the Data
        if dataset == 'Credit_Fraud': trainX, testX, trainy, testy = preprocess_numdata(data_trainX, data_testX, trainy, testy)
        # Transform the cat and Numerical features in the Data
        else: trainX, testX, trainy, testy = preprocess_mixdata(data_trainX, data_testX, trainy, testy, cat_features, num_features[:-1])

        if type=='Random' and class_balance==False: _, trainX, _, trainy = train_test_split(trainX, trainy, test_size=no_samples*num_classes, random_state=seed, stratify=trainy)
        if type == 'Random' and class_balance == True:
            for i in range(num_classes):
                if len(trainX[trainy == i]) > no_samples:
                    index = np.random.choice(len(trainX[trainy == i]), no_samples, replace=False)
                    if i == 0:
                        temp_trainX1 = np.array(trainX[trainy == i])
                        temp_trainX = temp_trainX1[[index]]
                        temp_trainy1 = np.array(trainy[trainy == i])
                        temp_trainy = temp_trainy1[index]
                    else:
                        temp_trainX1 = np.array(trainX[trainy == i])
                        temp_trainy1 = np.array(trainy[trainy == i])
                        temp_trainX = np.vstack((temp_trainX, np.array(temp_trainX1[[index]])))
                        temp_trainy = np.hstack((temp_trainy, np.array(temp_trainy1[index])))
                # if available samples per class is less selection number of samples then select all
                else:
                    if i == 0:
                        temp_trainX = np.array(trainX[trainy == i])
                        temp_trainy = np.array(trainy[trainy == i])
                    else:
                        temp_trainX1 = np.array(trainX[trainy == i])
                        temp_trainy1 = np.array(trainy[trainy == i])
                        temp_trainX = np.vstack((temp_trainX, np.array(temp_trainX1)))
                        temp_trainy = np.hstack((temp_trainy, np.array(temp_trainy1)))
            trainX, trainy = temp_trainX, temp_trainy
        trainX, testX, trainy, testy = np.array(trainX), np.array(testX), np.array(trainy), np.array(testy)
        # print('trainX:{}, testX:{}, trainy:{}, testy:{}: '.format(trainX.shape, testX.shape, trainy.shape, testy.shape))
        dst_train, dst_test = np.hstack((trainX, trainy.reshape(-1,1))), np.hstack((testX, testy.reshape(-1,1)))
        class_names = [str(c) for c in range(num_classes)]


    if dataset == 'Covertype':
        num_features = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Cover_Type']
        cat_features = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
                        'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
        num_classes = 7

        train_set = train_set[cat_features + num_features]
        for i in num_features: train_set[i] = train_set[i].astype('float64')
        data_trainX = train_set.iloc[:, :-1]
        trainy = train_set.iloc[:, -1]

        test_set = test_set[cat_features + num_features]
        for i in num_features: test_set[i] = test_set[i].astype('float64')
        data_testX = test_set.iloc[:, :-1]
        testy = test_set.iloc[:, -1]

        # Transform the Numerical features in the Data
        num_trainX_transformed, num_testX_transformed, trainy, testy = preprocess_numdata(data_trainX[num_features[:-1]], data_testX[num_features[:-1]], trainy, testy)
        num_trainX_transformed = pd.DataFrame(num_trainX_transformed, columns=num_features[:-1])
        num_testX_transformed = pd.DataFrame(num_testX_transformed, columns=num_features[:-1])
        num_trainX_transformed = num_trainX_transformed.reset_index(drop=True)
        num_testX_transformed = num_testX_transformed.reset_index(drop=True)
        trainX = pd.concat([data_trainX[cat_features], num_trainX_transformed], axis=1)
        testX = pd.concat([data_testX[cat_features], num_testX_transformed], axis=1)
        trainX, testX, trainy, testy = np.array(trainX), np.array(testX), np.array(trainy), np.array(testy)

        if type == 'Random' and class_balance==False: _, trainX, _, trainy = train_test_split(trainX, trainy, test_size=no_samples, random_state=seed, stratify=trainy)
        if type == 'Random' and class_balance == True:
            for i in range(num_classes):
                index = np.random.choice(len(trainX[trainy == i]), no_samples, replace=False)
                if i == 0:
                    temp_trainX1 = np.array(trainX[trainy == i])
                    temp_trainX = temp_trainX1[[index]]
                    temp_trainy1 = np.array(trainy[trainy == i])
                    temp_trainy = temp_trainy1[[index]]
                else:
                    temp_trainX1 = np.array(trainX[trainy == i])
                    temp_trainy1 = np.array(trainy[trainy == i])
                    temp_trainX = np.vstack((temp_trainX, np.array(temp_trainX1[[index]])))
                    temp_trainy = np.hstack((temp_trainy, np.array(temp_trainy1[[index]])))
            trainX, trainy = temp_trainX, temp_trainy
        # print('trainX:{}, testX:{}, trainy:{}, testy:{}: '.format(trainX.shape, testX.shape, trainy.shape, testy.shape))
        dst_train, dst_test = np.hstack((trainX, trainy.reshape(-1,1))), np.hstack((testX, testy.reshape(-1,1)))
        class_names = [str(c) for c in range(num_classes)]

    return num_classes, class_names, dst_train, dst_test, cat_uniqval_count


class TensorDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return self.samples.shape[0]


def get_network(model, hidden_size, act_fun_name):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)

    if model == 'MLP':
        net = MLP(hidden_size=hidden_size, act_fun_name=act_fun_name)
    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, in_acc_avg, out_acc_avg, bal_in_out_acc_avg, num_exp = 0, 0, 0, 0, 0, 0
    classwise_pred_prob_mean = np.array([0 for class_name in range(args.num_classes)], dtype='float64')
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    in_acc_list, out_acc_list, batchwise_f1, batchwise_auc, batchwise_roc_auc = [], [], [], [], []
    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        #### Only use it for BCE loss ####
        # output = output.argmax(dim=1, keepdim=True).squeeze(1).float()
        #### End for BCE Loss ############
        loss = criterion(output, lab)
        # acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        lab = lab.cpu().data.numpy()
        batch_classwise_pred_prob_mean = [0 for class_name in range(args.num_classes)]
        output_pred_prob = output.cpu().data.numpy()
        outlierness_score = output.cpu().data.numpy()[:, 1]
        output_pred_prob = np.true_divide(output_pred_prob, output_pred_prob.sum(axis=1, keepdims=True))
        output = np.argmax(output.cpu().data.numpy(), axis=-1)

        classwise_acc_list = []
        for class_name in range(args.num_classes):
            if len(lab[lab==class_name]) > 0:
                classwise_acc_list.append(accuracy_score(output[lab==class_name], lab[lab==class_name]))
                batch_classwise_output_pred_prob = output_pred_prob[lab==class_name]
                batch_classwise_pred_prob_mean[class_name] = np.mean(batch_classwise_output_pred_prob[:,class_name])
        bal_in_out_acc = np.mean(classwise_acc_list)

        if args.num_classes == 2:
            if len(lab[lab == 0]) > 0: in_acc_list.append(accuracy_score(output[lab == 0], lab[lab == 0]))
            if len(lab[lab == 1]) > 0: out_acc_list.append(accuracy_score(output[lab == 1], lab[lab == 1]))
            model_precision, model_recall, _ = precision_recall_curve(np.array(lab), np.array(outlierness_score), pos_label=1)
            batchwise_f1.append(f1_score(lab, output))
            if len(lab[lab == 0]) > 0 and len(lab[lab == 1]) > 0: batchwise_auc.append(auc(model_recall, model_precision))
            if len(lab[lab == 0]) > 0 and len(lab[lab == 1]) > 0: batchwise_roc_auc.append(roc_auc_score(lab, outlierness_score))

        else:
            in_acc_list.append(0)
            out_acc_list.append(0)
            batchwise_f1.append(f1_score(lab, output, average='micro'))
            batchwise_auc.append(0)
            batchwise_roc_auc.append(0)

        loss_avg += loss.item()*n_b

        bal_in_out_acc_avg += bal_in_out_acc
        classwise_pred_prob_mean += np.array(batch_classwise_pred_prob_mean, dtype='float64')

        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    bal_in_out_acc_avg /= len(dataloader)
    classwise_pred_prob_mean /= len(dataloader)

    # if args.num_classes == 2:
    mean_in_acc_avg = np.mean(in_acc_list)
    mean_out_acc_avg = np.mean(out_acc_list)
    model_f1 = np.mean(batchwise_f1)
    model_auc = np.mean(batchwise_auc)
    model_roc_auc = np.mean(batchwise_roc_auc)

    return loss_avg, acc_avg, bal_in_out_acc_avg, classwise_pred_prob_mean, mean_in_acc_avg, mean_out_acc_avg, model_f1, model_auc, model_roc_auc


def evaluate_synset(it_eval, net, samples_train, labels_train, dsttest, args):
    net = net.to(args.device)
    samples_train = samples_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    if args.optimizer_name == 'SGD': optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=args.momemntum_img, weight_decay=0.0005)
    else: optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(args.device)
    # criterion = nn.BCELoss().to(args.device)

    dst_train = TensorDataset(samples_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train, bal_acc_train, mean_pred_prob_train, _, _, _, _, _ = epoch('train', trainloader, net, optimizer, criterion, args, aug=False)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    data_test = dsttest[:, :-1]
    labels_test = np.array(dsttest[:, -1], dtype='int')
    data_test = torch.tensor(data_test).to(args.device)
    labels_test = torch.tensor(labels_test, dtype=torch.long, device=args.device)
    dst_test = TensorDataset(data_test, labels_test)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_train, shuffle=False, num_workers=0)
    time_train = time.time() - start
    loss_test, acc_test, bal_acc_test, mean_pred_prob_test, mean_in_acc_test, mean_out_acc_test, f1, pr_auc, roc_auc = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test, bal_acc_train, bal_acc_test, loss_test, mean_pred_prob_train, mean_pred_prob_test, mean_in_acc_test, mean_out_acc_test, f1, pr_auc, roc_auc

