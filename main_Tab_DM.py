import os
import time
import copy
import argparse
import numpy as np
import pandas as pd
import torch
from utils_Tab_DM import get_tabular_dataset, get_network, evaluate_synset, get_time


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='Credit_Default', help='dataset')
    parser.add_argument('--model', type=str, default='MLP', help='model')
    parser.add_argument('--eval_mode', type=str, default='SS', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=1, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic samples from random noise or randomly sampled real samples.')
    # parser.add_argument("--data_sel_exp", type=str, default="All", choices=["All", "Distillation", "Random_Sample", "Full", "Herding", "Forgeting"], help="Which datasets to include in the experiment")
    parser.add_argument("--full_ds_run", action="store_true", default=True, help="To inlcude experiment run on full dataset also")
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    args = parser.parse_args()
    args.method = 'DM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration+1, 2000).tolist() if args.eval_mode == 'S' or args.eval_mode == 'SS' else [args.Iteration] # The list of iterations when we evaluate models and record results.
    print('eval_it_pool: ', eval_it_pool)

    # Credit Default Best Hyperparameters
    if args.dataset == 'Credit_Default':
        model_arch_list = [[64, 32]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [32]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train=100
        args.Iteration = 20000
        total_no_exp = 1
        run_no = 1

    # Covertype and Credit Fraud
    if args.dataset == 'Credit_Fraud':
        model_arch_list = [[32, 16]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [512]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train = 100
        args.Iteration = 10000
        total_no_exp = 1
        run_no = 1

    if args.dataset == 'Covertype':
        model_arch_list = [[64, 32]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [512]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train = 100
        args.Iteration = 20000
        args.device = 'cpu'
        total_no_exp = 1
        run_no = 1

    if args.dataset == 'Census_Income':
        model_arch_list = [[256, 128, 64]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [512]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train = 100
        args.Iteration = 20000
        total_no_exp = 1
        run_no = 1

    if args.dataset == 'Adult_Data':
        model_arch_list = [[64, 32]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [32]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train = 100
        args.Iteration = 20000
        total_no_exp = 1
        run_no = 1

    if args.dataset == 'Bank_Marketing':
        model_arch_list = [[32, 16]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [32]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train = 100
        args.Iteration = 20000
        total_no_exp = 1
        run_no = 1

    if args.dataset == 'KDD_Cup':
        model_arch_list = [[64, 32]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [512]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train = 100
        args.Iteration = 10000
        total_no_exp = 1
        run_no = 1

    if args.dataset == 'IEEE_Fraud':
        model_arch_list = [[2048, 1024, 512, 256, 128]]
        act_func_list = ["Sigmoid"]
        momentum_list = [0.9]
        optimizer_list = ['SGD']
        batch_list = [512]
        spc_list = [10, 50, 100]
        lr_list = [1]
        args.epoch_eval_train = 100
        args.Iteration = 100
        args.device = 'cpu'
        total_no_exp = 1
        run_no = 1

    res_df_list = []
    for exp_no in range(total_no_exp):
        df_res = pd.DataFrame(columns=['Selection', 'SPC', 'Model', 'Architecture', 'Act_Fuct', 'Optimizer_Name', 'Momentum', 'batch', 'lr_img', 'Bal_Acc_Train', 'Bal_Acc_Test', 'Test_Loss', 'Inlier_Acc_Test', 'Outlier_Acc_Test', 'F1_Test', 'PR_AUC_Test', 'ROC_AUC_Test', 'Total_Time_in_Sec'])
        for optimizer_name in optimizer_list:
            for model_architecture in model_arch_list:
                for act_func in act_func_list:
                    for batch in batch_list:
                        for momentum in momentum_list:
                            for lr_val in lr_list:
                                for ipc in spc_list:

                                    print('optimizer_name, model_architecture, act_func, batch, momentum, lr_val, ipc: ', optimizer_name, str(model_architecture), act_func, batch, momentum, lr_val, ipc)
                                    # Assign selected hyper-parameters to args
                                    args.momemntum_img = momentum
                                    args.ipc = ipc
                                    args.lr_img = lr_val
                                    args.optimizer_name = optimizer_name
                                    args.batch_real = batch
                                    args.batch_train = batch

                                    # Read the original (Full) dataset for distillation, preprocess it, do a train test split and also get other information like number of classes and class labels
                                    num_classes, class_names, dst_train, dst_test, _ = get_tabular_dataset(args.dataset, run_no)

                                    args.num_classes = num_classes
                                    print('num_classes:{} class_names:{} dst_train:{} dst_test:{}'.format(num_classes, class_names, dst_train.shape, dst_test.shape))
                                    model_eval_pool = [args.model]

                                    accs_all_exps = dict() # record performances of all experiments
                                    for key in model_eval_pool:
                                        accs_all_exps[key] = []

                                    for exp in range(args.num_exp):
                                        print('\n================== Exp %d ==================\n '%exp)
                                        print('Hyper-parameters: \n', args.__dict__)
                                        print('Evaluation model pool: ', model_eval_pool)

                                        ### organize the real dataset ###
                                        data_all = []
                                        indices_class = [[] for c in range(num_classes)]

                                        data_all = dst_train[:,:-1]  # Get all data i.e, all columns except last one
                                        labels_all = np.array(dst_train[:,-1], dtype='int') # Get corresponding labels i.e, the last column
                                        print('data_all: ', data_all.shape)
                                        print('labels_all: ', labels_all.shape)
                                        for i, lab in enumerate(labels_all):
                                            indices_class[lab].append(i)
                                        data_all = torch.tensor(data_all).to(args.device)

                                        for c in range(num_classes):
                                            print('class c = %d: %d real samples'%(c, len(indices_class[c])))

                                        def get_class_specific_random_samples(c, n): # get random n samples from class c
                                            idx_shuffle = np.random.permutation(indices_class[c])[:n]
                                            return data_all[idx_shuffle]

                                        ### initialize the synthetic data ###
                                        data_syn = torch.randn(size=(num_classes * args.ipc, data_all.shape[1]), dtype=torch.float, requires_grad=True, device=args.device)
                                        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

                                        if args.init == 'real':
                                            print('initialize synthetic data from random real samples')
                                            for c in range(num_classes):
                                                data_syn.data[c*args.ipc:(c+1)*args.ipc] = get_class_specific_random_samples(c, args.ipc).detach().data
                                        else:
                                            print('initialize synthetic data from random noise')
                                        print('Initialized data_syn: ', data_syn.shape)


                                        ### training ###
                                        if args.optimizer_name == 'SGD': optimizer_img = torch.optim.SGD([data_syn, ], lr=args.lr_img, momentum=args.momemntum_img)  # optimizer_img for synthetic data
                                        else: optimizer_img = torch.optim.Adam([data_syn, ], lr=args.lr_img)
                                        optimizer_img.zero_grad()
                                        print('%s training begins'%get_time())

                                        loss_per_iter = []
                                        store_iter_no, all_bal_acc_train, all_bal_acc_test, all_loss_test, all_in_acc_test, all_out_acc_test, all_f1_test, all_pr_auc_test, all_roc_auc_test, all_compute_time = [], [], [], [], [], [], [], [], [], []
                                        max_acc_plug, max_acc = 0, 0
                                        for it in range(args.Iteration+1):

                                            ### Evaluate synthetic data ###
                                            if it == 0 or it == 100 or it == 500 or it % 1000 == 0 or it == args.Iteration:
                                                for model_eval in model_eval_pool:
                                                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                                                    accs = []
                                                    for it_eval in range(args.num_eval):
                                                        net_eval = get_network(model_eval, [data_syn.shape[1]]+model_architecture+[num_classes], act_func).to(args.device) # get the selected model
                                                        data_syn_eval, label_syn_eval = copy.deepcopy(data_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                                                        dm_start_time = time.time()
                                                        _, acc_train, acc_test, bal_acc_train, bal_acc_test, loss_test, _, _, in_acc_test, out_acc_test, f1, pr_auc, roc_auc = evaluate_synset(it_eval, net_eval, data_syn_eval, label_syn_eval, dst_test, args)
                                                        dm_end_time = time.time()
                                                        # Check if at the current iteration we get best accuracy
                                                        if bal_acc_test > max_acc:
                                                            max_acc = bal_acc_test
                                                            max_acc_plug = 1
                                                        else: max_acc_plug = 0
                                                        accs.append(acc_test)
                                                        store_iter_no.append(it)
                                                        all_bal_acc_train.append(bal_acc_train)
                                                        all_bal_acc_test.append(bal_acc_test)
                                                        all_loss_test.append(loss_test)
                                                        all_in_acc_test.append(in_acc_test)
                                                        all_out_acc_test.append(out_acc_test)
                                                        all_f1_test.append(f1)
                                                        all_pr_auc_test.append(pr_auc)
                                                        all_roc_auc_test.append(roc_auc)
                                                        all_compute_time.append(abs(dm_end_time-dm_start_time))
                                                    print('Evaluate %d random %s, test acc mean = %.4f test acc std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                                                    if it == args.Iteration: # record the final results
                                                        accs_all_exps[model_eval] += accs

                                            ### Train synthetic data ###
                                            net = get_network(args.model, [data_syn.shape[1]]+model_architecture+[num_classes], act_func).to(args.device)  # get the selected model
                                            net.train()
                                            for param in list(net.parameters()):
                                                param.requires_grad = False
                                            # Initialize the embedding model that will be later used for converting original samples to embedding space
                                            embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed # for GPU parallel

                                            loss_avg = 0

                                            ### Covert the real and synthetic samples to embedding space and then compute the distance (loss). Based on the loss value update synthetic data ###
                                            loss = torch.tensor(0.0).to(args.device)
                                            for c in range(num_classes):
                                                data_real = get_class_specific_random_samples(c, args.batch_real)
                                                data_syn1 = data_syn[c * args.ipc:(c + 1) * args.ipc] # .reshape((args.ipc, channel, im_size[0], im_size[1]))
                                                # Put the samples into embedding space
                                                output_real = embed(data_real.float()).detach()
                                                output_syn = embed(data_syn1.float())
                                                # Calculate loss or distance
                                                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

                                            optimizer_img.zero_grad()
                                            loss.backward()
                                            optimizer_img.step()
                                            loss_avg += loss.item()

                                            loss_avg /= (num_classes)
                                            loss_per_iter.append(loss_avg)

                                            if it%1000 == 0:
                                                print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))
                                            # Store the synthetic data
                                            if (it == 100 or it == 500 or it%1000 == 0 or it == args.Iteration) and max_acc_plug == 1:
                                                data_syn2 = pd.DataFrame(data_syn.cpu().detach().numpy(), columns= ['Col_'+str(i) for i in range(data_syn.cpu().detach().numpy().shape[1])])
                                                data_syn2['label'] = label_syn.cpu().detach().numpy()
                                                data_syn2.to_csv(os.path.join(args.save_path, str(args.dataset) + '_Synthetic_Distilled_Data_SPC_' + str(ipc)+'.csv'), index=False)

                                        print('##### DM Done #####')
                                        # print('iter_no = ', store_iter_no)
                                        # print('all_bal_acc_train = ', list(np.around(all_bal_acc_train, 4)))
                                        # print('all_bal_acc_test = ', list(np.around(all_bal_acc_test, 4)))
                                        # print('all_loss_test = ', list(np.around(all_loss_test, 4)))

                                    print('\n==================== Final Results ====================\n')
                                    for key in model_eval_pool:
                                        accs = accs_all_exps[key]
                                        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model, len(accs), key, np.mean(accs)*100, np.std(accs)*100))

                                    # Store the accuracy, f1 score, pr-auc, roc-auc into pandas dataframe
                                    df_res.loc[len(df_res)] = ['Distilled', int(args.ipc), args.model, str(model_architecture), act_func, args.optimizer_name, args.momemntum_img, args.batch_real, args.lr_img, np.max(all_bal_acc_train), np.max(all_bal_acc_test), np.min(all_loss_test), np.max(all_in_acc_test), np.max(all_out_acc_test), np.max(all_f1_test), np.max(all_pr_auc_test), np.max(all_roc_auc_test), np.mean(all_compute_time)]

                                    #########  Code section to compute the results for Random Selection  #####################
                                    num_classes, class_names, dst_train, dst_test, _ = get_tabular_dataset(args.dataset, run_no, 'Random', False, args.ipc)
                                    args.num_classes = num_classes
                                    print('num_classes:{} class_names:{} dst_train:{} dst_test:{}'.format(num_classes, class_names, dst_train.shape, dst_test.shape))
                                    model_eval_pool = [args.model]

                                    data_train_all = dst_train[:, :-1]
                                    labels_train_all = np.array(dst_train[:, -1], dtype='int')
                                    data_train_all = torch.tensor(data_train_all).to(args.device)
                                    labels_train_all = torch.tensor(labels_train_all, dtype=torch.long, device=args.device)
                                    for model_eval in model_eval_pool:
                                        net_eval = get_network(model_eval, [data_train_all.shape[1]] + model_architecture + [num_classes], act_func).to(args.device)  # get a random model
                                        random_start_time = time.time()
                                        _, acc_train, acc_test, bal_acc_train, bal_acc_test, loss_test, _, _, in_acc_test, out_acc_test, f1, pr_auc, roc_auc = evaluate_synset(100, net_eval, data_train_all, labels_train_all, dst_test, args)
                                        random_end_time = time.time()
                                        print('##### DM #####')
                                        print('Random bal_acc_train, bal_acc_test, loss_test: ', bal_acc_train, bal_acc_test, loss_test)
                                        df_res.loc[len(df_res)] = ['Random', int(args.ipc), args.model, str(model_architecture), act_func, args.optimizer_name, args.momemntum_img, args.batch_real, args.lr_img, bal_acc_train, bal_acc_test, loss_test, in_acc_test, out_acc_test, f1, pr_auc, roc_auc, abs(random_end_time-random_start_time)]

                                    dst_train1 = pd.DataFrame(dst_train, columns=['Col_' + str(i) for i in range(dst_train.shape[1] - 1)] + ['label'])
                                    dst_train1.to_csv(os.path.join(args.save_path, str(args.dataset) + '_Random_Set_SPC_' + str(ipc) + '.csv'), index=False)

                                    '''
                                    ##### Code section to compute results of Coreset Selection: Herding   ##################
                                    # num_classes, class_names, _, dst_test, _ = get_tabular_dataset(args.dataset, run_no)
                                    # args.num_classes = num_classes
                                    dst_train = pd.read_csv('/data/' + str(args.dataset) +'/run'+str(run_no) + '/' + str(args.dataset) + '_Herding_Coreset_SPC_' + str(args.ipc) + '.csv')
                                    dst_train = np.array(dst_train)
                                    print('num_classes:{} class_names:{} dst_train:{} dst_test:{}'.format(num_classes, class_names, dst_train.shape, dst_test.shape))
                                    model_eval_pool = [args.model]

                                    data_train_all = dst_train[:, :-1]
                                    labels_train_all = np.array(dst_train[:, -1], dtype='int')
                                    data_train_all = torch.tensor(data_train_all).to(args.device)
                                    labels_train_all = torch.tensor(labels_train_all, dtype=torch.long, device=args.device)
                                    for model_eval in model_eval_pool:
                                        net_eval = get_network(model_eval, [data_train_all.shape[1]] + model_architecture + [num_classes], act_func).to(args.device)  # get a random model
                                        herding_start_time = time.time()
                                        _, acc_train, acc_test, bal_acc_train, bal_acc_test, loss_test, mean_pred_prob_train, mean_pred_prob_test, in_acc_test, out_acc_test, f1, pr_auc, roc_auc = evaluate_synset(100, net_eval, data_train_all, labels_train_all, dst_test, args)
                                        herding_end_time = time.time()
                                        print('#### Herding ####')
                                        print('bal_acc_train, bal_acc_test, loss_test: ', bal_acc_train, bal_acc_test, loss_test)
                                        # print('mean_pred_prob_train, mean_pred_prob_test: ', mean_pred_prob_train, mean_pred_prob_test)
                                        df_res.loc[len(df_res)] = ['Herding', int(args.ipc), args.model, str(model_architecture), act_func, args.optimizer_name, args.momemntum_img, args.batch_real, args.lr_img, bal_acc_train, bal_acc_test, loss_test, in_acc_test, out_acc_test, f1, pr_auc, roc_auc, abs(herding_end_time-herding_start_time)]

                                    ##### Code Section to compute results for Coreset Selection: Forgetting   #######
                                    dst_train = pd.read_csv('/data/' + str(args.dataset) +'/run'+str(run_no) + '/' + str(args.dataset) + '_Forgetting_Coreset_SPC_' + str(args.ipc) + '.csv')
                                    dst_train = np.array(dst_train)
                                    print('num_classes:{} class_names:{} dst_train:{} dst_test:{}'.format(num_classes, class_names, dst_train.shape, dst_test.shape))
                                    model_eval_pool = [args.model]

                                    data_train_all = dst_train[:, :-1]
                                    labels_train_all = np.array(dst_train[:, -1], dtype='int')
                                    data_train_all = torch.tensor(data_train_all).to(args.device)
                                    labels_train_all = torch.tensor(labels_train_all, dtype=torch.long, device=args.device)
                                    for model_eval in model_eval_pool:
                                        net_eval = get_network(model_eval, [data_train_all.shape[1]] + model_architecture + [num_classes], act_func).to(args.device)  # get a random model
                                        forgetng_start_time = time.time()
                                        _, acc_train, acc_test, bal_acc_train, bal_acc_test, loss_test, mean_pred_prob_train, mean_pred_prob_test, in_acc_test, out_acc_test, f1, pr_auc, roc_auc = evaluate_synset(100, net_eval, data_train_all, labels_train_all, dst_test, args)
                                        forgeting_end_time = time.time()
                                        print('#### Forgetting ####')
                                        print('bal_acc_train, bal_acc_test, loss_test: ', bal_acc_train, bal_acc_test, loss_test)
                                        # print('mean_pred_prob_train, mean_pred_prob_test: ', mean_pred_prob_train, mean_pred_prob_test)
                                        df_res.loc[len(df_res)] = ['Forgetting', int(args.ipc), args.model, str(model_architecture), act_func, args.optimizer_name, args.momemntum_img, args.batch_real, args.lr_img, bal_acc_train, bal_acc_test, loss_test, in_acc_test, out_acc_test, f1, pr_auc, roc_auc, abs(forgeting_end_time-forgetng_start_time)]
                                    '''

                                    ####### Code Section to compute the results for Original (Full) Dataset  ##########
                                    if args.full_ds_run == True:
                                        num_classes, class_names, dst_train, dst_test, _ = get_tabular_dataset(args.dataset, run_no)
                                        args.num_classes = num_classes
                                        print('num_classes:{} class_names:{} dst_train:{} dst_test:{}'.format(num_classes, class_names, dst_train.shape, dst_test.shape))
                                        model_eval_pool = [args.model]

                                        data_train_all = dst_train[:, :-1]
                                        labels_train_all = np.array(dst_train[:, -1], dtype='int')
                                        data_train_all = torch.tensor(data_train_all).to(args.device)
                                        labels_train_all = torch.tensor(labels_train_all, dtype=torch.long, device=args.device)
                                        for model_eval in model_eval_pool:
                                            net_eval = get_network(model_eval, [data_train_all.shape[1]] + model_architecture + [num_classes], act_func).to(args.device)  # get a random model
                                            orig_start_time = time.time()
                                            _, acc_train, acc_test, bal_acc_train, bal_acc_test, loss_test, mean_pred_prob_train, mean_pred_prob_test, in_acc_test, out_acc_test, f1, pr_auc, roc_auc = evaluate_synset(100, net_eval, data_train_all, labels_train_all, dst_test, args)
                                            orig_end_time = time.time()
                                            print('#### Full ####')
                                            print('bal_acc_train, bal_acc_test, loss_test: ', bal_acc_train, bal_acc_test, loss_test)
                                            # print('mean_pred_prob_train, mean_pred_prob_test: ', mean_pred_prob_train, mean_pred_prob_test)
                                            df_res.loc[len(df_res)] = ['Orig', len(data_train_all), args.model, str(model_architecture), act_func, args.optimizer_name, args.momemntum_img, args.batch_real, args.lr_img, bal_acc_train, bal_acc_test, loss_test, in_acc_test, out_acc_test, f1, pr_auc, roc_auc, abs(orig_end_time-orig_start_time)]
                                        args.full_ds_run = False

                                if exp_no==0: df_res.to_csv(os.path.join(args.save_path, 'Result_DM_and_baseline_Exp0_' + str(args.dataset) + '_model_' + str(args.model) + '.csv'), index=False)
        res_df_list.append(df_res)
    cat_cols = ['Selection', 'SPC', 'Model', 'Architecture', 'Act_Fuct', 'Optimizer_Name', 'batch', 'lr_img']
    df_concat = pd.concat(res_df_list, keys=range(len(res_df_list[0])))
    df_concat_mean = df_concat.groupby(level=1).mean()
    df_concat_std = df_concat.groupby(level=1).std()
    for cat_col in cat_cols:
        df_concat_mean[cat_col] = res_df_list[0][cat_col]
        df_concat_std[cat_col] = res_df_list[0][cat_col]
    df_concat_mean = df_concat_mean[list(res_df_list[0].columns)]
    df_concat_std = df_concat_std[list(res_df_list[0].columns)]
    df_concat_mean.to_csv(os.path.join(args.save_path, 'Result_Mean_DM_and_baseline_' + str(args.dataset) + '_model_' + str(args.model) + '.csv'), index=False)
    df_concat_std.to_csv(os.path.join(args.save_path, 'Result_Std_DM_and_baseline_' + str(args.dataset) + '_model_' + str(args.model) + '.csv'), index=False)

# python main_Tab_DM.py  --dataset Credit_Default  --model MLP  --init real  --num_exp 1  --num_eval 1

if __name__ == '__main__':
    main()


