import sys
sys.path.insert(0, '/home/Hotspots/code/hotspot_transfer')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from TSX.utils import load_data, load_county_data, get_initial_county_data, train_model, load_county_name, get_importance_value, \
    plot_temporal_importance, get_top_importance_value, get_normalize_importance_value, get_hotspot_weight, \
    train_model_multitask, load_ckp, mean_absolute_percentage_error, load_confirmed_data, get_weight
from TSX.models import IMVTensorLSTM, IMVTensorLSTMMultiTask

import os
import argparse
import tqdm
import torch
import numpy as np
import pandas as pd
import pickle
import math
import timeit
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.style.use('seaborn')

from matplotlib import rc
rc('font', weight='bold')


def train_one_county(fips, state, train_loader, valid_loader, ft_size):
    hidden_size = args.hidden_size
    n_epochs = args.n_epochs
    if args.explainer == 'IMVTensorLSTM':
        model = IMVTensorLSTM(ft_size, 1, hidden_size, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
        epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=1)
        if args.train:
            train_model(model, args.explainer, train_loader, valid_loader, optimizer=optimizer,
                        epoch_scheduler=epoch_scheduler, n_epochs=n_epochs,
                        device=device, county_fips_code=fips, state=state, cv=0)
        else:
            model_path = '../model_save/' + args.explainer + "/" + state + '/' + str(fips) + ".pt"
            model.load_state_dict(torch.load(model_path))
    else:
        model = []
    return model


def evaluate_performance(model, state, fips, county_name, scaler, X_train, y_train, X_test, y_test, X_total, y_total):
    scaler_for_cases = MinMaxScaler()
    scaler_for_cases.min_, scaler_for_cases.scale_ = scaler.min_[0], scaler.scale_[0]
    model.eval()
    date_sequence = []
    no_transfer_predict = 0
    if args.explainer == "IMVTensorLSTMMultiTask" or args.explainer == 'TransferLearning':
        X_total, task_idx, activated_share_columns, date_sequence = X_total
        # activated_share_columns = activated_share_columns[0, :]
        task_idx = task_idx.type(torch.LongTensor)
        with torch.no_grad():
            total_predict, total_alphas, total_betas, theta, neg_llk = model(X_total.to(device), y_total.to(device), task_idx, activated_share_columns)
            train_predict, train_alphas, train_betas, theta, neg_llk = model(X_train.to(device), y_train.to(device), task_idx, activated_share_columns)
            test_predict, test_alphas, test_betas, theta, neg_llk = model(X_test.to(device), y_test.to(device), task_idx, activated_share_columns)
    else:
        with torch.no_grad():
            total_predict, total_alphas, total_betas = model(X_total.to(device))
            train_predict, train_alphas, train_betas = model(X_train.to(device))
            test_predict, test_alphas, test_betas = model(X_test.to(device))
    total_predict_back = scaler_for_cases.inverse_transform(total_predict.detach().cpu().numpy())
    total_y_back = scaler_for_cases.inverse_transform(y_total.data.numpy())

    train_predict_back = scaler_for_cases.inverse_transform(train_predict.detach().cpu().numpy())
    train_y_back = scaler_for_cases.inverse_transform(y_train.data.numpy())

    if args.test_data_size != 0:
        test_predict_back = scaler_for_cases.inverse_transform(test_predict.detach().cpu().numpy())
        test_y_back = scaler_for_cases.inverse_transform(y_test.data.numpy())
        mse = mean_squared_error(test_y_back, test_predict_back)
        mae = mean_absolute_error(test_y_back, test_predict_back)
        mape = mean_absolute_percentage_error(test_y_back, test_predict_back)
        rmse = round(np.sqrt(mse), 3)
        mae = round(mae, 3)
        y_predict = np.concatenate([train_predict_back[:-1, 0], train_predict_back[-1, :], test_predict_back.squeeze(0)])
        y_true = np.concatenate([train_y_back[:-1, 0], train_y_back[-1, :], test_y_back.squeeze(0)])
        if args.explainer == "TransferLearning":
            pickle_path = '../model_save/' + args.explainer + "/" + "NY" + '/pickle'
            if not os.path.exists(pickle_path):
                os.mkdir(pickle_path)
            if state == "NY":
                with open(pickle_path + '/' + str(fips) + '.pkl', 'wb') as f:
                    pickle.dump(y_predict, f)
            elif state == "NY_flu_covid":
                with open(pickle_path + '/' + str(fips) + '.pkl', 'rb') as f:
                    no_transfer_predict = pickle.load(f)
        #y_predict = np.concatenate([train_predict_back[:, 0], test_predict_back.squeeze(0)])
        #y_true = np.concatenate([train_y_back[:, 0], test_y_back.squeeze(0)])
    else:
        mae = 0
        rmse = 0
        mape = 0
        test_predict_back = scaler_for_cases.inverse_transform(test_predict.detach().cpu().numpy())
        test_y_back = scaler_for_cases.inverse_transform(y_test.data.numpy())
        y_predict = np.concatenate([train_predict_back[:-1, 0], train_predict_back[-1, :]])
        y_true = np.concatenate([train_y_back[:-1, 0], train_y_back[-1, :]])
    alphas, betas = get_importance_value(train_alphas, train_betas)

    #print('County {} MAE: {}'.format(fips, mae))
    #print('County {} RMSE: {}'.format(fips, rmse))

    if args.save:
        plt.axvline(x=date_sequence[-args.decoding_steps-1], c='r', linestyle='--')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
        plt.xticks(rotation=0)
        title = '{} ({})'.format(county_name, fips)
        plot_path = '../plots/' + args.explainer + "/" + state + '/' + str(fips) + "/" + "time_series.pdf"
        if state == "NY_flu":
            plt.plot(date_sequence, y_true, color='blue', label='Active Flu Cases')
            plt.plot(date_sequence, y_predict, color='orange', label='Predicted Active Flu Cases')
        elif state == "NY_flu_covid":
            plt.plot(date_sequence, y_true, color='blue', label='Active Cases')
            plt.plot(date_sequence, y_predict, color='orange', label='Predicted Active Cases with Transfer')
            plt.plot(date_sequence, no_transfer_predict, color='red', label='Predicted Active Cases without Transfer')
            plot_path = '../plots/' + args.explainer + "/" + state + '/' + str(fips) + "/" + str(args.lambda_trans) + "_time_series.pdf"
        else:
            plt.plot(date_sequence, y_true, color='blue', label='Active Cases')
            plt.plot(date_sequence, y_predict, color='orange', label='Predicted Active Cases')
        plt.title(title)
        plt.xlabel('Prediction Date')
        plt.ylabel('Population Count')
        plt.legend()
        plt.savefig(plot_path, dpi=300, orientation='landscape')
        #plt.show()
        plt.close()
    return mae, rmse, mape, alphas, betas


def state_level_computation(state):
    # Load Data
    path = '../data/state_mobility_link/' + state + '_mobility_link.csv'
    df = load_data(path)
    feature_county_list = list(df.columns[4:])
    county_list = list(set(df['next_area_fip'].tolist()))[9:10]
    county_list = [str(ct) for ct in county_list]
    county_name_list = load_county_name(county_list)
    county_dict = dict(zip(county_list, county_name_list))
    mae_list = []
    rmse_list = []
    mape_list = []
    valid_county_list = []
    county_importance_dict = dict(zip(['county_covid_cases'] + feature_county_list, np.zeros(len(feature_county_list) + 1)))
    error_county_list = []
    state_adjacency_matrix = np.zeros((len(county_list), len(county_list)))
    num = 0
    for county_fips, county_name in tqdm.tqdm(county_dict.items(), file=sys.stdout):
        county_plot_path = '../plots/' + args.explainer + "/" + state + '/' + str(county_fips)
        if not os.path.exists(county_plot_path):
            os.mkdir(county_plot_path)
        print("Computing for county {}, fips {}".format(county_name, county_fips))
        try:
            data_train_loader, data_test_loader, X_train, y_train, X_test, y_test, X_total, \
                 y_total, scaler, feature_name, feature_fips = \
                load_county_data(df, county_fips, args.seq_length, args.batch_size, args.test_data_size, args.fillna)
        except ValueError as err:
            print(err.args)
            error_county_list += [county_fips]
            continue
        feature_size = X_total.shape[2]
        model = train_one_county(county_fips, state, data_train_loader, data_test_loader, feature_size)
        mae, rmse, mape, alphas, betas = evaluate_performance(model, state, county_fips, county_name, scaler, X_train, y_train,
                                                       X_test, y_test, X_total, y_total)
        normalize_betas = get_normalize_importance_value(betas)
        hotspot_weight = get_hotspot_weight(y_train)
        for i, ct in enumerate(feature_fips):
            county_importance_dict[ct] += normalize_betas[i] * hotspot_weight

        # adjacency matrix
        for i, ct in enumerate(feature_fips):
            if ct in county_list:
                state_adjacency_matrix[num, county_list.index(ct)] = normalize_betas[i] * hotspot_weight
        num = num + 1
        alphas, betas, feature_name = get_top_importance_value(alphas, betas, feature_name)
        plot_temporal_importance(alphas, feature_name, args.explainer, state, county_fips, county_name)

        mae_list += [mae]
        rmse_list += [rmse]
        mape_list += [mape]
        valid_county_list += [county_fips]
    performance_df = pd.DataFrame({'county': valid_county_list, 'MAE': mae_list, 'RMSE': rmse_list, 'MAPE': mape_list})
    importance_df = pd.DataFrame(county_importance_dict.items(), columns=['fips', 'Importance_score']).\
        sort_values(by='Importance_score', ascending=False)
    importance_df['county_name'] = load_county_name(importance_df['fips'])
    print(performance_df.head())
    print(importance_df.head())
    output_path = "../outputs/" + args.explainer + "/" + state + "/"
    performance_path = output_path + "Total_county_performance.csv"
    importance_path = output_path + "Total_county_importance.csv"
    performance_df.to_csv(performance_path)
    importance_df.to_csv(importance_path)

    adjacency_df = pd.DataFrame(state_adjacency_matrix)
    adjacency_df.columns = county_name_list
    adjacency_df.index = county_list
    # adjacency_df.to_csv(output_path + "adjacency_matrix.csv")

    with open(output_path + "error_county.txt", "wb") as fp:  # Pickling
        pickle.dump(error_county_list, fp)


def state_level_computation_multitask(state, transfer):
    # Load Data
    if not transfer:
        if "weekly" in state:
            if "test" in state:
                path = '../data/Weekly_mobility_link_test_data/' + state.split("_")[0] + '_mobility_link_weekly_test.csv'
            else:
                path = '../data/Weekly_mobility_link/' + state.split("_")[0] + '_mobility_link_weekly.csv'
            df = load_data(path)
        else:
            path = '../data/state_mobility_link/' + state + '_mobility_link.csv'
            df = load_data(path)
    else:
        path = '../data/NY_flu_covid/'
        if state == "NY_flu":
            df = load_data(path + 'NY_flu_mobility_link_weekly.csv')
        else:
            df = load_data(path + 'NY_covid_mobility_link_weekly.csv')
    input_task_feature = args.input_task_feature
    if "test" in state:
        input_task_feature_name = ['county_positivity_rate', 'next_area_cmi']
    else:
        input_task_feature_name = ['county_covid_cases', 'next_area_cmi']
    feature_county_list = list(df.columns[3 + input_task_feature:])
    county_list = list(set(df['next_area_fip'].tolist()))
    county_list = [str(ct) for ct in county_list]
    county_list = county_list[0:args.num_nodes]
    county_name_list = load_county_name(county_list)
    county_dict = dict(zip(county_list, county_name_list))
    error_county_list = []
    county_importance_dict = dict(
        zip(input_task_feature_name + feature_county_list, np.zeros(len(feature_county_list) + input_task_feature)))

    mae_list = []
    rmse_list = []
    mape_list = []
    valid_county_list = []
    data_train_loader_list = []
    data_test_loader_list = []
    print(county_list)
    print(county_name_list)
    task_num = 0
    for county_fips, county_name in county_dict.items():
        try:
            county_plot_path = '../plots/' + args.explainer + "/" + state + '/' + str(county_fips)
            if not os.path.exists(county_plot_path):
                os.mkdir(county_plot_path)
            _ = get_initial_county_data(df, county_fips, args.fillna, input_task_feature, args.seq_length, task_num)
            task_num += 1
        except ValueError as err:
            print(err.args)
            error_county_list += [county_fips]
            continue
    task_idx = 0
    train_size = 0
    task_scaler_dict = {}
    for county_fips, county_name in county_dict.items():
        try:
            data_train_loader, data_test_loader, X_train, y_train, X_test, y_test, X_total, \
                y_total, scaler, feature_name, feature_fips = \
                load_county_data(df, county_fips, args.seq_length, args.batch_size, args.test_data_size,
                                args.fillna, input_task_feature, input_task_feature_name, args.decoding_steps, args.validation_data_size, task_idx)
            task_scaler_dict[task_idx] = scaler
            task_idx += 1
            data_train_loader_list += [data_train_loader]
            data_test_loader_list += [data_test_loader]
            if X_train.shape[0] > train_size:
                train_size = X_train.shape[0]
        except ValueError as err:
            print(err)
            continue

    input_share_dim = len(feature_county_list)
    hidden_size = args.hidden_size
    n_epochs = args.n_epochs
    iterations = math.ceil(train_size / args.batch_size)
    start_time = timeit.default_timer()

    flu_model = IMVTensorLSTMMultiTask(input_share_dim, input_task_feature, task_num, 1, hidden_size, device, args.em, args.drop_prob, args.decoding_steps).to(device)
    if transfer:
        if state == "NY_flu_covid":
            print("loading flu model-------")
            flu_optimizer = torch.optim.Adam(flu_model.parameters(), lr=0.001, amsgrad=True)
            model_path = '../model_save/' + args.explainer + '/NY_flu/NY_flu_best.pt'
            flu_model, _, _, _ = load_ckp(model_path, flu_model, flu_optimizer)

    model = IMVTensorLSTMMultiTask(input_share_dim, input_task_feature, task_num, 1, hidden_size, device, args.em,
                                    args.drop_prob, args.decoding_steps).to(device)
    if torch.cuda.device_count() > 1 and device == 'cuda':
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    epoch_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=1)
    start_epoch = 0
    valid_loss_min = 9999
    if not args.train:
        model_path = '../model_save/' + args.explainer + '/' + state + '/' + state + ".pt" # should be _best
        model, optimizer, start_epoch, valid_loss_min = load_ckp(model_path, model, optimizer)
        print("model = ", model)
        print("optimizer = ", optimizer)
        print("start_epoch = ", start_epoch)
        print("valid_loss_min = {:.6f}".format(valid_loss_min))
        n_epochs = n_epochs - start_epoch
    if not args.evaluate:
        train_model_multitask(model, args.explainer, data_train_loader_list, data_test_loader_list, flu_model,
                              input_task_feature, start_epoch, valid_loss_min, optimizer=optimizer, epoch_scheduler=epoch_scheduler, n_epochs=n_epochs,
                              device=device, state=state, iterations=iterations, lambda_reg=args.lambda_reg, lambda_trans=args.lambda_trans,
                              task_scaler_dict=task_scaler_dict, decoding_steps=args.decoding_steps, save=args.save, cv=0)

    stop_time = timeit.default_timer()
    task_idx = 0
    for county_fips, county_name in county_dict.items():
        try:
            data_train_loader, data_test_loader, X_train, y_train, X_test, y_test, X_total, \
                y_total, scaler, feature_name, feature_fips = \
                load_county_data(df, county_fips, args.seq_length, args.batch_size, args.test_data_size,
                                 args.fillna, input_task_feature, input_task_feature_name, args.decoding_steps, args.validation_data_size, task_idx)
        except ValueError as err:
            continue
        mae, rmse, mape, alphas, betas = evaluate_performance(model, state, county_fips, county_name, scaler, X_train,
                                                        y_train, X_test, y_test, X_total, y_total)
        confirmed = load_confirmed_data()
        for i, ct in enumerate(feature_fips):
            if i < input_task_feature:
                county_importance_dict[ct] += betas[input_task_feature*task_idx + i]
            else:
                weight = get_weight(ct, county_fips, confirmed)
                if args.weight and weight != -1:
                    county_importance_dict[ct] += betas[task_num * input_task_feature + i - input_task_feature] * weight
                else:
                    county_importance_dict[ct] += betas[task_num * input_task_feature + i - input_task_feature]
        task_idx += 1
        mae_list += [mae]
        rmse_list += [rmse]
        mape_list += [mape]
        valid_county_list += [county_fips]
    performance_df = pd.DataFrame({'county': valid_county_list, 'MAE': mae_list, 'RMSE': rmse_list, 'MAPE': mape_list})
    importance_df = pd.DataFrame(county_importance_dict.items(), columns=['fips', 'Importance_score']). \
        sort_values(by='Importance_score', ascending=False)
    importance_df['county_name'] = load_county_name(importance_df['fips'])
    print(performance_df.head())
    print(importance_df.head(10))
    output_path = "../outputs/" + args.explainer + "/" + state + "/"
    if state == "NY_flu_covid":
        performance_path = output_path + str(args.lambda_trans) + "_Total_county_performance.csv"
        importance_path = output_path + str(args.lambda_trans) + "_Total_county_importance.csv"
    else:
        performance_path = output_path + "Total_county_performance.csv"
        importance_path = output_path + "Total_county_importance.csv"
    if args.save:
        print("aa")
        performance_df.to_csv(performance_path)
        importance_df.to_csv(importance_path)
    print('Training time: ', stop_time - start_time)


def main(state):
    # load full data list
    state_list = ["AK", "NY", "WA", "NV", "AZ", "AL", "FL", "GA", "MS", "TN", "MI", "AR", "LA", "MO", "OK", "TX",
                  "NM", "CA", "UT", "ND", "HI", "MN", "OR", "MT", "CO", "KS", "WY", "NE", "SD", "CT", "MA", "ME",
                  "VT", "RI", "MD", "VA", "DE", "PA", "OH", "NJ", "SC", "NC", "IA", "WI", "IL", "ID", "KY", "IN",
                  "WV", "NH", "DC"]
    state_list = [state]
    for i, state in enumerate(state_list):
        if not os.path.exists('../plots/' + args.explainer + '/' + state):
            os.mkdir('../plots/' + args.explainer + '/' + state)
        if not os.path.exists('../model_save/' + args.explainer + '/' + state):
            os.mkdir('../model_save/' + args.explainer + '/' + state)
        if not os.path.exists('../outputs/' + args.explainer + '/' + state):
            os.mkdir('../outputs/' + args.explainer + '/' + state)
        print("Start for state {}, num {}".format(state, i))
        if args.explainer == "IMVTensorLSTMMultiTask":
            state_level_computation_multitask(state, 0)
        elif args.explainer == "IMVTensorLSTM":
            state_level_computation(state)
        elif args.explainer == "TransferLearning":
            state_level_computation_multitask(state, 1)


if __name__ == '__main__':
    np.random.seed(2021)
    parser = argparse.ArgumentParser(description='Run baseline model for covid')
    parser.add_argument('--explainer', type=str, default='IMVTensorLSTMMultiTask', help='Explainer model')
    parser.add_argument('--fillna', type=str, default='zero', help='fill na')
    parser.add_argument('--input_task_feature', type=int, default=2, help='input_task_feature')
    parser.add_argument('--seq_length', type=int, default=4, help='seq_length')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--hidden_size', type=int, default=128, help='hidden_size')
    parser.add_argument('--n_epochs', type=int, default=300, help='n_epochs')
    parser.add_argument('--test_data_size', type=int, default=1, help='test_data_size')
    parser.add_argument('--validation_data_size', type=int, default=0, help='validation_data_size')
    parser.add_argument('--drop_prob', type=float, default=0.1, help='drop prob')
    parser.add_argument('--lambda_reg', type=float, default=0.001, help='lambda regulation')
    parser.add_argument('--lambda_trans', type=float, default=0.5, help='lambda_trans')
    parser.add_argument('--decoding_steps', type=int, default=1, help='decoding_steps')
    parser.add_argument('--num_nodes', type=int, default=1, help='num_nodes')
    parser.add_argument('--train', action='store_false')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--weight', action='store_false')
    parser.add_argument('--em', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    if not os.path.exists('../plots'):
        os.mkdir('../plots')
    if not os.path.exists('../model_save'):
        os.mkdir('../model_save')
    if not os.path.exists('../outputs'):
        os.mkdir('../outputs')
    if not os.path.exists('../model_save/' + args.explainer):
        os.mkdir('../model_save/' + args.explainer)
    if not os.path.exists('../outputs/' + args.explainer):
        os.mkdir('../outputs/' + args.explainer)
    if not os.path.exists('../plots/' + args.explainer):
        os.mkdir('../plots/' + args.explainer)

    for step in [1]:
        for state in ["IL_weekly"]:
            node_list = [2, 5, 10, 15, 20, 25, 30, 35]
            # node_list = [2, 5]
            time_list = []
            for num_node in node_list:
                print("state: {}".format(state))
                print("num node: {}".format(num_node))
                args.decoding_steps = step
                args.num_nodes = num_node
                start_time = time.time()
                main(state)
                length = (time.time() - start_time) / 60
                time_list += [length]
            len_list = len(node_list)
            runtime_path = '../outputs/' + args.explainer + "/" + state + "/"
            runtime_raw = zip([state] * len_list, node_list, time_list)
            run_time_df = pd.DataFrame(runtime_raw, columns=['state', 'num_components', 'Time'])
            run_time_df.to_csv(runtime_path + 'component_runtime_df_' + '.csv')
    #for i in [0.1, 0.01, 0.001]:
    #    for j in [0.0, 0.1, 0.2]:
    #        args.lambda_reg = i
    #        args.drop_prob = j
    #        print("regulation: {}, drop prob: {}".format(i, j))
    #        main()
    #for lambda_trans in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5]:
    #    args.lambda_trans = lambda_trans
    #    print("now computing for lambda trans = {}".format(lambda_trans))
    #    main('NY_flu_covid')

    #for state in ['NY_flu', 'NY_flu_covid']:
    #    print("now run for {}".format(state))
    #    print("----------------------------")
    #    if state == 'NY_flu':
    #        args.test_data_size = 0
    #        args.n_epochs = 250
    #        main(state)
    #    elif state == 'NY_flu_covid':
    #        args.test_data_size = 1
    #        args.n_epochs = 400
    #        for lambda_trans in [0.2, 0.5, 1.0]:
    #            print("lambda transfer = {}".format(lambda_trans))
    #            args.lambda_trans = lambda_trans
    #            main(state)
    #    else:
    #       args.n_epochs = 400
    #        args.test_data_size = 1
    #        main(state)

