import os

use_gpu_num = '0'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = use_gpu_num

import numpy as np
from torch.utils.data.dataset import Dataset, ConcatDataset
import json
import torch
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import xlwt

import model_head
import dataloaders
import core
import attack

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# log_dir_path = 'log_20/bert_base_sequence_level_1-94'
# log_dir_path = 'log_20/bert_base_sequence_level_2-15_94'
log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94'
# log_dir_path = 'log_21/bert_base_sequence_level_1-123'
# log_dir_path = 'log_21/bert_base_sequence_level_2-83_123'
# log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123'

model_weights_dir = 'models'


def run(log_dir_path, attack_step=0):
    print(log_dir_path)

    args_json_path = os.path.join(log_dir_path, 'args.json')

    with open(args_json_path, 'r') as f:
        args = json.load(f)

    precision_1_list = []
    recall_1_list = []
    f1_1_list = []

    precision_2_list = []
    recall_2_list = []
    f1_2_list = []

    accuracy_list = []
    rmse_list = []

    all_round_pred_prob_list_classification = []
    all_round_pred_list_regression = []

    # test_dataset = load_original_test_dataset(dataset_name, args)
    # test_dataset = apply_attack_to_dataset(test_dataset)

    for model_round_number, model_weights_filename in \
            enumerate(sorted(os.listdir(os.path.join(log_dir_path, model_weights_dir)))):
        if model_weights_filename.endswith('.pth'):
            model_weights_path = os.path.join(log_dir_path, model_weights_dir, model_weights_filename)
            model, tokenizer = load_model(model_weights_path, args)

            pred_list_classification = []
            pred_prob_list_classification = []
            ground_truth_list_classification = []
            pred_list_regression = []
            ground_truth_list_regression = []

            test_dataset = load_attacked_dataset(args, attack_step, model_round_number + 1)

            for data in test_dataset:
                inputs = [data['text']]
                # inputs = [data['attack_text']]
                # inputs = [attack.attack_function(data['text'], data['label'], tokenizer, model)]
                inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
                if args['model_name'].startswith('pre_train/'):
                    length_label, audio_embedding = core.generate_shift_label(
                        data['text'], data['original_text'], data['audio_embedding'], tokenizer, use_length=False)
                    inputs['shifter_embedding'] = torch.tensor([audio_embedding])
                for k in inputs:
                    inputs[k] = inputs[k].to(device)
                labels = torch.LongTensor([data['label']]).to(device)
                mmse_labels = torch.FloatTensor([data['label_mmse']]).to(device)

                outputs = model(inputs)
                outputs_classification, outputs_regression = outputs[0], outputs[1]
                _, preds = torch.max(outputs_classification, 1)

                outputs_classification_prob = torch.softmax(outputs_classification, dim=-1)
                pred_prob_list_classification.append(outputs_classification_prob[0].tolist())

                pred_list_classification.extend(preds.tolist())
                ground_truth_list_classification.extend(labels.data.tolist())
                pred_list_regression.extend(outputs_regression.tolist())
                ground_truth_list_regression.extend(mmse_labels.data.tolist())

            round_report_dict = classification_report(y_true=ground_truth_list_classification,
                                                      y_pred=pred_list_classification, output_dict=True)
            round_rmse = mean_squared_error(y_true=ground_truth_list_regression,
                                            y_pred=pred_list_regression, squared=False)
            round_report_dict['rmse'] = round_rmse

            precision_1_list.append(round_report_dict['0']['precision'])
            recall_1_list.append(round_report_dict['0']['recall'])
            f1_1_list.append(round_report_dict['0']['f1-score'])

            precision_2_list.append(round_report_dict['1']['precision'])
            recall_2_list.append(round_report_dict['1']['recall'])
            f1_2_list.append(round_report_dict['1']['f1-score'])

            accuracy_list.append(round_report_dict['accuracy'])
            rmse_list.append(round_report_dict['rmse'])

            all_round_pred_prob_list_classification.append(pred_prob_list_classification)
            all_round_pred_list_regression.append(pred_list_regression)
    print(accuracy_list)
    print(rmse_list)
    precision_1_mean = np.mean(precision_1_list)
    recall_1_mean = np.mean(recall_1_list)
    f1_1_mean = np.mean(f1_1_list)
    precision_2_mean = np.mean(precision_2_list)
    recall_2_mean = np.mean(recall_2_list)
    f1_2_mean = np.mean(f1_2_list)
    accuracy_mean = np.mean(accuracy_list)
    rmse_mean = np.mean(rmse_list)
    precision_1_std = np.std(precision_1_list)
    recall_1_std = np.std(recall_1_list)
    f1_1_std = np.std(f1_1_list)
    precision_2_std = np.std(precision_2_list)
    recall_2_std = np.std(recall_2_list)
    f1_2_std = np.std(f1_2_list)
    accuracy_std = np.std(accuracy_list)
    rmse_std = np.std(rmse_list)
    print('precision_1: {:.2f} \pm {:.2f}'.format(precision_1_mean * 100, precision_1_std * 100))
    print('recall_1: {:.2f} \pm {:.2f}'.format(recall_1_mean * 100, recall_1_std * 100))
    print('f1_1: {:.2f} \pm {:.2f}'.format(f1_1_mean * 100, f1_1_std * 100))
    print('precision_2: {:.2f} \pm {:.2f}'.format(precision_2_mean * 100, precision_2_std * 100))
    print('recall_2: {:.2f} \pm {:.2f}'.format(recall_2_mean * 100, recall_2_std * 100))
    print('f1_2: {:.2f} \pm {:.2f}'.format(f1_2_mean * 100, f1_2_std * 100))
    print('accuracy: {:.2f} \pm {:.2f}'.format(accuracy_mean * 100, accuracy_std * 100))
    print('rmse: {:.2f} \pm {:.2f}'.format(rmse_mean, rmse_std))
    all_round_pred_prob_list_classification = np.array(all_round_pred_prob_list_classification)
    all_round_pred_list_regression = np.array(all_round_pred_list_regression)
    all_round_pred_prob_list_classification = np.mean(all_round_pred_prob_list_classification, axis=0)
    all_round_pred_list_classification = np.argmax(all_round_pred_prob_list_classification, axis=-1)
    all_report_dict = classification_report(y_true=ground_truth_list_classification,
                                            y_pred=all_round_pred_list_classification, output_dict=True)
    all_round_pred_list_regression = np.mean(all_round_pred_list_regression, axis=0)
    all_rmse = mean_squared_error(y_true=ground_truth_list_regression,
                                  y_pred=all_round_pred_list_regression, squared=False)
    all_report_dict['rmse'] = all_rmse
    print()
    print('Ensemble in 10 rounds:')
    print('precision_1: {:.2f}'.format(all_report_dict['0']['precision'] * 100))
    print('recall_1: {:.2f}'.format(all_report_dict['0']['recall'] * 100))
    print('f1_1: {:.2f}'.format(all_report_dict['0']['f1-score'] * 100))
    print('precision_2: {:.2f}'.format(all_report_dict['1']['precision'] * 100))
    print('recall_2: {:.2f}'.format(all_report_dict['1']['recall'] * 100))
    print('f1_2: {:.2f}'.format(all_report_dict['1']['f1-score'] * 100))
    print('accuracy: {:.2f}'.format(all_report_dict['accuracy'] * 100))
    print('rmse: {:.2f}'.format(all_report_dict['rmse']))

    return accuracy_mean * 100, accuracy_std * 100, all_report_dict['accuracy'] * 100


def apply_attack_to_dataset(test_dataset_data_loader, args):
    test_dataset = []
    model_weights_path = os.path.join(log_dir_path, model_weights_dir, 'model_round001.pth')
    model, tokenizer = load_model(model_weights_path, args)
    for data in test_dataset_data_loader:
        data['attack_text'] = attack.attack_function(data['text'], data['label'], tokenizer, model, step=5,
                                                     reverse=False, punctuation_list=args['punctuation_list'])
        # print(data['attack_text'])
        test_dataset.append(data)
    return test_dataset


def get_data(log_dir_path):
    accuracy_mean, accuracy_std, accuracy_ensemble = run(log_dir_path, attack_step=0)
    accuracy_mean_list = [accuracy_mean]
    accuracy_std_list = [accuracy_std]
    for i in range(1, 21):
        accuracy_mean, accuracy_std, _ = run(log_dir_path, attack_step=i)
        print('Step {}:'.format(i))
        print('accuracy: {:.2f} \pm {:.2f}'.format(accuracy_mean, accuracy_std))
        accuracy_mean_list.append(accuracy_mean)
        accuracy_std_list.append(accuracy_std)
    return np.array(accuracy_mean_list), np.array(accuracy_std_list), accuracy_ensemble


def run_batch_21():
    log_dir_path = 'log_21/bert_base_sequence_level_1-123'
    accuracy_mean_list_21_1, accuracy_std_list_21_1, accuracy_ensemble_21_1 = get_data(log_dir_path)

    log_dir_path = 'log_21/bert_base_sequence_level_2-83_123'
    accuracy_mean_list_21_2, accuracy_std_list_21_2, accuracy_ensemble_21_2 = get_data(log_dir_path)

    log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123'
    accuracy_mean_list_21_3, accuracy_std_list_21_3, accuracy_ensemble_21_3 = get_data(log_dir_path)

    step_list = [x for x in range(0, 21)]

    fig, ax = plt.subplots()
    ax.plot(step_list, accuracy_mean_list_21_1, label="level 1", linestyle="-", marker='o')
    ax.fill_between(step_list, accuracy_mean_list_21_1 - accuracy_std_list_21_1,
                    accuracy_mean_list_21_1 + accuracy_std_list_21_1, alpha=.1)
    ax.plot(step_list, accuracy_mean_list_21_2, label="level 2", linestyle="-", marker='v')
    ax.fill_between(step_list, accuracy_mean_list_21_2 - accuracy_std_list_21_2,
                    accuracy_mean_list_21_2 + accuracy_std_list_21_2, alpha=.1)
    ax.plot(step_list, accuracy_mean_list_21_3, label="level 3", linestyle="-", marker='s')
    ax.fill_between(step_list, accuracy_mean_list_21_3 - accuracy_std_list_21_3,
                    accuracy_mean_list_21_3 + accuracy_std_list_21_3, alpha=.1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Attack steps', fontsize=18)
    ax.set_ylabel('Accuracy %', fontsize=18)
    plt.legend()
    plt.savefig('vis/adversarial_attack_21.pdf')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx in range(1, 4):
        ws.write(idx, 0, "level " + str(idx))

    ws.write(1, 1, '{:.2f}'.format(accuracy_ensemble_21_1))
    ws.write(2, 1, '{:.2f}'.format(accuracy_ensemble_21_2))
    ws.write(3, 1, '{:.2f}'.format(accuracy_ensemble_21_3))

    for idx in range(21):
        ws.write(0, idx + 2, str(idx))
        ws.write(1, idx + 2, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_1[idx], accuracy_std_list_21_1[idx]))
        ws.write(2, idx + 2, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_2[idx], accuracy_std_list_21_2[idx]))
        ws.write(3, idx + 2, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_3[idx], accuracy_std_list_21_3[idx]))
    wb.save('vis/adversarial_attack_21.xls')


def run_batch_20():
    log_dir_path = 'log_20/bert_base_sequence_level_1-94'
    accuracy_mean_list_20_1, accuracy_std_list_20_1, accuracy_ensemble_20_1 = get_data(log_dir_path)

    log_dir_path = 'log_20/bert_base_sequence_level_2-15_94'
    accuracy_mean_list_20_2, accuracy_std_list_20_2, accuracy_ensemble_20_2 = get_data(log_dir_path)

    log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94'
    accuracy_mean_list_20_3, accuracy_std_list_20_3, accuracy_ensemble_20_3 = get_data(log_dir_path)

    step_list = [x for x in range(0, 21)]

    fig, ax = plt.subplots()
    ax.plot(step_list, accuracy_mean_list_20_1, label="ADReSS 20 level 1", linestyle="-", marker='o')
    ax.fill_between(step_list, accuracy_mean_list_20_1 - accuracy_std_list_20_1,
                    accuracy_mean_list_20_1 + accuracy_std_list_20_1, alpha=.1)
    ax.plot(step_list, accuracy_mean_list_20_2, label="ADReSS 20 level 2", linestyle="-", marker='o')
    ax.fill_between(step_list, accuracy_mean_list_20_2 - accuracy_std_list_20_2,
                    accuracy_mean_list_20_2 + accuracy_std_list_20_2, alpha=.1)
    ax.plot(step_list, accuracy_mean_list_20_3, label="ADReSS 20 level 3", linestyle="-", marker='o')
    ax.fill_between(step_list, accuracy_mean_list_20_3 - accuracy_std_list_20_3,
                    accuracy_mean_list_20_3 + accuracy_std_list_20_3, alpha=.1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Attack steps')
    ax.set_ylabel('Accuracy %')
    plt.legend()
    plt.savefig('vis/adversarial_attack_20.pdf')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx in range(1, 4):
        ws.write(idx, 0, "level " + str(idx))

    ws.write(1, 1, '{:.2f}'.format(accuracy_ensemble_20_1))
    ws.write(2, 1, '{:.2f}'.format(accuracy_ensemble_20_2))
    ws.write(3, 1, '{:.2f}'.format(accuracy_ensemble_20_3))

    for idx in range(21):
        ws.write(0, idx + 2, str(idx))
        ws.write(1, idx + 2, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_1[idx], accuracy_std_list_20_1[idx]))
        ws.write(2, idx + 2, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_2[idx], accuracy_std_list_20_2[idx]))
        ws.write(3, idx + 2, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_3[idx], accuracy_std_list_20_3[idx]))
    wb.save('vis/adversarial_attack_20.xls')


def load_original_test_dataset(dataset_name, args):
    # test_dataset_list = [dataloaders.ADReSSTextTrainDataset(
    #     train_path, args['level_list'], args['punctuation_list'],
    #     filter_min_word_length=filter_min_word_length),
    #     dataloaders.ADReSSTextTestDataset(
    #         test_path, test_label_path, args['level_list'], args['punctuation_list'],
    #         filter_min_word_length=filter_min_word_length)]
    # test_dataset = ConcatDataset(test_dataset_list)
    # test_dataset = dataloaders.ADReSSTextTrainDataset(
    #     train_path, args['level_list'], args['punctuation_list'],
    #     filter_min_word_length=filter_min_word_length)
    # test_dataset = dataloaders.ADReSSTextTestDataset(
    #     test_path, test_label_path, args['level_list'], args['punctuation_list'],
    #     filter_min_word_length=filter_min_word_length)
    test_dataset = dataloaders.load_test_dataset(
        dataset_name, args['level_list'], args['punctuation_list'], args['test_filter_min_word_length'])
    # test_dataset = dataloaders.load_test_dataset(
    #     'ADReSS20-test', args['level_list'], args['punctuation_list'], args['test_filter_min_word_length'])
    # test_dataset = dataloaders.load_test_dataset(
    #     'ADReSSo21-test', args['level_list'], args['punctuation_list'], args['train_filter_min_word_length'])
    return test_dataset


def load_attacked_dataset(args, attack_step, model_round_number):
    test_dataset = dataloaders.load_test_attack_dataset(
        args['test_dataset'], args['model_description'], args['log_dir'], attack_step, model_round_number)
    return test_dataset


def load_model(model_weights_path, args):
    model, tokenizer = model_head.load_model(args)
    model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)
    model.eval()
    return model, tokenizer


if __name__ == '__main__':
    # run(log_dir_path)
    run_batch_21()
    # run_batch_20()
