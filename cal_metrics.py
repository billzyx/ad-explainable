import numpy as np
import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import xlwt



# log_dir_path = 'log_20/bert_base_sequence_level_1-94'
# log_dir_path = 'log_20/bert_base_sequence_level_2-15_94'
# log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94'
# log_dir_path = 'log_21/bert_base_sequence_level_1-123'
# log_dir_path = 'log_21/bert_base_sequence_level_2-83_123'
log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123'

# log_dir_path = 'log_20/bert_base_sequence_level_1-94-attack_reverse-1'
# log_dir_path = 'log_20/bert_base_sequence_level_2-15_94-attack_reverse-1'
# log_dir_path = 'log_21/bert_base_sequence_level_1-123-attack_reverse-1'
# log_dir_path = 'log_21/bert_base_sequence_level_2-83_123-attack_reverse-5'

training_log_dir = 'training_logs'


def run(log_dir_path):
    print(log_dir_path)

    precision_1_list = []
    recall_1_list = []
    f1_1_list = []

    precision_2_list = []
    recall_2_list = []
    f1_2_list = []

    accuracy_list = []
    rmse_list = []

    max_accuracy = 0.
    min_rmse = 99999.
    for training_log_filename in sorted(os.listdir(os.path.join(log_dir_path, training_log_dir))):
        if training_log_filename.endswith('.json'):
            log_json_path = os.path.join(log_dir_path, training_log_dir, training_log_filename)
            with open(log_json_path) as f:
                round_data = json.load(f)
            min_loss = 999999.
            choose_epoch_report_dict = None
            for epoch_report_dict in round_data:
                # loss = epoch_report_dict['train']['loss_classification'] + epoch_report_dict['train']['loss_regression']
                loss = epoch_report_dict['train']['loss_sum']
                if loss < min_loss:
                    min_loss = loss
                    choose_epoch_report_dict = epoch_report_dict
                max_accuracy = max(max_accuracy, epoch_report_dict['test']['accuracy'])
                min_rmse = min(min_rmse, epoch_report_dict['test']['rmse'])

            precision_1_list.append(choose_epoch_report_dict['test']['0']['precision'])
            recall_1_list.append(choose_epoch_report_dict['test']['0']['recall'])
            f1_1_list.append(choose_epoch_report_dict['test']['0']['f1-score'])

            precision_2_list.append(choose_epoch_report_dict['test']['1']['precision'])
            recall_2_list.append(choose_epoch_report_dict['test']['1']['recall'])
            f1_2_list.append(choose_epoch_report_dict['test']['1']['f1-score'])

            accuracy_list.append(choose_epoch_report_dict['test']['accuracy'])
            rmse_list.append(choose_epoch_report_dict['test']['rmse'])

    print(accuracy_list)
    print('accuracy_list:', np.argmax(accuracy_list) + 1)
    print(rmse_list)
    print('rmse_list:', np.argmin(rmse_list) + 1)
    # assert len(accuracy_list) == 5

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

    print('best accuracy: {:.2f}'.format(max_accuracy * 100))

    print('rmse: {:.2f} \pm {:.2f}'.format(rmse_mean, rmse_std))
    print('best rmse: {:.2f}'.format(min_rmse))

    return accuracy_mean * 100, accuracy_std * 100


def run_batch_20():
    log_dir_path = 'log_20/bert_base_sequence_level_1-94-attack-'
    accuracy_mean_list_20_1, accuracy_std_list_20_1 = get_data(log_dir_path)

    log_dir_path = 'log_20/bert_base_sequence_level_2-15_94-attack-'
    accuracy_mean_list_20_2, accuracy_std_list_20_2 = get_data(log_dir_path)

    log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94-attack-'
    accuracy_mean_list_20_3, accuracy_std_list_20_3 = get_data(log_dir_path)

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
    plt.savefig('vis/adversarial_training_20.pdf')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx in range(1, 4):
        ws.write(idx, 0, "level " + str(idx))

    for idx in range(21):
        ws.write(0, idx + 1, str(idx))
        ws.write(1, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_1[idx], accuracy_std_list_20_1[idx]))
        ws.write(2, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_2[idx], accuracy_std_list_20_2[idx]))
        ws.write(3, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_3[idx], accuracy_std_list_20_3[idx]))
    wb.save('vis/adversarial_training_20.xls')


def run_batch_20_reversed():
    log_dir_path = 'log_20/bert_base_sequence_level_1-94-attack_reverse-'
    accuracy_mean_list_20_1, accuracy_std_list_20_1 = get_data(log_dir_path)

    log_dir_path = 'log_20/bert_base_sequence_level_2-15_94-attack_reverse-'
    accuracy_mean_list_20_2, accuracy_std_list_20_2 = get_data(log_dir_path)

    log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94-attack_reverse-'
    accuracy_mean_list_20_3, accuracy_std_list_20_3 = get_data(log_dir_path)

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
    plt.savefig('vis/adversarial_training_reverse_20.pdf')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx in range(1, 4):
        ws.write(idx, 0, "level " + str(idx))

    for idx in range(21):
        ws.write(0, idx + 1, str(idx))
        ws.write(1, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_1[idx], accuracy_std_list_20_1[idx]))
        ws.write(2, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_2[idx], accuracy_std_list_20_2[idx]))
        ws.write(3, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_20_3[idx], accuracy_std_list_20_3[idx]))
    wb.save('vis/adversarial_training_reverse_20.xls')


def run_batch_21():
    log_dir_path = 'log_21/bert_base_sequence_level_1-123-attack-'
    accuracy_mean_list_21_1, accuracy_std_list_21_1 = get_data(log_dir_path)

    log_dir_path = 'log_21/bert_base_sequence_level_2-83_123-attack-'
    accuracy_mean_list_21_2, accuracy_std_list_21_2 = get_data(log_dir_path)

    log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123-attack-'
    accuracy_mean_list_21_3, accuracy_std_list_21_3 = get_data(log_dir_path)

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
    plt.savefig('vis/adversarial_training_21.pdf')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx in range(1, 4):
        ws.write(idx, 0, "level " + str(idx))

    for idx in range(21):
        ws.write(0, idx + 1, str(idx))
        ws.write(1, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_1[idx], accuracy_std_list_21_1[idx]))
        ws.write(2, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_2[idx], accuracy_std_list_21_2[idx]))
        ws.write(3, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_3[idx], accuracy_std_list_21_3[idx]))
    wb.save('vis/adversarial_training_21.xls')


def run_batch_21_reversed():
    log_dir_path = 'log_21/bert_base_sequence_level_1-123-attack_reverse-'
    accuracy_mean_list_21_1, accuracy_std_list_21_1 = get_data(log_dir_path)

    log_dir_path = 'log_21/bert_base_sequence_level_2-83_123-attack_reverse-'
    accuracy_mean_list_21_2, accuracy_std_list_21_2 = get_data(log_dir_path)

    log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123-attack_reverse-'
    accuracy_mean_list_21_3, accuracy_std_list_21_3 = get_data(log_dir_path)

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
    plt.savefig('vis/adversarial_training_reverse_21.pdf')

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx in range(1, 4):
        ws.write(idx, 0, "level " + str(idx))

    for idx in range(21):
        ws.write(0, idx + 1, str(idx))
        ws.write(1, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_1[idx], accuracy_std_list_21_1[idx]))
        ws.write(2, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_2[idx], accuracy_std_list_21_2[idx]))
        ws.write(3, idx + 1, '{:.2f} \pm {:.2f}'.format(accuracy_mean_list_21_3[idx], accuracy_std_list_21_3[idx]))
    wb.save('vis/adversarial_training_reverse_21.xls')


def get_data(log_dir_path):
    accuracy_mean_list = []
    accuracy_std_list = []
    accuracy_mean, accuracy_std = run(log_dir_path.split('-attack')[0])
    accuracy_mean_list.append(accuracy_mean)
    accuracy_std_list.append(accuracy_std)
    for i in range(1, 21):
        try:
            accuracy_mean, accuracy_std = run(log_dir_path + str(i))
            print('Step {}:'.format(i))
            print('accuracy: {:.2f} \pm {:.2f}'.format(accuracy_mean, accuracy_std))
            accuracy_mean_list.append(accuracy_mean)
            accuracy_std_list.append(accuracy_std)
        except:
            print(i)
            accuracy_mean_list.append(0.)
            accuracy_std_list.append(0.)
    return np.array(accuracy_mean_list), np.array(accuracy_std_list)


if __name__ == '__main__':
    # run(log_dir_path)
    # run_batch()
    run_batch_21()
    run_batch_21_reversed()