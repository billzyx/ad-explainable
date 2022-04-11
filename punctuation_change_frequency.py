import os
from collections import Counter
import re
import xlwt
import dataloaders
import json


def preprocess_string(str1):
    str1 = str1.lower()
    str1 = ' '.join(str1.split())
    str1 = ' ' + str1 + ' '
    str1 = re.sub(r'\s([,.\'"](?:\s|$))', r'\1', str1)
    return str1


def find_punctuation_change(str1, str2):
    punctuation_list = [',', '.', ';']
    change_type = None

    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)
    original_idx = 0
    is_found_diff = False
    while original_idx < len(str1) and original_idx < len(str2):
        if not is_found_diff and original_idx < len(str2) and original_idx < len(str1) \
                and str1[original_idx] != str2[original_idx]:
            str1_ch = str1[original_idx]
            str2_ch = str2[original_idx]
            if str1_ch in punctuation_list and str2_ch in punctuation_list:
                change_type = 'Replace ' + str1_ch + '->' + str2_ch
            elif str1_ch in punctuation_list:
                change_type = 'Delete ' + str1_ch
            elif str2_ch in punctuation_list:
                change_type = 'Add ' + str2_ch
            is_found_diff = True
        original_idx += 1
    return change_type


def load_texts(log_dir_path):
    args_json_path = os.path.join(log_dir_path, 'args.json')

    with open(args_json_path, 'r') as f:
        args = json.load(f)

    attack_text_file_list_ad = []
    attack_text_file_list_hc = []
    for model_round_number in range(1, 11):
        attacked_dataset_list = []
        for attack_step in range(21):
            test_dataset = dataloaders.load_test_attack_dataset(
                args['test_dataset'], args['model_description'], args['log_dir'], attack_step, model_round_number)
            test_dataset = [data for data in test_dataset]
            attacked_dataset_list.append(test_dataset)
        for data_num, data in enumerate(attacked_dataset_list[0]):
            original_text = data['text']
            original_label = data['label']
            attack_text_file = [original_text]
            for attack_step in range(1, 21):
                attack_text_file.append(attacked_dataset_list[attack_step][data_num]['text'])
            if original_label == 1:
                attack_text_file_list_ad.append(attack_text_file)
            elif original_label == 0:
                attack_text_file_list_hc.append(attack_text_file)

    return attack_text_file_list_ad, attack_text_file_list_hc


def count_punctuation_change(log_dir_path, steps=5):
    attack_text_file_list_ad, attack_text_file_list_hc = load_texts(log_dir_path)

    change_type_dict_ad = count_punctuation_change_dataset(attack_text_file_list_ad, steps)
    change_type_dict_hc = count_punctuation_change_dataset(attack_text_file_list_hc, steps)

    return change_type_dict_ad, change_type_dict_hc


def count_punctuation_change_dataset(attack_text_file_list, steps=5):
    change_type_list = []
    for attack_text_file in attack_text_file_list:
        for attack_text_idx in range(steps):
            str1 = attack_text_file[attack_text_idx].strip()
            str2 = attack_text_file[attack_text_idx + 1].strip()
            if str2 == '' or str1 == str2:
                break
            change_type = find_punctuation_change(str1, str2)
            change_type_list.append(change_type)
    change_type_dict = Counter(change_type_list)
    return change_type_dict


def get_counters(log_dir_path):
    step_list = [1, 5, 20]

    change_type_dict_list = []
    description_list = []
    for step in step_list:
        change_type_dict_ad, change_type_dict_hc = count_punctuation_change(log_dir_path, steps=step)
        change_type_dict_list.append((change_type_dict_ad, change_type_dict_hc))
        description = dict()
        description['step'] = step
        description_list.append(description)
    return description_list, change_type_dict_list


def write_xls(log_dir_path, save_path='vis.xls'):
    description_list, change_type_dict_list = get_counters(log_dir_path)

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx, description in enumerate(description_list):
        ws.write(0, idx * 3, 'step=' + str(description['step']))
        ws.write(1, idx * 3 + 1, 'AD->HC')
        ws.write(1, idx * 3 + 2, 'HC->AD')

    row_idx = 0
    col_idx = 0

    for change_type_dict_ad, change_type_dict_hc in change_type_dict_list:
        change_type_keys = sorted(list(set(list(change_type_dict_ad.keys()) + list(change_type_dict_hc.keys()))))
        for idx, key in enumerate(change_type_keys):
            row_idx = idx + 2
            ws.write(row_idx, col_idx, key)
            ws.write(row_idx, col_idx + 1, change_type_dict_ad[key])
            ws.write(row_idx, col_idx + 2, change_type_dict_hc[key])

        col_idx += 3
    wb.save(save_path)


def write_batch():
    log_dir_path = 'log_20/bert_base_sequence_level_1-94'
    write_xls(log_dir_path, save_path='vis_change/20_1_test.xls')

    log_dir_path = 'log_20/bert_base_sequence_level_2-15_94'
    write_xls(log_dir_path, save_path='vis_change/20_2_test.xls')

    log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94'
    write_xls(log_dir_path, save_path='vis_change/20_3_test.xls')

    log_dir_path = 'log_21/bert_base_sequence_level_1-123'
    write_xls(log_dir_path, save_path='vis_change/21_1_test.xls')

    log_dir_path = 'log_21/bert_base_sequence_level_2-83_123'
    write_xls(log_dir_path, save_path='vis_change/21_2_test.xls')

    log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123'
    write_xls(log_dir_path, save_path='vis_change/21_3_test.xls')


def main():
    log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94'
    write_xls(log_dir_path, save_path='vis.xls')


if __name__ == '__main__':
    # main()
    # write_xls()
    write_batch()
    # write_batch_train()


