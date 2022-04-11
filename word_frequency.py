import os
import difflib
from collections import Counter
import re
import xlwt


def count_word_frequency(text_dir_path, n_gram=5, steps=5, both_side=True):
    # n_gram_word_dict = dict()
    context_words_list = []
    attack_text_file_list = load_texts(text_dir_path)

    for attack_text_file in attack_text_file_list:
        for attack_text_idx in range(steps):
            str1 = attack_text_file[attack_text_idx].strip()
            str2 = attack_text_file[attack_text_idx + 1].strip()
            if str2 == '' or str1 == str2:
                break
            new_str, change_idx = find_different_punctuation_index(str1, str2)
            if both_side:
                context_words = find_context_words_both_side(new_str, change_idx, gram=n_gram)
                context_words_list.append(context_words)
            else:
                context_words = find_context_words_one_side(new_str, change_idx, gram=n_gram)
                context_words_list.extend(context_words)
    n_gram_word_dict = Counter(context_words_list)

    return n_gram_word_dict


def load_texts(text_dir_path):
    attack_text_file_list = []
    for attack_text_file_name in sorted(os.listdir(text_dir_path)):
        attack_text_file_path = os.path.join(text_dir_path, attack_text_file_name)
        if attack_text_file_path.endswith('.txt') and os.path.isfile(attack_text_file_path):
            with open(attack_text_file_path, 'r') as f:
                attack_text_file = f.readlines()
                attack_text_file_list.append(attack_text_file)
    return attack_text_file_list


def preprocess_string(str1):
    str1 = str1.lower()
    str1 = ' '.join(str1.split())
    str1 = ' ' + str1 + ' '
    str1 = re.sub(r'\s([,.\'"](?:\s|$))', r'\1', str1)
    return str1


def find_different_punctuation_index_ndiff(str1, str2):
    # There is a bug of ndiff when encounter long sequence, our own implementation is preferred
    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)
    diff = difflib.ndiff(str1, str2)
    new_str = ''
    change_idx = -1

    for i, s in enumerate(diff):
        # print(s)
        if s[-1] not in [',', '.', ';']:
            new_str += s[-1]
        if not s[0] == ' ':
            change_idx = len(new_str) if len(new_str) != 1 else len(new_str) - 1

    return new_str, change_idx


def find_different_punctuation_index(str1, str2):
    str1 = preprocess_string(str1)
    str2 = preprocess_string(str2)
    original_idx = 0
    change_idx = -1
    new_str = ''
    if len(str2) > len(str1):
        str1, str2 = str2, str1
    is_found_diff = False
    while original_idx < len(str1):
        if not is_found_diff and original_idx < len(str2) and str1[original_idx] != str2[original_idx]:
            change_idx = len(new_str) if len(new_str) != 1 else len(new_str) - 1
            is_found_diff = True
        if str1[original_idx] not in [',', '.', ';']:
            new_str += str1[original_idx]
        original_idx += 1
    return new_str, change_idx


def search_word(new_str, change_idx, left=True):
    if left:
        idx = change_idx - 1
        while idx >= 0 and new_str[idx] != ' ':
            idx -= 1
        return new_str[idx + 1:change_idx], idx
    else:
        idx = change_idx + 1
        while idx < len(new_str) - 1 and new_str[idx] != ' ':
            idx += 1
        return new_str[change_idx + 1:idx] if idx < len(new_str) else '', idx


def find_context_words_one_side(new_str, change_idx, gram=1):
    context_words_list = []

    context_word_left = ''
    idx = change_idx
    for _ in range(gram):
        context_word, idx = search_word(new_str, idx, left=True)
        if context_word is not '':
            context_word_left = context_word + ' ' + context_word_left
        else:
            break
    context_words_list.append(context_word_left.strip())

    context_word_right = ''
    idx = change_idx
    for _ in range(gram):
        context_word, idx = search_word(new_str, idx, left=False)
        if context_word is not '':
            context_word_right = context_word + ' ' + context_word_right
        else:
            break
    context_words_list.append(context_word_right.strip())

    return context_words_list


def find_context_words_both_side(new_str, change_idx, gram=1):
    context_word_all = ''
    idx = change_idx
    for _ in range(gram):
        context_word, idx = search_word(new_str, idx, left=True)
        if context_word is not '':
            context_word_all = context_word + ' ' + context_word_all
        else:
            break

    context_word_all += '#'

    idx = change_idx
    for _ in range(gram):
        context_word, idx = search_word(new_str, idx, left=False)
        if context_word is not '':
            context_word_all = context_word_all + ' ' + context_word
        else:
            break

    return context_word_all


def get_counters(attack_text_dir_path):
    n_gram_list = [1, 2]
    step_list = [1, 5, 20]


    n_gram_word_dict_list = []
    description_list = []
    for n_gram in n_gram_list:
        for step in step_list:
            n_gram_word_dict = count_word_frequency(attack_text_dir_path, n_gram=n_gram, steps=step, both_side=True)
            n_gram_word_dict_list.append(n_gram_word_dict)
            description = dict()
            description['n_gram'] = n_gram
            description['step'] = step
            description_list.append(description)
    return description_list, n_gram_word_dict_list


def write_xls(attack_text_dir_path, save_path='vis.xls'):
    description_list, n_gram_word_dict_list = get_counters(attack_text_dir_path)

    wb = xlwt.Workbook()
    ws = wb.add_sheet('Summary')

    for idx, description in enumerate(description_list):
        ws.write(0, idx * 2, 'n_gram=' + str(description['n_gram']))
        ws.write(1, idx * 2, 'step=' + str(description['step']))

    row_idx = 0
    col_idx = 0

    for n_gram_word_dict in n_gram_word_dict_list:
        n_gram_word_dict = sorted(n_gram_word_dict.items(), key=lambda item: item[1], reverse=True)
        for idx, (key, value) in enumerate(n_gram_word_dict):
            row_idx = idx + 2
            ws.write(row_idx, col_idx, key)
            ws.write(row_idx, col_idx + 1, value)
            # if idx > 5:
            #     break

        col_idx += 2
    wb.save(save_path)


def write_batch():
    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_1-94/ADReSS-IS2020-data/test/asr_text'
    write_xls(attack_text_dir_path, save_path='vis/20_1_test.xls')

    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_2-15_94/ADReSS-IS2020-data/test/asr_text'
    write_xls(attack_text_dir_path, save_path='vis/20_2_test.xls')

    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_3-15_32_94/ADReSS-IS2020-data/test/asr_text'
    write_xls(attack_text_dir_path, save_path='vis/20_3_test.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_1-123/ADReSSo21/diagnosis/test-dist/asr_text'
    write_xls(attack_text_dir_path, save_path='vis/21_1_test.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_2-83_123/ADReSSo21/diagnosis/test-dist/asr_text'
    write_xls(attack_text_dir_path, save_path='vis/21_2_test.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_3-43_83_123/ADReSSo21/diagnosis/test-dist/asr_text'
    write_xls(attack_text_dir_path, save_path='vis/21_3_test.xls')


def write_batch_train():
    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_1-94/ADReSS-IS2020-data/train/asr_text/cc'
    write_xls(attack_text_dir_path, save_path='vis/20_1_train_hc.xls')

    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_2-15_94/ADReSS-IS2020-data/train/asr_text/cc'
    write_xls(attack_text_dir_path, save_path='vis/20_2_train_hc.xls')

    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_3-15_32_94/ADReSS-IS2020-data/train/asr_text/cc'
    write_xls(attack_text_dir_path, save_path='vis/20_3_train_hc.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_1-123/ADReSSo21/diagnosis/train/asr_text/cn'
    write_xls(attack_text_dir_path, save_path='vis/21_1_train_hc.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_2-83_123/ADReSSo21/diagnosis/train/asr_text/cn'
    write_xls(attack_text_dir_path, save_path='vis/21_2_train_hc.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_3-43_83_123/ADReSSo21/diagnosis/train/asr_text/cn'
    write_xls(attack_text_dir_path, save_path='vis/21_3_train_hc.xls')

    # #
    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_1-94/ADReSS-IS2020-data/train/asr_text/cd'
    write_xls(attack_text_dir_path, save_path='vis/20_1_train_ad.xls')

    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_2-15_94/ADReSS-IS2020-data/train/asr_text/cd'
    write_xls(attack_text_dir_path, save_path='vis/20_2_train_ad.xls')

    attack_text_dir_path = 'attack_text/log_20/bert_base_sequence_level_3-15_32_94/ADReSS-IS2020-data/train/asr_text/cd'
    write_xls(attack_text_dir_path, save_path='vis/20_3_train_ad.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_1-123/ADReSSo21/diagnosis/train/asr_text/ad'
    write_xls(attack_text_dir_path, save_path='vis/21_1_train_ad.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_2-83_123/ADReSSo21/diagnosis/train/asr_text/ad'
    write_xls(attack_text_dir_path, save_path='vis/21_2_train_ad.xls')

    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_3-43_83_123/ADReSSo21/diagnosis/train/asr_text/ad'
    write_xls(attack_text_dir_path, save_path='vis/21_3_train_ad.xls')


def main():
    attack_text_dir_path = 'attack_text/log_21/bert_base_sequence_level_2-83_123/ADReSSo21/diagnosis/test-dist/asr_text'
    n_gram_word_dict = count_word_frequency(attack_text_dir_path, n_gram=2, steps=20, both_side=True)
    print(n_gram_word_dict)


if __name__ == '__main__':
    # main()
    # write_xls()
    write_batch()
    # write_batch_train()

