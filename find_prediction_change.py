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

import model_head
import dataloaders
import core
import attack
from cal_metrics_run import load_model, load_attacked_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def find_prediction_change(log_dir_path):
    lines = [log_dir_path]
    model_weights_path = os.path.join(log_dir_path, 'models', 'model_round001.pth')
    args_json_path = os.path.join(log_dir_path, 'args.json')

    with open(args_json_path, 'r') as f:
        args = json.load(f)
    model, tokenizer = load_model(model_weights_path, args)

    attacked_dataset_list = []
    for attack_step in range(21):
        test_dataset = load_attacked_dataset(args, attack_step, model_round_number=1)
        test_dataset = [data for data in test_dataset]
        attacked_dataset_list.append(test_dataset)

    for data_num, data in enumerate(attacked_dataset_list[0]):
        original_text = data['text']
        original_label = data['label']
        outputs_classification_original, outputs_regression_original = attack.run_model_inference(original_text, tokenizer, model)
        outputs_classification_prob_original = torch.softmax(outputs_classification_original, dim=-1)[0]
        # print(outputs_classification_prob_original)

        if outputs_classification_prob_original[original_label] < 0.5:
            continue

        step_1_text = attacked_dataset_list[1][data_num]['text']
        outputs_classification, outputs_regression = attack.run_model_inference(step_1_text, tokenizer, model)
        outputs_classification_prob = torch.softmax(outputs_classification, dim=-1)[0]

        if outputs_classification_prob[original_label] < 0.5:
            print(float(outputs_classification_prob_original[original_label]), float(outputs_classification_prob[original_label]))
            print(original_text.lower())
            print(step_1_text.lower())
            lines.append(data['file_path'])
            lines.append('AD' if original_label == 1 else 'HC')
            lines.append(str(float(outputs_classification_prob_original[original_label])))
            lines.append(str(float(outputs_classification_prob[original_label])))
            lines.append(original_text.lower())
            lines.append(step_1_text.lower())

    lines.append(' ')
    return lines


def run_batch():
    all_lines = []

    log_dir_path = 'log_20/bert_base_sequence_level_1-94'
    lines = find_prediction_change(log_dir_path)
    all_lines.extend(lines)

    log_dir_path = 'log_20/bert_base_sequence_level_2-15_94'
    lines = find_prediction_change(log_dir_path)
    all_lines.extend(lines)

    log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94'
    lines = find_prediction_change(log_dir_path)
    all_lines.extend(lines)

    log_dir_path = 'log_21/bert_base_sequence_level_1-123'
    lines = find_prediction_change(log_dir_path)
    all_lines.extend(lines)

    log_dir_path = 'log_21/bert_base_sequence_level_2-83_123'
    lines = find_prediction_change(log_dir_path)
    all_lines.extend(lines)

    log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123'
    lines = find_prediction_change(log_dir_path)
    all_lines.extend(lines)
    
    save_path = 'vis/change_text.txt'
    with open(save_path, 'w') as f:
        f.writelines(s + '\n' for s in all_lines)


if __name__ == '__main__':
    # find_prediction_change()
    run_batch()

