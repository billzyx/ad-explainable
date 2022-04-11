import os

use_gpu_num = '2'

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# log_dir_path = 'log_20/bert_base_sequence_level_1-94'
# log_dir_path = 'log_20/bert_base_sequence_level_2-15_94'
log_dir_path = 'log_20/bert_base_sequence_level_3-15_32_94'
# log_dir_path = 'log_21/bert_base_sequence_level_1-123'
# log_dir_path = 'log_21/bert_base_sequence_level_2-83_123'
# log_dir_path = 'log_21/bert_base_sequence_level_3-43_83_123'

attack_text_output_dir = 'attack_text'
attack_text_output_dir = os.path.join(attack_text_output_dir, log_dir_path)
reverse_attack_text_output_dir = 'reverse_attack_text'
reverse_attack_text_output_dir = os.path.join(reverse_attack_text_output_dir, log_dir_path)

model_weights_dir = 'models'
args_json_path = os.path.join(log_dir_path, 'args.json')

with open(args_json_path, 'r') as f:
    args = json.load(f)


def load_model(model_weights_path):
    model, tokenizer = model_head.load_model(args)
    model.load_state_dict(torch.load(model_weights_path))
    model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_attack_text(steps=20, reverse=False):
    # train_dataset = dataloaders.load_train_dataset(
    #     'ADReSS20-train', args['level_list'], args['punctuation_list'], args['train_filter_min_word_length'])
    # train_dataset = dataloaders.load_test_dataset(
    #     'ADReSS20-test', args['level_list'], args['punctuation_list'], args['train_filter_min_word_length'])
    train_dataset = dataloaders.load_train_dataset(
        'ADReSSo21-train', args['level_list'], args['punctuation_list'], args['train_filter_min_word_length'])
    # train_dataset = dataloaders.load_test_dataset(
    #     'ADReSSo21-test', args['level_list'], args['punctuation_list'], args['train_filter_min_word_length'])

    for model_weights_filename in sorted(os.listdir(os.path.join(log_dir_path, model_weights_dir))):
        if model_weights_filename.endswith('.pth'):
            model_weights_path = os.path.join(log_dir_path, model_weights_dir, model_weights_filename)
            model, tokenizer = load_model(model_weights_path)
            for data in train_dataset:
                step_text_list = []
                attack_text = data['text']
                step_text_list.append(attack_text)
                for step in range(steps + 1):
                    attack_text = attack.attack_function(attack_text, data['label'], tokenizer, model, step=1,
                                                         reverse=reverse, punctuation_list=args['punctuation_list'])
                    step_text_list.append(attack_text)
                if reverse:
                    file_path = os.path.join(reverse_attack_text_output_dir, data['file_path'])
                else:
                    file_path = os.path.join(attack_text_output_dir, data['file_path'])
                file_path_split = file_path.rsplit('.', 1)
                file_path = file_path_split[0] + model_weights_filename.split('.')[0] + '.' + file_path_split[1]
                if not os.path.isdir(os.path.dirname(file_path)):
                    os.makedirs(os.path.dirname(file_path))
                with open(file_path, 'w') as f:
                    f.writelines(s + '\n' for s in step_text_list)


if __name__ == '__main__':
    generate_attack_text(reverse=False)
