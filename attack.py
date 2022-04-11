import numpy as np
from torch.utils.data.dataset import Dataset, ConcatDataset
import json
import torch
from sklearn.metrics import classification_report, mean_squared_error
import re

import model_head
import dataloaders
import core

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_attack_text_list_replace(text, punctuation_list=(',', '.')):

    punctuation_idx_list = [m.start() for m in re.finditer('[' + ''.join(punctuation_list) + ']', text)]
    # print(punctuation_idx_list)

    attack_text_list = []

    for punctuation_idx in punctuation_idx_list:
        for punctuation in punctuation_list:
            if text[punctuation_idx] != punctuation:
                attack_text = text[:punctuation_idx] + punctuation + text[punctuation_idx + 1:]
                attack_text_list.append(attack_text)

        attack_text = text[:punctuation_idx] + '' + text[punctuation_idx + 1:]
        attack_text_list.append(attack_text)

    # print(attack_text_list)
    return attack_text_list


def generate_attack_text_list_add(text, punctuation_list=(',', '.')):
    attack_text_list = []

    text_split_list = text.split()
    for idx in range(1, len(text_split_list)):
        if len(set(punctuation_list) - set(list(text_split_list[idx - 1]))) == len(punctuation_list):
            # if ',' not in text_split_list[idx - 1] and '.' not in text_split_list[idx - 1]:
            for punctuation in punctuation_list:
                attack_text = ' '.join(text_split_list[:idx]) + punctuation + ' ' + ' '.join(text_split_list[idx:])
                attack_text_list.append(attack_text)

    # print(attack_text_list)
    return attack_text_list


def mix_attack(text, attack_method_list):
    attack_text_list = [text]
    for attack_method in attack_method_list:
        attack_text_list_new = []
        for attack_text in attack_text_list:
            attack_text_list_new.extend(attack_method(attack_text))
        attack_text_list.extend(attack_text_list_new)
    return attack_text_list


def run_model_inference(text, tokenizer, model):
    inputs = [text]
    inputs = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(device)
    outputs = model(inputs)
    outputs_classification, outputs_regression = outputs[0], outputs[1]
    return outputs_classification, outputs_regression


def attack_function(text, label, tokenizer, model, step=20, reverse=False, punctuation_list=(',', '.')):
    best_attack_text = text
    for _ in range(step):
        attack_text_list_replace = generate_attack_text_list_replace(best_attack_text, punctuation_list=punctuation_list)
        attack_text_list_add = generate_attack_text_list_add(best_attack_text, punctuation_list=punctuation_list)
        attack_text_list = attack_text_list_replace + attack_text_list_add
        # attack_text_list = mix_attack(text, [generate_attack_text_list_replace, generate_attack_text_list_add])
        best_attack_text = attack_step_function(attack_text_list, label, model, best_attack_text, tokenizer,
                                                reverse=reverse)
    return best_attack_text


def attack_step_function(attack_text_list, label, model, text, tokenizer, reverse=False):
    outputs_classification, outputs_regression = run_model_inference(text, tokenizer, model)
    outputs_classification_prob = torch.softmax(outputs_classification, dim=-1)
    best_attack_text = text
    largest_output_difference = 0.0
    for attack_text in attack_text_list:
        attack_outputs_classification, attack_outputs_regression = run_model_inference(attack_text, tokenizer, model)
        attack_outputs_classification_prob = torch.softmax(attack_outputs_classification, dim=-1)
        if not reverse:
            output_difference = outputs_classification_prob[0, label] - attack_outputs_classification_prob[0, label]
        else:
            output_difference = attack_outputs_classification_prob[0, label] - outputs_classification_prob[0, label]
        if output_difference > largest_output_difference:
            largest_output_difference = output_difference
            best_attack_text = attack_text
    return best_attack_text


if __name__ == '__main__':
    generate_attack_text_list_add('animals, and you can begin. now at a a. goat a horse a a dog a bit es notan animal')
