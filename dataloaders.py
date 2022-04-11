from torch.utils.data.dataset import Dataset, ConcatDataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import core
import os
import re

train_path = 'ADReSS-IS2020-data/train'
test_path = 'ADReSS-IS2020-data/test'
test_label_path = 'ADReSS-IS2020-data/test/meta_data_test.txt'

train_21_path = 'ADReSSo21/diagnosis/train'
train_21_progression_path = 'ADReSSo21/progression/train'
test_21_path = 'ADReSSo21/diagnosis/test-dist'
test_21_label_task_1_path = 'ADReSSo21/diagnosis/test-dist/test_results_task1_groundtruth.csv'
test_21_label_task_2_path = 'ADReSSo21/diagnosis/test-dist/test_results_task2_groundtruth.csv'

attack_text_output_dir = 'attack_text'
reverse_attack_text_output_dir = 'reverse_attack_text'


def load_train_dataset(train_dataset_name, level_list, punctuation_list, train_filter_min_word_length=0):
    train_dataset = None
    if train_dataset_name == 'ADReSSo21-train':
        train_dataset = ADReSSo21TextTrainDataset(
            train_21_path, level_list, punctuation_list,
            filter_min_word_length=train_filter_min_word_length)
    elif train_dataset_name == 'ADReSS20-train':
        train_dataset = ADReSSTextTrainDataset(
            train_path, level_list, punctuation_list,
            filter_min_word_length=train_filter_min_word_length)
    elif train_dataset_name == 'ADReSS20-train-transcript':
        train_dataset = ADReSSTextTranscriptTrainDataset(
            train_path)
    elif train_dataset_name == 'ADReSSo21-progression-train':
        train_dataset = ADReSSo21TextProgressionTrainDataset(
            train_21_progression_path, level_list, punctuation_list,
            filter_min_word_length=train_filter_min_word_length)
    return train_dataset


def load_train_attack_dataset(train_dataset_name, model_description, log_dir, train_attack_step,
                              train_filter_min_word_length=0, reverse=False):
    train_dataset = None
    if train_dataset_name == 'ADReSS20-train':
        train_dataset = ADReSSTextAttackTrainDataset(
            train_path, model_description, log_dir, train_attack_step,
            filter_min_word_length=train_filter_min_word_length, reverse=reverse)
    if train_dataset_name == 'ADReSSo21-train':
        train_dataset = ADReSSo21TextAttackTrainDataset(
            train_21_path, model_description, log_dir, train_attack_step,
            filter_min_word_length=train_filter_min_word_length, reverse=reverse)
    return train_dataset


def load_test_attack_dataset(test_dataset_name, model_description, log_dir, attack_step, model_round_number,
                              test_filter_min_word_length=0):
    train_dataset = None
    if test_dataset_name == 'ADReSS20-test':
        train_dataset = ADReSSTextAttackTestDataset(
            test_path, test_label_path, model_description, log_dir, attack_step, model_round_number,
            filter_min_word_length=test_filter_min_word_length)
    if test_dataset_name == 'ADReSSo21-test':
        train_dataset = ADReSSo21TextAttackTestDataset(
            test_21_path, test_21_label_task_1_path, test_21_label_task_2_path, model_description, log_dir,
            attack_step, model_round_number,
            filter_min_word_length=test_filter_min_word_length)
    return train_dataset


def load_test_dataset(test_dataset_name, level_list, punctuation_list, test_filter_min_word_length=0):
    test_dataset = None
    if test_dataset_name == 'ADReSS20-train':
        test_dataset = ADReSSTextTrainDataset(
            train_path, level_list, punctuation_list,
            filter_min_word_length=test_filter_min_word_length)
    elif test_dataset_name == 'ADReSS20-test':
        test_dataset = ADReSSTextTestDataset(
            test_path, test_label_path, level_list, punctuation_list,
            filter_min_word_length=test_filter_min_word_length)
    elif test_dataset_name == 'ADReSS20':
        test_dataset_list = [ADReSSTextTrainDataset(
            train_path, level_list, punctuation_list,
            filter_min_word_length=test_filter_min_word_length),
            ADReSSTextTestDataset(
                test_path, test_label_path, level_list, punctuation_list,
                filter_min_word_length=test_filter_min_word_length)]
        test_dataset = ConcatDataset(test_dataset_list)
    elif test_dataset_name == 'ADReSS20-test-transcript':
        test_dataset = ADReSSTextTranscriptTestDataset(
            test_path, test_label_path)
    elif test_dataset_name == 'ADReSSo21-progression-train':
        test_dataset = ADReSSo21TextProgressionTrainDataset(
            train_21_progression_path, level_list, punctuation_list,
            filter_min_word_length=test_filter_min_word_length)
    elif test_dataset_name == 'ADReSSo21-test':
        test_dataset = ADReSSo21TextTestDataset(
            test_21_path, test_21_label_task_1_path, test_21_label_task_2_path, level_list, punctuation_list,
            filter_min_word_length=test_filter_min_word_length)
    return test_dataset


def get_file_text(file_path, level_list, punctuation_list):
    return core.load_file(file_path, level_list, punctuation_list)


def get_audio_embedding(file_path):
    file_path = file_path.replace('asr_text', 'asr_embedding').replace('.txt', '.npy')
    return np.load(file_path)


def get_attack_file_text(file_path, train_attack_step):
    text_file_list = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        text_file = dict()
        text_file['text'] = lines[0].strip()
        text_file_list.append(text_file)
        for line in lines[1:train_attack_step + 1]:
            text_file = dict()
            text_file['text'] = line.strip()
            if text_file['text'] == text_file_list[-1]['text'] or text_file['text'] == '':
                break
            text_file_list.append(text_file)
    return text_file_list


def get_attack_file_text_one_line(file_path, attack_step):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        text_file = dict()
        text_file['text'] = lines[0].strip()
        for line in lines[1:attack_step + 1]:
            text = line.strip()
            if text == text_file['text'] or text == '':
                break
            text_file['text'] = text
    return text_file


class ADReSSTextTrainDataset(Dataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, sentiment in (('cc', 0), ('cd', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = get_file_text(file_path, level_list, punctuation_list)
                # text_file['audio_embedding'] = self.get_audio_embedding(file_path)
                if len(text_file['text'].split()) > filter_min_word_length:
                    self.X.append(text_file)
                    self.Y.append(sentiment)
                    self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                    self.file_idx.append(name.split('.')[0])
                    self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        with open(os.path.join(dir_path, 'cc_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = lines[i].split(';')[3].strip()
                if file_label == 'NA':
                    file_label = 29
                else:
                    file_label = int(file_label)
                labels[file_id] = file_label
        with open(os.path.join(dir_path, 'cd_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                labels[file_id] = file_label
        # print(labels)
        return labels


class ADReSSo21TextTrainDataset(Dataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, sentiment in (('cn', 0), ('ad', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = get_file_text(file_path, level_list, punctuation_list)
                # text_file['audio_embedding'] = self.get_audio_embedding(file_path)
                if len(text_file['text'].split()) > filter_min_word_length:
                    self.X.append(text_file)
                    self.Y.append(sentiment)
                    self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                    self.file_idx.append(name.split('.')[0])
                    self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        data = pd.read_csv(os.path.join(dir_path, 'adresso-train-mmse-scores.csv'))
        df = pd.DataFrame(data)

        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            labels[row['adressfname']] = int(row['mmse'])
        # print(labels)
        return labels


class ADReSSTextTestDataset(Dataset):
    def __init__(self, dir_path, label_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        labels, mmse_labels = self.load_test_label(label_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, 'asr_text')))):
            file_path = os.path.join(dir_path, 'asr_text', name)
            text_file = get_file_text(file_path, level_list, punctuation_list)
            # text_file['audio_embedding'] = self.get_audio_embedding(file_path)
            if len(text_file['text'].split()) > filter_min_word_length:
                self.X.append(text_file)
                self.Y.append(labels[name.split('.')[0]])
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_path):
        labels = {}
        mmse_labels = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                file_mmse_label = int(lines[i].split(';')[4].strip())
                labels[file_id] = file_label
                mmse_labels[file_id] = file_mmse_label
        # print(labels)
        # print(mmse_labels)
        return labels, mmse_labels


class ADReSSTextTranscriptDataset(Dataset):
    def get_file_text(self, file_path):
        text_file = ''
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                text = line.strip().replace('	', ' ')
                if line.startswith('*'):
                    text = text.split(':', maxsplit=1)[1] + ' '
                    temp_idx = idx
                    while not '' in lines[temp_idx]:
                        temp_idx += 1
                        text += lines[temp_idx].strip() + ' '
                    text = text.split('')[0]
                    text_file += text

        # print(text_file)
        text_file = text_file.replace('_', ' ')
        text_file = re.sub(r'\[[^\]]+\]', '', text_file)
        text_file = re.sub('[^0-9a-zA-Z,. \'?]+', '', text_file)
        text_file = text_file.replace('...', '').replace('..', '')
        # print(text_file)
        return text_file


class ADReSSTextTranscriptTrainDataset(ADReSSTextTranscriptDataset):
    def __init__(self, dir_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        for folder, ad_label in (('cc', 0), ('cd', 1)):
            folder = os.path.join(dir_path, 'transcription', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = self.get_file_text(file_path)
                self.X.append(text_file)
                self.Y.append(ad_label)
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_mmse(self, dir_path):
        labels = {}
        with open(os.path.join(dir_path, 'cc_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = lines[i].split(';')[3].strip()
                if file_label == 'NA':
                    file_label = 29
                else:
                    file_label = int(file_label)
                labels[file_id] = file_label
        with open(os.path.join(dir_path, 'cd_meta_data.txt'), 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                labels[file_id] = file_label
        # print(labels)
        return labels


class ADReSSTextTranscriptTestDataset(ADReSSTextTranscriptDataset):
    def __init__(self, dir_path, label_path):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        labels, mmse_labels = self.load_test_label(label_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, 'transcription')))):
            file_path = os.path.join(dir_path, 'transcription', name)
            text_file = self.get_file_text(file_path)
            self.X.append(text_file)
            self.Y.append(labels[name.split('.')[0]])
            self.Y_mmse.append(mmse_labels[name.split('.')[0]])
            self.file_idx.append(name.split('.')[0])
            self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_path):
        labels = {}
        mmse_labels = {}
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines)):
                file_id = lines[i].split(';')[0].strip()
                file_label = int(lines[i].split(';')[3].strip())
                file_mmse_label = int(lines[i].split(';')[4].strip())
                labels[file_id] = file_label
                mmse_labels[file_id] = file_mmse_label
        # print(labels)
        # print(mmse_labels)
        return labels, mmse_labels


class ADReSSo21TextProgressionTrainDataset(Dataset):
    def __init__(self, dir_path, level_list, punctuation_list, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.file_idx = []
        self.file_path_list = []
        for folder, sentiment in (('no_decline', 0), ('decline', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file = get_file_text(file_path, level_list, punctuation_list)
                # text_file['audio_embedding'] = self.get_audio_embedding(file_path)
                if len(text_file['text'].split()) > filter_min_word_length:
                    self.X.append(text_file)
                    self.Y.append(sentiment)
                    self.file_idx.append(name.split('.')[0])
                    self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': 0.0,
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)


class ADReSSTextAttackTrainDataset(ADReSSTextTrainDataset):
    def __init__(self, dir_path, model_description, log_dir, train_attack_step, filter_min_word_length=0,
                 reverse=False):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        if not reverse:
            dir_path = os.path.join(attack_text_output_dir, log_dir, model_description, dir_path)
        else:
            dir_path = os.path.join(reverse_attack_text_output_dir, log_dir, model_description, dir_path)
        for folder, sentiment in (('cc', 0), ('cd', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file_list = get_attack_file_text(file_path, train_attack_step)
                for text_file in text_file_list:
                    if len(text_file['text'].split()) > filter_min_word_length:
                        self.X.append(text_file)
                        self.Y.append(sentiment)
                        self.Y_mmse.append(mmse_labels[name.split('model_round')[0]])
                        self.file_idx.append(name.split('.')[0])
                        self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            # 'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }


class ADReSSo21TextTestDataset(Dataset):
    def __init__(self, dir_path, label_ad_path, label_mmse_path, level_list, punctuation_list,
                 filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        labels, mmse_labels = self.load_test_label(label_ad_path, label_mmse_path)
        for name in tqdm(sorted(os.listdir(os.path.join(dir_path, 'asr_text')))):
            file_path = os.path.join(dir_path, 'asr_text', name)
            text_file = get_file_text(file_path, level_list, punctuation_list)
            # text_file['audio_embedding'] = self.get_audio_embedding(file_path)
            if len(text_file['text'].split()) > filter_min_word_length:
                self.X.append(text_file)
                self.Y.append(labels[name.split('.')[0]])
                self.Y_mmse.append(mmse_labels[name.split('.')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)

    def load_test_label(self, label_ad_path, label_mmse_path):
        labels = {}
        mmse_labels = {}

        data = pd.read_csv(label_ad_path)
        df = pd.DataFrame(data)
        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            labels[row['ID']] = 0 if row['Dx'] == "Control" else 1

        data = pd.read_csv(label_mmse_path)
        df = pd.DataFrame(data)
        for index, row in df.iterrows():
            # print(row[0], row['mmse'])
            mmse_labels[row['ID']] = int(row['MMSE'])

        # print(labels)
        # print(mmse_labels)
        return labels, mmse_labels


class ADReSSo21TextAttackTrainDataset(ADReSSo21TextTrainDataset):
    def __init__(self, dir_path, model_description, log_dir, train_attack_step, filter_min_word_length=0,
                 reverse=False):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        mmse_labels = self.load_mmse(dir_path)
        if not reverse:
            dir_path = os.path.join(attack_text_output_dir, log_dir, model_description, dir_path)
        else:
            dir_path = os.path.join(reverse_attack_text_output_dir, log_dir, model_description, dir_path)
        for folder, sentiment in (('cn', 0), ('ad', 1)):
            folder = os.path.join(dir_path, 'asr_text', folder)
            for name in tqdm(sorted(os.listdir(folder))):
                file_path = os.path.join(folder, name)
                text_file_list = get_attack_file_text(file_path, train_attack_step)
                for text_file in text_file_list:
                    if len(text_file['text'].split()) > filter_min_word_length:
                        self.X.append(text_file)
                        self.Y.append(sentiment)
                        self.Y_mmse.append(mmse_labels[name.split('model_round')[0]])
                        self.file_idx.append(name.split('.')[0])
                        self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            # 'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }


class ADReSSo21TextAttackTestDataset(ADReSSo21TextTestDataset):
    def __init__(self, dir_path, label_ad_path, label_mmse_path, model_description, log_dir,
                 attack_step=0, model_round_number=1, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        dir_path = os.path.join(attack_text_output_dir, log_dir, model_description, dir_path)
        labels, mmse_labels = self.load_test_label(label_ad_path, label_mmse_path)
        file_list = [x for x in os.listdir(os.path.join(dir_path, 'asr_text'))
                     if 'model_round{:03}'.format(model_round_number) in x]
        for name in tqdm(sorted(file_list)):
            file_path = os.path.join(dir_path, 'asr_text', name)
            text_file = get_attack_file_text_one_line(file_path, attack_step)
            # text_file['audio_embedding'] = self.get_audio_embedding(file_path)
            if len(text_file['text'].split()) > filter_min_word_length:
                self.X.append(text_file)
                self.Y.append(labels[name.split('model_round')[0]])
                self.Y_mmse.append(mmse_labels[name.split('model_round')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            # 'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)


class ADReSSTextAttackTestDataset(ADReSSTextTestDataset):
    def __init__(self, dir_path, label_path, model_description, log_dir,
                 attack_step=0, model_round_number=1, filter_min_word_length=0):
        self.X, self.Y = [], []
        self.Y_mmse = []
        self.file_idx = []
        self.file_path_list = []
        labels, mmse_labels = self.load_test_label(label_path)
        dir_path = os.path.join(attack_text_output_dir, log_dir, model_description, dir_path)
        file_list = [x for x in os.listdir(os.path.join(dir_path, 'asr_text'))
                     if 'model_round{:03}'.format(model_round_number) in x]
        for name in tqdm(sorted(file_list)):
            file_path = os.path.join(dir_path, 'asr_text', name)
            text_file = get_attack_file_text_one_line(file_path, attack_step)
            # text_file['audio_embedding'] = self.get_audio_embedding(file_path)
            if len(text_file['text'].split()) > filter_min_word_length:
                self.X.append(text_file)
                self.Y.append(labels[name.split('model_round')[0]])
                self.Y_mmse.append(mmse_labels[name.split('model_round')[0]])
                self.file_idx.append(name.split('.')[0])
                self.file_path_list.append(file_path)

    def __getitem__(self, idx):
        return {
            'file_idx': self.file_idx[idx],
            'text': self.X[idx]['text'],
            # 'original_text': self.X[idx]['original_text'],
            # 'audio_embedding': self.X[idx]['audio_embedding'],
            'label': self.Y[idx],
            'label_mmse': self.Y_mmse[idx],
            'file_path': self.file_path_list[idx],
        }

    def __len__(self):
        return len(self.X)


