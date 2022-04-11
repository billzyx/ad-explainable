import os
import re

import dataloaders


# find_str = "is getting"
# find_str = "is drying"

# find_str = ' is, '
# find_str = ' is; '
# find_str = ' is. '
# find_str = ' to. '
# find_str = ' it. '
# find_str = 'mother is, '
# find_str = 'it is o'
find_str = '. the '

print(find_str)
train_dataset = dataloaders.load_train_dataset('ADReSSo21-train', [43, 83, 123], [',', ';', '.'])
# train_dataset = dataloaders.load_test_dataset('ADReSSo21-test', [43, 83, 123], [',', ';', '.'])
train_dataset = [data for data in train_dataset]

for data in train_dataset:
    text = data['text'].lower()
    text = re.sub(r'\s([,.\'"](?:\s|$))', r'\1', text)
    text_no_punctuation = text.replace(',', '').replace(';', '').replace('.', '')
    # if find_str in text_no_punctuation:
    #     print(data['file_path'])
    #     print(text)
    #     print(data['label'])
    if find_str in text:
        print(data['file_path'])
        print(text)
        print(data['label'])
