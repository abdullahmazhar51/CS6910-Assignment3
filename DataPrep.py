import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Reading the files train, val and test
train_df = pd.read_csv('aksharantar_sampled/hin/hin_train.csv', header=None)
val_df = pd.read_csv('aksharantar_sampled/hin/hin_valid.csv', header=None)
test_df = pd.read_csv('aksharantar_sampled/hin/hin_test.csv', header=None)

# Renaming the header of DataFrame
train_df = train_df.rename(columns={0:'english', 1:'hindi'}) 
val_df = val_df.rename(columns={0:'english', 1:'hindi'}) 
test_df = test_df.rename(columns={0:'english', 1:'hindi'}) 

# Converting the dataframe into list
train_eng_words = train_df['english'].to_list()
train_hin_words = train_df['hindi'].to_list()
val_eng_words = val_df['english'].to_list()
val_hin_words = val_df['hindi'].to_list()
test_eng_words = test_df['english'].to_list()
test_hin_words = test_df['hindi'].to_list()

# Adding the SOS and EOS tokens
train_eng_sequence = [['<SOS>'] + [c for c in w] + ['<EOS>'] for w in train_eng_words]
train_hin_sequence = [['<SOS>'] + [c for c in w] + ['<EOS>'] for w in train_hin_words]

val_eng_sequence = [['<SOS>'] + [c for c in w] + ['<EOS>'] for w in val_eng_words]
val_hin_sequence = [['<SOS>'] + [c for c in w] + ['<EOS>'] for w in val_hin_words]

test_eng_sequence = [['<SOS>'] + [c for c in w] + ['<EOS>'] for w in test_eng_words]
test_hin_sequence = [['<SOS>'] + [c for c in w] + ['<EOS>'] for w in test_hin_words]

# Defining Empty Dictionary
char_to_index = {}

# Creating Dictionary to map each character to index for train data
for seq in train_eng_sequence + train_hin_sequence:
    for cha in seq:
        if cha not in char_to_index:
            char_to_index[cha] = len(char_to_index)

# Creating Dictionary to map each character to index for val data
for seq in val_eng_sequence + val_hin_sequence:
    for cha in seq:
        if cha not in char_to_index:
            char_to_index[cha] = len(char_to_index)
      
# Creating Dictionary to map each character to index for test data
for seq in test_eng_sequence + test_hin_sequence:
    for cha in seq:
        if cha not in char_to_index:
            char_to_index[cha] = len(char_to_index)

train_eng_int_sequence = [[char_to_index[c] for c in seq] for seq in train_eng_sequence]
print(len(train_eng_int_sequence))
train_hin_int_sequence = [[char_to_index[c] for c in seq] for seq in train_hin_sequence]

val_eng_int_sequence = [[char_to_index[c] for c in seq] for seq in val_eng_sequence]
val_hin_int_sequence = [[char_to_index[c] for c in seq] for seq in val_hin_sequence]

test_eng_int_sequence = [[char_to_index[c] for c in seq] for seq in test_eng_sequence]
test_hin_int_sequence = [[char_to_index[c] for c in seq] for seq in test_hin_sequence]

# Padding the sequences to fixed length
max_sequence_len = max(len(seq) for seq in test_eng_int_sequence + test_hin_int_sequence)
train_eng_int_seq = [seq + [0] * (max_sequence_len-len(seq)) for seq in train_eng_int_sequence] 
train_hin_int_seq = [seq + [0] * (max_sequence_len-len(seq)) for seq in train_hin_int_sequence] 

val_eng_int_seq = [seq + [0] * (max_sequence_len-len(seq)) for seq in val_eng_int_sequence] 
val_hin_int_seq = [seq + [0] * (max_sequence_len-len(seq)) for seq in val_hin_int_sequence] 

test_eng_int_seq = [seq + [0] * (max_sequence_len-len(seq)) for seq in test_eng_int_sequence] 
test_hin_int_seq = [seq + [0] * (max_sequence_len-len(seq)) for seq in test_hin_int_sequence] 

# Converting Data into Pytorch Tensor
train_eng_tensor = torch.LongTensor(train_eng_int_seq)
train_hin_tensor = torch.LongTensor(train_hin_int_seq)

val_eng_tensor = torch.LongTensor(val_eng_int_seq)
val_hin_tensor = torch.LongTensor(val_hin_int_seq)

test_eng_tensor = torch.LongTensor(test_eng_int_seq)
test_hin_tensor = torch.LongTensor(test_hin_int_seq)

# create TensorDatasets for the training, validation, and test sets
train_dataset = TensorDataset(train_eng_tensor, train_hin_tensor)
print(train_dataset)
val_dataset = TensorDataset(val_eng_tensor, val_hin_tensor)
test_dataset = TensorDataset(test_eng_tensor, test_hin_tensor)

# create DataLoaders for the training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
