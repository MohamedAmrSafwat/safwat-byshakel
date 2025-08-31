# import necessary library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from re import sub
import json
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
Learning_rate = 0.01
max_char_per_seq = 133
num_epochs = 30
batch_size = 64

# data preprocessing
# creating X and y


# Load
with open('char_to_idx.json', 'r', encoding='utf-8') as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}

with open('diac_to_idx.json', 'r', encoding='utf-8') as f:
    diac_to_idx = json.load(f)
idx_to_diac = {v: k for k, v in diac_to_idx.items()}


all_letters_code = []
all_diacritics_code = []
sequence_lengths  = []

no_of_lines = 0
sum_of_letters = 0
max_no_letters = 0
diac_in_row_count = 0

main_file = "data/tashkeela_train" 
training_files = [f for f in os.listdir(main_file) if f.endswith('.txt')]
for filename in training_files:
    filepath = os.path.join(main_file, filename)
    with open(filepath, 'r') as file:
        for line in file:
            no_of_lines += 1
            letters_code = []
            diacritics_code = []
            text = sub(r'[^\u0600-\u06FF\u064B-\u065F\u0660-\u0669\u06F0-\u06F9\s]',  '', line).strip()
            text = sub(r'[،؛؟]', '', text)
            text = sub(r'[٪]', '', text)
            text = sub(r'[ٱ]', 'ا', text)
            text = sub(r'[ۘ ]', '', text)
            previous_char = False
            for char in text:
                code = ord(char)
                if (0x064B <= code <= 0x065F) or (code == 0x0670) or (0x0610 <= code <= 0x061A):
                    if (previous_char):
                        diacritics_code.append(diac_to_idx[char])
                        previous_char = False
                    else:
                        diac_in_row_count += 1
                elif previous_char:
                    diacritics_code.append(diac_to_idx[""])
                    letters_code.append(char_to_idx[char])
                    previous_char = True
                else:
                    letters_code.append(char_to_idx[char])
                    previous_char = True
            if (previous_char):
                diacritics_code.append(diac_to_idx[""])
                previous_char = False
            sum_of_letters += len(letters_code)
            if len(letters_code) > max_no_letters:
                max_no_letters = len(letters_code)
            while len(letters_code) > max_char_per_seq:
                sequence_lengths.append(max_char_per_seq)
                all_letters_code.append(letters_code[:max_char_per_seq])
                all_diacritics_code.append(diacritics_code[:max_char_per_seq])
                letters_code = letters_code[max_char_per_seq:]
                diacritics_code = diacritics_code[max_char_per_seq:]
            
            sequence_lengths.append(len(letters_code))

            for i in range(len(letters_code), max_char_per_seq):
                letters_code.append(char_to_idx["<PAD>"])
                diacritics_code.append(diac_to_idx["<PAD>"])
                
            
            #print(len(letters_code), len(diacritics_code))
            #print(letters_code)
            #print(diacritics_code)
            all_letters_code.append(letters_code)
            all_diacritics_code.append(diacritics_code)
    
print("Number of lines: ", no_of_lines)
print("Number of letters: ", sum_of_letters)
print("Average number of letters per line: ", sum_of_letters/no_of_lines)
print("Max number of letters per line: ", max_no_letters) # line 8981 was 768, 2086 was 1112
print("shape of letters_code: ", len(all_letters_code), len(all_letters_code[2086]))
print("shape of diacritics_code: ", len(all_diacritics_code), len(all_diacritics_code[2086]))
print("Number of diacritics in a row: ", diac_in_row_count)
# plot seqence lengths
"""
plt.figure(figsize=(12, 6))
plt.hist(sequence_lengths, bins=150, edgecolor='black', alpha=0.7)
plt.title('Distribution of Arabic Text Sequence Lengths (Characters)')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.axvline(np.mean(sequence_lengths), color='k', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(sequence_lengths):.2f}')
plt.axvline(np.percentile(sequence_lengths, 95), color='r', linestyle='dashed', linewidth=1, label=f'95th Percentile: {np.percentile(sequence_lengths, 95)}')
plt.legend()
plt.show()
"""
#for i in range(len(all_letters_code)-10 ,len(all_letters_code)-1):
#    print(all_letters_code[i], " : ", all_diacritics_code[i])


class TasheelModel(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(TasheelModel, self).__init__()
        self.embedding = nn.Embedding(input_size, 512, padding_idx=0)
        #self.prenet = nn.Sequential(
        #    nn.Linear(512, 512),
        #    nn.ReLU(),
        #    nn.Dropout(0.5),
        #    nn.Linear(512, 256),
        #    nn.ReLU(),
        #    nn.Dropout(0.5)
        #)
        self.bilstm = nn.LSTM(512, 256, bidirectional=True,num_layers=3, batch_first=True)
        self.fc = nn.Linear(512, output_size)  # *2 for bidirectional GRU
        #self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, lengths): # add lengths
        embedded = self.embedding(x)  # (batch, seq_len, 512)
        #x = self.prenet(x)  # (batch, seq_len, 256)
        packed_input = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        bilstm_out, _ = self.bilstm(packed_input)
        packed_out, _ = pad_packed_sequence(bilstm_out, batch_first=True, total_length=max_char_per_seq)
        x = self.fc(packed_out) 
        #x = self.softmax(x)
        return x
def predict(sentence):
    sequence_length = torch.tensor([len(sentence)])#.to(device)
    print(sequence_length)
    chars = list(sentence) 
    if len(chars) < max_char_per_seq:
        chars = chars + ["<PAD>"] * (max_char_per_seq - len(chars))
    #print(chars)
    indexes = [char_to_idx[char] for char in chars]    
    indeces = torch.tensor(indexes, dtype=torch.int64).to(device)
    indeces = indeces.unsqueeze(0)
    
    with torch.no_grad():
        logits =  model(indeces, sequence_length)
    out = torch.softmax(logits, dim=-1)
    print(out.shape)
    letters = []
    #n, _ = out.shape
    #print(out.shape)
    for i in range(sequence_length):
        if chars[i] == " ":
            letters.append(" ")
            continue
        if chars[i] == "<PAD>":
            break
        letters.append(chars[i])
        char = torch.argmax(out[0][i]).item()
        letters.append(idx_to_diac[char])
    prediction = "".join(letters)
    return prediction

X = torch.tensor(all_letters_code).to(device)
Y = torch.tensor(all_diacritics_code).to(device)
lengths = torch.tensor(sequence_lengths, dtype=torch.int64)#.to(device)
print(X.shape, Y.shape, lengths.shape)#, lengths.shape)
print(X[0], "\n" ,Y[0])
print(X[0].shape, Y[0].shape)
dataset = TensorDataset(X, Y,lengths)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
input_size = 39  # Vocabulary size
output_size = 35 # Number of diacritics
X_trial = X[0:2].to(device)
length_trial = lengths[0:2].to(device)
#print(X_trial)
#print(length_trial)
#embedding = nn.Embedding(input_size, 512)
#bilstm = nn.LSTM(512, 256, bidirectional=True,)
#out = embedding(X_trial)
#print(out.shape)
#out  = pack_padded_sequence(out, length_trial.cpu(), batch_first=True, enforce_sorted=False)
#bilstm_out, _ = bilstm(out)
#print(out)

model = TasheelModel(input_size, output_size).to(device)

#input("Press Enter to continue...")
#sentence = input("Enter a sentence: ")
print(predict("انا اسمي محمد")) # اَنَا اسْمِي مُحَمَّدْ
time.sleep(2)

with torch.no_grad():
    out =  model(X_trial, length_trial)

for i in range(len(out[0])):
    if Y[0][i].item() == 0:
        break
    print(X[0][i].item() , " : " , Y[0][i].item() , " : " , torch.argmax(out[0][i]).item())
# define loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=Learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=0)
minloss = 1000000000000000000000000000000000
# training loop
print("Training...")
for epoch in range(num_epochs):
    Loss = 0
    for i, (X_batch, Y_batch, lengths_batch) in enumerate(dataloader):# add lenght
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        # forward
        out = model(X_batch, lengths_batch) # (batch_size, seq_len)
        out = out.permute(0, 2, 1)
        # loss
        loss = criterion(out, Y_batch)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print loss
        Loss += loss.item()
        print(f'\rEpoch [{epoch + 1}/{num_epochs}] Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}', end="")
        if (i + 1) % 100 == 0:
            model_name = f"model-{epoch + 1}-{i + 1}.pth"
            torch.save(model, model_name)
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {Loss:.4f}')
    if (Loss < minloss):
        minloss = Loss
        model_name = f"model-{epoch + 1}.pth"
        torch.save(model, model_name)

with torch.no_grad():
    out =  model(X_trial, length_trial)


for i in range(len(out[0])):
    if Y[0][i].item() == 0:
        break
    print(X[0][i] , " : " , Y[0][i] , " : " , torch.argmax(out[0][i]).item())

print(predict("انا اسمي محمد")) # اَنَا اسْمِي مُحَمَّدْ


# save model
#torch.save(model.state_dict(), 'model.ckpt')