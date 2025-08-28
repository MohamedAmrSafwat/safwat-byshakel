import streamlit as st
# model definition and the predict function

import torch
import json
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('char_to_idx.json', 'r', encoding='utf-8') as f:
    char_to_idx = json.load(f)
idx_to_char = {v: k for k, v in char_to_idx.items()}

with open('diac_to_idx.json', 'r', encoding='utf-8') as f:
    diac_to_idx = json.load(f)
idx_to_diac = {v: k for k, v in diac_to_idx.items()}
max_char_per_seq = 133
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
model = TasheelModel(len(char_to_idx), len(diac_to_idx)).cpu()
model = torch.load("model-1-8500.pth", map_location=torch.device('cpu'))

def predict(sentence):
    sequence_length = torch.tensor([len(sentence)]).cpu()
    chars = list(sentence) 
    if len(chars) < max_char_per_seq:
        chars = chars + ["<PAD>"] * (max_char_per_seq - len(chars))
    indexes = [char_to_idx[char] for char in chars]    
    indeces = torch.tensor(indexes, dtype=torch.int64).cpu()
    indeces = indeces.unsqueeze(0)
    
    with torch.no_grad():
        logits =  model(indeces, sequence_length)
    out = torch.softmax(logits, dim=-1)
    letters = []
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

#sentence = input("Enter a sentence: ")
#prediction = predict(sentence)
#print(prediction)
st.markdown(
    """
    <style>
        /* Remove extra bottom padding */
        .block-container {
            padding: 0rem 1rem 0rem 1rem;
        }

        /* Title styling */
        h1 {
            font-size: 100px !important;
            font-weight: bold;
            font-family: 'Cairo', sans-serif;
            color: #004d00; /* dark green */
            text-align: center;
        }

        /* Predict button */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #66ff99, #99ffcc);
            color: #004d00;
            font-size: 40px;
            padding: 18px 45px;
            border-radius: 12px;
            border: 2px solid #004d00;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            font-family: 'Cairo', sans-serif;
            box-shadow: 0px 4px 10px rgba(0, 77, 0, 0.3);
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #99ffcc, #66ff99);
            transform: scale(1.08);
            box-shadow: 0px 6px 15px rgba(0, 77, 0, 0.5);
        }

        /* Background green-white gradient */
        .stApp {
            background: linear-gradient(-45deg, #ccffdd, #ffffff, #e6ffee, #f9fff9);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            color: black;
        }

        @keyframes gradientBG {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* Labels + markdown (Arabic right aligned) */
        .stMarkdown, .stTextArea label {
            font-family: 'Cairo', sans-serif !important;
            direction: rtl !important;
            text-align: right !important;
            color: black !important;
        }

        /* TextArea styling (make inside text right aligned) */
        textarea {
            background: #ffffff !important;
            color: #000000 !important;
            border: 2px solid #66cc99 !important;
            border-radius: 10px !important;
            font-family: 'Cairo', sans-serif !important;
            direction: rtl !important;
            text-align: right !important;
            padding-right: 20px !important;
            box-shadow: 0px 3px 10px rgba(0, 77, 0, 0.2) !important;
        }

        /* Fix disabled textarea color */
        textarea[disabled] {
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            opacity: 1 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)



st.markdown(
    """
    <h1 style='text-align: center; margin-top: -50px;'>
        صَفْوَت بِيِشْكِل
    </h1>
    """,
    unsafe_allow_html=True
)

# Two columns: input / output
col1, col2 = st.columns([1, 1])

# Initialize session state
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "predicted_text" not in st.session_state:
    st.session_state.predicted_text = "بانتظار إدخال النص..."

with col2:
    st.markdown("### النص الأصلي")
    st.session_state.input_text = st.text_area("أدخل النص هنا", value=st.session_state.input_text, height=400, placeholder="اكتب نصك...")    

with col1:
    st.markdown("### النص الناتج")
    st.text_area("الناتج", value=st.session_state.predicted_text, height=400, key="readonly_output", disabled=True, placeholder="بانتظار إدخال النص...")

# Predict button centered
col1, col2, col3 = st.columns([1.4,1,1])

with col2:
    if st.button("شكل", key="predict_btn"):
        text = st.session_state.input_text.strip()
        if text:
            st.session_state.predicted_text = predict(text)  # model call
        else:
            st.session_state.predicted_text = "⚠️ الرجاء إدخال نص."
        st.rerun()
