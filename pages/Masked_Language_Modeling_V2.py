import streamlit as st
import streamlit.components.v1 as components
import os, sys, io, urllib, time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from datetime import datetime
from pathlib import Path
from PIL import Image

# importing pytorch libraries
import torch
from torch import nn
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader, TensorDataset

# importing data science libraries
import pandas as pd
import numpy as np
from numpy import random

from utils.data_config import input_config
from utils.fraud_data import FraudData

# model interpretation
import shap
from sklearn import metrics
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from transformers_interpret import SequenceClassificationExplainer
from bertviz import head_view, model_view
from bertviz.neuron_view import show

import transformers
from transformers import BertForSequenceClassification
from transformers import BertTokenizer,BertConfig,BertForMaskedLM,MobileBertTokenizer
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import capture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pure_comma_separation(list_str, return_list=True):
    r = unique_list([item.strip() for item in list_str.lower().split(",") if item.strip()])
    if return_list:
        return r
    return ", ".join(r)


@st.cache_data
def load_sample_data():
    #### load final features ######
    with open('./data/final_selected_features.npy','rb') as f:
        vars_rc_list = np.load(f)

    vars_rc_list = vars_rc_list.tolist()
    vars_rc_list = [x for x in vars_rc_list if x.startswith(('tsrc', 'tmxrc'))]

    print('total var length:', len(vars_rc_list))

    csrt_test = pd.read_csv('./data/newdatawithmorefeatures.csv',usecols=vars_rc_list + ['frd'])

    return csrt_test,vars_rc_list


@st.cache_resource
def prepare_mlm_model():
    # Set Tokenizer
    vocab_file = "./checkpoints/v2/vocab.txt"
    special_tokens_dict = {"unk_token": "[UNK]",
                           "sep_token": "[SEP]",
                           "pad_token": "[PAD]",
                           "cls_token": "[CLS]",
                           "mask_token": "[MASK]",
                           "eos_token": "[EOS]",
                           "bos_token": "[BOS]"}
    tokenizer = MobileBertTokenizer(vocab_file, do_basic_tokenize=False)
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load Model
    newmodel = BertForMaskedLM.from_pretrained('checkpoints/v2/pretrain/checkpoint-36000',output_attentions=True)
    mask_filler = transformers.pipeline("fill-mask", model=newmodel, tokenizer=tokenizer, top_k=2)

    return mask_filler, newmodel, tokenizer


def st_plot_text_shap(shap_val, height=600):
    InteractiveShell().instance()
    with capture.capture_output() as cap:
        shap.plots.text(shap_val[:, :, "LABEL_1"])
    components.html(cap.outputs[0].data['text/html'], height=height, scrolling=True)


if __name__ == '__main__':

    st.set_page_config(page_title="FraudTransformer Demo V2", layout='wide')

    if "load_state" not in st.session_state:
        st.session_state.load_state = 0

    input_data, features_name = load_sample_data()

    test_data_fraud = input_data.loc[input_data['frd'] == 1][features_name]

    st.sidebar.header("FraudTransformer")
    st.sidebar.markdown("---")
    st.sidebar.markdown("1. Choose Input Data")

    picked_data = st.sidebar.button('Picked One Fraud Sample Randomly', key='step10')

    # st.sidebar.markdown("---")
    # st.sidebar.markdown("2. Model Evaluation")
    #
    # shap_plot = st.sidebar.checkbox('Shap Explainer', key='step31')
    # NLP_plot = st.sidebar.checkbox('NLP Explainer', key='step32')
    # run_model = st.sidebar.button('Run the Model', key='step20')

    st.sidebar.markdown("---")

    st.title("FraudTransformer Demo V2")
    st.header("")

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.write(
                """
        **About This Demo**
        - Experience the future of fraud detection with FraudTransformer! 
        - This trailblazing model leverages the power of transformer-based Large Language Models (LLMs) and deep learning to revolutionize the processing of fraud-specific data.

        - Some reference papers that help build this idea:
            - TabTransformer: Tabular Data Modeling Using Contextual Embeddings
            - TabNet: Attentive Interpretable Tabular Learning
            - SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
            - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

        ***

        **FraudTransformer Architecture (See Right Figure)** 

        - BERT (Bidirectional Encoder Representations from Transformers) inspired,
        - Fraud specific tokenizer,
        - Pre-training and fine-tuning phases
                """
            )

    with col2:
        with st.container():
            image = Image.open('data/transformer_NLP.png')
            st.image(image)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        with st.container():
            if picked_data:
                st.subheader("**Data Sample**")
                st.session_state.load_state = random.randint(test_data_fraud.shape[0])
                X_test_df_picked_frd = test_data_fraud.sample(random_state=st.session_state.load_state).T
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)
            elif st.session_state.load_state:
                st.subheader("**Data Sample**")
                X_test_df_picked_frd = test_data_fraud.sample(random_state=st.session_state.load_state).T
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Samße as st.write(df)
            else:
                st.subheader("**Data Sample**")
                st.session_state.load_state = random.randint(test_data_fraud.shape[0])
                X_test_df_picked_frd = test_data_fraud.sample(random_state=st.session_state.load_state).T
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)
    with col2:
        with st.container():
            st.subheader("**Masked Language Modelling**")
            st.write(
                """
                **Masked language modeling is the task of masking some of the words in a sentence and predicting which words should replace those masks.**
                - Mask a feature means our model does not have this feature in the input.
                - The model will try to guess what is the missing value.
                - This is different from classification, it is a kind of generation task.
                """
            )
            columns_name = X_test_df_picked_frd.index.values.tolist()
            columns_value = X_test_df_picked_frd.iloc[:,0].tolist()
            text = [x + '_' + str(y) for x, y in zip(columns_name,columns_value)]
            s = ' '.join(text)

            option = st.selectbox(
                'Which feature you want to mask? :sunglasses:',
                text)

            st.write('You selected:', option)

            masked_text = s.replace(option, "[MASK]")


            txt = st.text_area('Use [MASK] to replace any input feature', value=masked_text,height=800)



    st.markdown("---")

    with st.container():

        st.subheader("**Model Performance**")

        mask_filler,_,_ = prepare_mlm_model()

        st.write('Filled Mask:', mask_filler(txt))



    st.markdown("---")

    with st.container():

        st.subheader("**Visualizing Model Attention**")

        _, newmodel, tokenizer = prepare_mlm_model()

        inputs = tokenizer.encode_plus(s, return_tensors='pt')
        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention = newmodel(input_ids, token_type_ids=token_type_ids)[-1]
        input_id_list = input_ids[0].tolist()  # Batch index 0
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)

        html_head_view = head_view(attention, tokens, html_action='return')

        with open("head_view.html", 'w') as file:
            file.write(html_head_view.data)

        with open('head_view.html', 'r') as f:
            html_data = f.read()

        ## Show in webpage
        st.components.v1.html(html_data, height=2000, scrolling=True)
