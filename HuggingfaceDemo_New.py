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

import transformers
from transformers import BertForSequenceClassification
from transformers import BertTokenizer,MobileBertTokenizer
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import capture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def prepare_model():
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
    finetunemodel = BertForSequenceClassification.from_pretrained('./checkpoints/v2/finetune/checkpoint-139290',
                                                                  num_labels=2)
    classifier = transformers.pipeline("sentiment-analysis", model=finetunemodel, tokenizer=tokenizer, top_k=None)
    explainer = shap.Explainer(classifier)
    cls_explainer = SequenceClassificationExplainer(finetunemodel, tokenizer)

    return classifier, explainer, cls_explainer


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

    st.sidebar.markdown("---")
    st.sidebar.markdown("2. Model Evaluation")

    shap_plot = st.sidebar.checkbox('Shap Explainer', key='step31')
    NLP_plot = st.sidebar.checkbox('NLP Explainer', key='step32')
    run_model = st.sidebar.button('Run the Model', key='step20')

    st.sidebar.markdown("---")

    st.title("FraudTransformer Demo V2")
    st.header("")

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.write(
                """
        **About This Demo**
        - The *FraudTransformer* utilizes transformer architecture for categorical fraud data,
        - Trained on **1273502** samples with **119** features (tsrc and tmxrc)
        - Evaluated on **333086** samples.
        - This model is tested and deployed on mltitan server

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
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)
            else:
                st.subheader("**Data Sample**")
                st.session_state.load_state = random.randint(test_data_fraud.shape[0])
                X_test_df_picked_frd = test_data_fraud.sample(random_state=st.session_state.load_state).T
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)
    with col2:
        with st.container():
            st.subheader("**Model Structure**")
            code = '''BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(245, 256, padding_idx=0)
      (position_embeddings): Embedding(512, 256)
      (token_type_embeddings): Embedding(2, 256)
      (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-7): 8 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=256, out_features=256, bias=True)
            ......
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=256, bias=True)
            (LayerNorm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=256, out_features=256, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=256, out_features=2, bias=True)
)'''
            st.code(code, language='python')
            # st.metric(label="Overall AUC on Test Data (333086 samples)", value="0.9005044738690602")
            auc_image = Image.open('data/auc.png')
            st.image(auc_image,caption='AUC 0.9005044738690602 on Test Data (333086 samples)')

    st.markdown("---")

    with st.container():

        st.subheader("**Model Performance**")

        if run_model:

            ### load data
            with st.spinner('Model Loading ......'):
                classifier, explainer, cls_explainer = prepare_model()
                test_data_features = test_data_fraud.sample(random_state=st.session_state.load_state)

                for i in range(0, test_data_features.shape[1]):
                    test_data_features.iloc[:, i] = test_data_features.iloc[:, i].apply(
                        lambda x: f"{features_name[i]}_{x}")

                x = test_data_features.astype(str).values.tolist()

                for i, j in enumerate(x):
                    x[i] = ' '.join(j)

                predictions = classifier(x[0])
            st.success('Model Loaded!')
            st.markdown("**classification probability**", unsafe_allow_html=False)
            st.json(predictions[0], expanded=True)

            if shap_plot:
                st.markdown("**SHAP value plot**", unsafe_allow_html=False)
                with st.spinner('Calculating SHAP value ......'):
                    shap_values = explainer([x[0]])
                    st_plot_text_shap(shap_values, 400)

            if NLP_plot:
                st.markdown("**NLP explainer**", unsafe_allow_html=False)
                with st.spinner('Preparing NLP explainer ......'):
                    word_attributions = cls_explainer(x[0], class_name="LABEL_1")
                    # word_attributions = cls_explainer(x[0])
                    cls_explainer.visualize("distilbert_viz.html")
                    with open('distilbert_viz.html', 'r') as f:
                        html_data = f.read()
                    ## Show in webpage
                    st.components.v1.html(html_data, height=400, scrolling=True)