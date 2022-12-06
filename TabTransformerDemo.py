import streamlit as st
import os, sys, io, urllib, time
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from datetime import datetime
from pathlib import Path
from PIL import Image

# from utils.data_config import input_config
# from models.tab_transformer import TabTransformer
# from utils.fraud_data import FraudData
# from utils.model_training import train_tool, test_tool, predict_prob,predict_prob_noy
# from utils.early_stopping import EarlyStopping

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
from utils.supports import numpy_to_tensor
from models.tab_transformer import TabTransformer
from utils.model_training import train_tool, test_tool, predict_prob,predict_prob_noy
from utils.early_stopping import EarlyStopping

# model interpretation
import shap
from sklearn import metrics
from sklearn.metrics import roc_auc_score,classification_report,confusion_matrix, ConfusionMatrixDisplay,roc_curve

# importing python plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sample_data():
    config = input_config("US_banking_login_test")  #### like "US_ecommerce_login_test"

    date, event_type, primary_industry = config["date_range"][0], config["event_types"][0], config["primary_industry"][
        0]
    df = pd.read_csv(f'./data/fe_{primary_industry}_{event_type}_{date[0]}_{date[1]}.csv')

    org_rcs = [x for x in df.columns if x.startswith(('tmxrc', 'sumrc'))]

    modified_coef = pd.read_csv('./data/modified_coef_final4th.csv', index_col='Unnamed: 0')

    modified_rcs = list(modified_coef.index)

    fix_feat = ['frd', 'datetime', 'eval', 'policy_score', 'org_id', 'policy', 'transaction_amount']

    org_rcs_input = org_rcs + fix_feat
    keep_rcs = modified_rcs + fix_feat

    df_2 = df[[rc for rc in org_rcs_input if rc in set(df.columns)]]
    remove_rcs = [rc for rc in org_rcs_input if rc not in keep_rcs]
    df_input = df_2[df_2['policy_score'].notna()]

    if len(remove_rcs) > 1:
        removed_rc = "multiple rcs"
    elif len(remove_rcs) == 1:
        removed_rc = remove_rcs[0]
    df_rcs = set(list(df_input.columns))
    rcs_to_remove = set([rc for rc in remove_rcs if rc in df_rcs])  ## when remove rc, make sure it is in the list

    input_data = FraudData(data_source=df_input)

    input_data.train = input_data.train.drop([rc for rc in rcs_to_remove], axis=1)
    input_data.val = input_data.val.drop([rc for rc in rcs_to_remove], axis=1)
    input_data.test = input_data.test.drop([rc for rc in rcs_to_remove], axis=1)

    for rc in rcs_to_remove:
        input_data.predictors.remove(rc)  ## remove targetted rc from predictor list

    X_train = input_data.train[input_data.predictors].to_numpy()
    y_train = input_data.train['frd'].to_numpy()
    X_val = input_data.val[input_data.predictors].to_numpy()
    y_val = input_data.val['frd'].to_numpy()
    X_test = input_data.test[input_data.predictors].to_numpy()
    y_test = input_data.test['frd'].to_numpy()

    return (X_train,y_train),(X_val,y_val),(X_test,y_test),modified_rcs



def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'red' if val == 1 else 'black'
    return 'color: %s' % color

# @st.cache
def prepare_input(X_test,y_test):

    # prepare PyTorch Datasets
    X_test_tensor = numpy_to_tensor(
        X_test, torch.long)
    y_test_tensor = numpy_to_tensor(
        y_test, torch.long)
    test_ds = TensorDataset(X_test_tensor, y_test_tensor)
    # #### model inference
    BATCH_SIZE = 50

    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=False, num_workers=0)

    return test_dl


@st.cache
def prepare_model(model_path):
    cat_feature_dims = [2] * 45

    model = TabTransformer(cat_feature_dims, embed_dim=32, depth=2, n_heads=4, att_dropout=0.5, an_dropout=0.5,
                           ffn_dropout=0.5, mlp_dims=[16, 16]).to(device)

    model.load_state_dict(torch.load(model_path))

    return model


def check_prediction(pred_classes, pred_label, ndvi_nrow, groud_truth_df):
    class_names = ['Barley', 'Canola', 'Chickpea', 'Lentils', 'Wheat']
    ground_truth = groud_truth_df.iloc[ndvi_nrow].to_numpy()
    st.markdown("Ground Truth:{}".format(class_names[ground_truth[0]]))
    if np.equal(ground_truth, pred_label):
        st.markdown("Correct!!!")
        st.balloons()


if __name__ == '__main__':
    # (X_train,y_train),(X_val,y_val),(X_test,y_test),cols = load_sample_data()
    # np.save('data/X_train.npy', X_train)
    # np.save('data/y_train.npy', y_train)
    # np.save('data/X_val.npy', X_val)
    # np.save('data/y_val.npy', y_val)
    # np.save('data/X_test.npy', X_test)
    # np.save('data/y_test.npy', y_test)

    st.set_page_config(page_title="FraudTransformer Demo",layout='wide')

    if "load_state" not in st.session_state:
        st.session_state.load_state = 0

    modified_coef = pd.read_csv('./data/modified_coef_final4th.csv', index_col='Unnamed: 0')
    cols = list(modified_coef.index)

    X_test = np.load('data/X_test.npy') # load
    y_test = np.load('data/y_test.npy')

    X_test_df = pd.DataFrame(X_test,columns=cols)
    X_test_df['frd'] = y_test


    st.sidebar.header("FraudTransformer")
    st.sidebar.markdown("---")
    st.sidebar.markdown("1. Choose Input Data")

    picked_data = st.sidebar.button('Picked One Fraud Sample Randomly',key='step1')

    st.sidebar.markdown("---")
    st.sidebar.markdown("2. Model Execution")

    data_model_loaded = st.sidebar.checkbox('Load Data&Model',key='step21')
    model_run = st.sidebar.checkbox('Run the Model', key='step22')

    st.sidebar.markdown("---")
    st.sidebar.markdown("3. Model Evaluation")

    Confusion_matrix_plot = st.sidebar.checkbox('Confusion Matrix',key='step31')
    AUC_plot = st.sidebar.checkbox('Run the Model', key='step32')

    st.title("FraudTransformer Demo")
    st.header("")

    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            st.write(
                """
        **About This Demo**     
        
        - The *FraudTransformer* is my first try at applying transformer architecture to fraud features (categorical type),
        - Differ from what we plan to test in the summer internship project, this one is designed to support original fraud data,
        - This demo model is trained on the MLTitan CPU server, without hyperparameter tuning, using about 20 epochs, 
        - Some reference papers to help build this idea:
            - TabTransformer: Tabular Data Modeling Using Contextual Embeddings
            - TabNet: Attentive Interpretable Tabular Learning
            - SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training
            
        ***
            
        **FraudTransformer Architecture (Right Figure)** 
        
        - Customized Transformer architecture, only keep multi-head encoder,
        - A MLP layer for classification task,
        - Embedding layer modified for fraud data.
                """
            )

    with col2:
        with st.container():
            image = Image.open('data/FraudTransformer.png')
            st.image(image,width=300)

    st.markdown("---")

    # st.subheader("")




    # if picked_data:
    #     st.subheader("**Fraud Data Sample**")
    #     X_test_df_picked_frd = X_test_df[X_test_df['frd'] == 1].sample().T
    #     st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)
    # else:
    #     st.subheader("**Fraud Data Sample**")
    #     X_test_df_picked_frd = X_test_df[X_test_df['frd'] == 1].sample().T
    #     st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)




    col1, col2 = st.columns(2)

    with col1:
        with st.container():
            if picked_data:
                st.subheader("**Fraud Data Sample**")
                chosen_frddf = X_test_df[X_test_df['frd'] == 1]
                st.session_state.load_state = random.randint(chosen_frddf.shape[0])
                X_test_df_picked_frd = chosen_frddf.sample(random_state=st.session_state.load_state).T
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)
            elif st.session_state.load_state:
                st.subheader("**Fraud Data Sample**")
                chosen_frddf = X_test_df[X_test_df['frd'] == 1]
                X_test_df_picked_frd = chosen_frddf.sample(random_state=st.session_state.load_state).T
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)
            else:
                st.subheader("**Fraud Data Sample**")
                chosen_frddf = X_test_df[X_test_df['frd'] == 1]
                X_test_df_picked_frd = chosen_frddf.sample(random_state=st.session_state.load_state).T
                st.dataframe(X_test_df_picked_frd.style.highlight_max(axis=0), height=1400)  # Same as st.write(df)

    with col2:
        with st.container():
            st.subheader("**Model Structure**")
            code = '''TabTransformer(
  (embedding): FeaturesEmbedding(
    (embedding): Embedding(90, 32)
  )
  (transformer): TabTransformerEncoder(
    (transformer): Sequential(
      (0): TabTransformerEncoderBlock(
        (attention): MultiHeadAttention(
          (q_w): Linear(in_features=32, out_features=32, bias=False)
          (k_w): Linear(in_features=32, out_features=32, bias=False)
          (v_w): Linear(in_features=32, out_features=32, bias=False)
          (fc): Linear(in_features=32, out_features=32, bias=False)
          (attention): ScaleDotProductAttention(
            (dropout): Dropout(p=0.5, inplace=False)
          )
        )
        (ffn): FeedForward(
          (net): Sequential(
            (0): Linear(in_features=32, out_features=128, bias=True)
            (1): GEGLU()
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=64, out_features=32, bias=True)
          )
        )
        (add_norm1): AddNormConnection(
          (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (add_norm2): AddNormConnection(
          (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
      (1): TabTransformerEncoderBlock(
        (attention): MultiHeadAttention(
          (q_w): Linear(in_features=32, out_features=32, bias=False)
          (k_w): Linear(in_features=32, out_features=32, bias=False)
          (v_w): Linear(in_features=32, out_features=32, bias=False)
          (fc): Linear(in_features=32, out_features=32, bias=False)
          (attention): ScaleDotProductAttention(
            (dropout): Dropout(p=0.5, inplace=False)
          )
        )
        (ffn): FeedForward(
          (net): Sequential(
            (0): Linear(in_features=32, out_features=128, bias=True)
            (1): GEGLU()
            (2): Dropout(p=0.5, inplace=False)
            (3): Linear(in_features=64, out_features=32, bias=True)
          )
        )
        (add_norm1): AddNormConnection(
          (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
        (add_norm2): AddNormConnection(
          (layer_norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.5, inplace=False)
        )
      )
    )
  )
  (mlp): MultiLayerPerceptron(
    (mlp): Sequential(
      (0): Linear(in_features=1440, out_features=16, bias=True)
      (1): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Dropout(p=0.5, inplace=False)
      (4): Linear(in_features=16, out_features=16, bias=True)
      (5): BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (6): ReLU()
      (7): Dropout(p=0.5, inplace=False)
      (8): Linear(in_features=16, out_features=1, bias=True)
    )
  )
)'''
            st.code(code, language='python')

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        with st.container():

            st.subheader("**Model Performance**")

            if data_model_loaded:

                ### load data
                st.write('Test Dataset Loading!')
                my_bar = st.progress(0)
                for percent_complete in range(1):
                    test_dl = prepare_input(X_test,y_test)
                    X_test_info = pd.DataFrame(X_test)
                    X_test_info['frd'] = y_test
                    my_bar.progress(percent_complete + 99)

                st.write('Test Dataset Loaded!')

                code = '''
Test Data Details:
                15saug00 0     5241
                         1     2252
                16xsqggn 0    20855
                         1     2208
                4rvrfbxt 0    23509
                         1     1724
                5h8i3ud8 0    21828
                         1     1846
                5mem3i7z 0     6737
                         1     2409
                60tfck6a 0    12845
                         1     2811
                89oebq5k 0    14070
                         1     1780
                8s1rqgxh 0    21574
                         1     2386
                i8n5h0pw 0    22936
                         1     2660
                qn4omaj3 0    14400
                         1     1370
                '''
                st.code(code, language='python')


                ### load model
                st.write('Trained Model Loading!')
                my_bar = st.progress(0)
                for percent_complete in range(1):
                    trained_model = prepare_model('checkpoints/Attention_Test.pt')
                    my_bar.progress(percent_complete + 99)
                st.write('Trained Model Loaded!')

            if model_run:

                ### model inference
                st.write('Model Inference start!')
                my_bar = st.progress(0)
                for percent_complete in range(1):
                    pred,pred_tensor = predict_prob(trained_model, test_dl, device=device)
                    my_bar.progress(percent_complete + 99)
                st.write('Model Inference Finished!')

                pred_labels = []
                for i in pred_tensor:
                    pred_label = torch.round(i).cpu().numpy()
                    pred_labels.append(pred_label)

                pred_labels = [round(a.squeeze().tolist()) for a in pred_labels]



            if Confusion_matrix_plot:

                conf_matrix = confusion_matrix(y_test, pred_labels)
                #
                # Print the confusion matrix using Matplotlib
                #
                fig, ax = plt.subplots(figsize=(7.5, 7.5))
                ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
                for i in range(conf_matrix.shape[0]):
                    for j in range(conf_matrix.shape[1]):
                        ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

                plt.xlabel('Predictions', fontsize=18)
                plt.ylabel('Actuals', fontsize=18)
                plt.title('Confusion Matrix', fontsize=18)
                st.pyplot(fig)

                # ### load model
                #
                #
                #

                # eval with loaded model

                # st.write('Test Dataset Loaded!')
                # st.write('Trained Model Loaded!')
                # st.write('Model Inference Finished!')
                # st.write(f"AUC of loaded model's prediction is: {roc_auc_score(y_test, preds)}")














    # st.sidebar.subheader("LSTM with Self Attention")
    # st.sidebar.markdown("---")
    # # model_structure = st.sidebar.button("Show Model Structure")
    #
    # st.header('Crop Classification Demo')
    # st.subheader("Upload Crop NDVI Data")
    # k = st.number_input("Maximum No. of Rows to Read", min_value=10,
    #                     max_value=1000, step=1, value=10, key='readinput')
    # results = None
    # uploaded_file = st.file_uploader(
    #     "Choose a CSV file (Maximum 1000 Rows for Performance)", type="csv", key='test')
    # st.subheader("Upload Ground Truth Label (Only for Testing)")
    # ground_truth_file = st.file_uploader(
    #     "Choose a CSV file (Should Match the NDVI CSV)", type="csv", key='truth')
    # if uploaded_file is not None:
    #     data = pd.read_csv(uploaded_file, nrows=k)
    #     st.write(data)
    #     st.subheader("Curve Visualization")
    #     st.line_chart(data.T.to_numpy())
    #     max_row = data.shape[0] - 1
    #     st.subheader("Plot single NDVI curve")
    #     ndvi_nrow = st.number_input(
    #         "Pick up a row", min_value=0, max_value=max_row, step=1, value=0, key='singleinput')
    #     picked_ndvi = data.iloc[ndvi_nrow]
    #     show_ndvi = st.button("Show single NDVI Curve")
    #     if show_ndvi:
    #         play_line_plots(picked_ndvi)
    #     st.subheader("Crop Classification")
    #     run_model = st.button("Run ML model")
    #     if run_model:
    #         with st.spinner('Model Running, Input Curve Row No.{}'.format(ndvi_nrow)):
    #             picked_input = data.iloc[[ndvi_nrow]]
    #             # scaler_info = read_scaler('./standard_scaler.npy')
    #             model_input = prepare_input(picked_input)
    #             model_instance = prepare_model()
    #             probabilities, pred_classes, pred_labels, results, attentions = execute_model(
    #                 model_instance, model_input)
    #             st.success('Finish. Input Curve Row No.{}'.format(ndvi_nrow))
    #     st.subheader("Results")
    #     if results is not None:
    #         st.write(results.style.highlight_max(axis=1))
    #         play_bar_plots(attentions)
    #         if ground_truth_file is not None:
    #             data = pd.read_csv(ground_truth_file)
    #             check_prediction(pred_classes, pred_labels, ndvi_nrow, data)