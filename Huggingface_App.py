import streamlit as st

def intro():
    import streamlit as st

    # st.write("# Fraud Detection with Deep Learning")

    st.sidebar.success("Select a demo above.")

    # st.markdown(
    #     """This app uses BERT from the Hugging Face library library to detect frauds.
    # """
    # )

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
            st.image(image, width=300)

    st.markdown("---")


def single_demo():
    import numpy as np
    import pandas as pd
    import transformers
    from matplotlib import pyplot as plt
    import shap
    from transformers import BertForSequenceClassification
    from transformers import BertTokenizer
    import streamlit.components.v1 as components
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils import capture

    # Load Data

    data_path = './data/fe_banking_login_20201201_20210531.csv'
    feature_path = './data/modified_coef_final4th.csv'
    raw_data = pd.read_csv(data_path)
    feature_list = pd.read_csv(feature_path)
    features = feature_list["Unnamed: 0"].astype("str").tolist()
    selected_data = raw_data[features + ['frd', 'eval']]
    selected_data.rename({'tmxrc_Possible VPN or Tunnel': 'tmxrc_Possible_VPN_or_Tunnel'}, axis=1, inplace=True)
    features[29] = 'tmxrc_Possible_VPN_or_Tunnel'
    test_data = selected_data[selected_data['eval'] == True].iloc[:, 0:46]

    # Set Tokenizer
    vocab_file = "./checkpoints/vocab.txt"
    special_tokens_dict = {"unk_token": "[UNK]",
                           "sep_token": "[SEP]",
                           "pad_token": "[PAD]",
                           "cls_token": "[CLS]",
                           "mask_token": "[MASK]",
                           "eos_token": "[EOS]",
                           "bos_token": "[BOS]"}
    tokenizer = BertTokenizer(vocab_file, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.add_special_tokens(special_tokens_dict)

    # Load Model
    finetunemodel = BertForSequenceClassification.from_pretrained('./checkpoints/finetune/checkpoint-14040',
                                                                  num_labels=2)
    classifier = transformers.pipeline("sentiment-analysis", model=finetunemodel, tokenizer=tokenizer,
                                       return_all_scores=True)
    explainer = shap.Explainer(classifier)

    def st_plot_text_shap(shap_val, height=None):
        InteractiveShell().instance()
        with capture.capture_output() as cap:
            shap.plots.text(shap_val[:, :, "LABEL_1"])
        components.html(cap.outputs[0].data['text/html'], height=height, scrolling=True)

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')

    # Promt Input
    st.markdown("Now select a row from the test set")
    max_row = len(test_data) - 1
    index = st.number_input(
        "Pick up a row", min_value=0, max_value=max_row, step=1, value=0, key='singleinput')
    input_list = test_data.iloc[index]
    input_df = test_data.iloc[[index]]
    show_row = st.button("Show selected row")
    if show_row:
        st.text("Your input is:")
        st.dataframe(input_df)

    st.subheader("Fraud Classification")
    run_model = st.button("Run BERT model")
    if run_model:
        input_list = test_data.iloc[index]
        test_data_features = input_list[0:45]
        test_data_labels = input_list[45]
        x = test_data_features.to_string(header=False).split('\n')
        vals = ['_'.join(ele.split()) for ele in x]
        val = " ".join(vals)
        st.text('{}'.format(classifier(val)))
        st.text('{}'.format(input_list[45]))

        st.subheader("SHAP values plot")
        shap_values = explainer([val])
        # st_shap(shap.force_plot(explainer.expected_value, shap_values[:, :, "LABEL_1"], X.iloc[0,:]))
        fig_ttl = shap.plots.text(shap_values[:, :, "LABEL_1"])
        st.pyplot(fig_ttl)
        plt.clf()
        # st_plot_text_shap(shap_values)


def dataset_demo():
    import numpy as np
    import pandas as pd
    import transformers
    from transformers import BertForSequenceClassification
    from transformers import BertTokenizer
    from transformers import pipeline
    from datasets import Dataset
    import torch

    st.write('Now upload your own fraud detection dataset with, of course, the same format')
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type="csv")
    if uploaded_file is not None:
        run_model = st.button("Run ML model")
        if run_model:
            test_data = pd.read_csv(uploaded_file)
            feature_path = './data/modified_coef_final4th.csv'
            feature_list = pd.read_csv(feature_path)
            features = feature_list["Unnamed: 0"].astype("str").tolist()
            features[29] = 'tmxrc_Possible_VPN_or_Tunnel'

            vocab_file = "./checkpoints/vocab.txt"
            special_tokens_dict = {"unk_token": "[UNK]",
                                   "sep_token": "[SEP]",
                                   "pad_token": "[PAD]",
                                   "cls_token": "[CLS]",
                                   "mask_token": "[MASK]",
                                   "eos_token": "[EOS]",
                                   "bos_token": "[BOS]"}
            tokenizer = BertTokenizer(vocab_file, do_lower_case=False, do_basic_tokenize=False)
            tokenizer.add_special_tokens(special_tokens_dict)

            test_data_features = test_data.iloc[:, 0:45]
            test_data_labels = test_data.iloc[:, 45]
            vocab_dict = tokenizer.get_vocab()

            for column in test_data_features:
                test_data_features[column] = test_data_features[column].map(lambda x: column + "_" + str(x))
                test_data_features[column] = test_data_features[column].map(
                    lambda x: vocab_dict.get(x, vocab_dict['[UNK]']))
            input_initial = test_data_features.to_numpy().tolist()
            input = [[3] + x + [1] for x in input_initial]
            my_dict = {'input_ids': input,
                       'labels': test_data_labels.to_list(),
                       'token_type_ids': [[0] * (len(test_data_features.columns) + 2)] * len(test_data_features),
                       'attention_mask': [[1] * (len(test_data_features.columns) + 2)] * len(test_data_features)}
            test_df = Dataset.from_dict(my_dict)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            test_df = test_df.with_format("torch", device=device)
            finetunemodel = BertForSequenceClassification.from_pretrained(
                './checkpoints/finetune/checkpoint-14040', num_labels=2).to("cuda")
            finetunemodel.eval()
            from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
            val_data = TensorDataset(test_df['input_ids'], test_df['attention_mask'], test_df['labels'])
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=32)
            import torch.nn.functional as F
            def bert_predict(model, test_dataloader):
                """Perform a forward pass on the trained BERT model to predict probabilities
                on the test set.
                """
                # Put the model into the evaluation mode. The dropout layers are disabled during
                # the test time.
                model.eval()

                all_logits = []

                # For each batch in our test set...
                for batch in test_dataloader:
                    # Load batch to GPU
                    b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

                    # Compute logits
                    with torch.no_grad():
                        logits = model(b_input_ids, b_attn_mask)['logits']
                    all_logits.append(logits)

                # Concatenate logits from each batch
                all_logits = torch.cat(all_logits, dim=0)

                # Apply softmax to calculate probabilities
                probs = F.softmax(all_logits, dim=1).cpu().numpy()

                return probs

            probs = bert_predict(finetunemodel, val_dataloader)
            from sklearn.metrics import roc_auc_score
            import numpy as np
            y_true = np.array(test_data.iloc[:, 45])
            y_scores = probs[:, 1]
            st.text('{}'.format(roc_auc_score(y_true, y_scores)))


st.set_page_config(page_title="FraudTransformer Demo", layout='wide')
page_names_to_funcs = {
    "—": intro,
    "Single Demo": single_demo,
    "Dataset Demo": dataset_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()