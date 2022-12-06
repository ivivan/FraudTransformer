import numpy as np
import pandas as pd
import datetime as DT
from datetime import datetime


class FraudData(object):

    def __init__(self, data_source=None, variable_feature=False, eval_partition=5, by_time=True, bb=False,
                 bb_feature=[]):

        self.train, self.val, self.test = None, None, None
        self.eval_partition = eval_partition

        if data_source is not None:
            train, val, test = self.split_train_val_test_set(data_source, variable_feature=variable_feature, bb=bb,
                                                             by_time=by_time, var_name='new_eval',
                                                             bb_feature=bb_feature)
            print(
                "NF/F rate in train:", round(float(len(train[train["frd"] == 0])) / len(train[train["frd"] == 1]), 2))
            print("NF/F rate in val:", round(float(len(val[val["frd"] == 0])) / len(val[val["frd"] == 1]), 2))
            print("NF/F rate in test:", round(float(len(test[test["frd"] == 0])) / len(test[test["frd"] == 1]), 2))

            self.train, self.val, self.test = train, val, test
        else:
            print("No input data to build model, need input while training!")

    def create_eval_column(self, df, var_name="eval"):
        '''
        ***********************************************************************************
        Split the data to training and test datasets, if self.eval_partition = 5, take first 4/5 (in time) to training and last 1/5 as evaluation (test dataset)
        ***********************************************************************************

        * INPUT: df - pandas DataFrame
        * RETURN: pandas DataFrame with new `eval` columns
        '''

        df["datetime"] = df.datetime.apply(lambda x: datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S.%f'))

        frd_df = df[df.frd == 1]
        df[var_name] = True
        for org in np.unique(df.org_id):
            temp = frd_df[frd_df.org_id == org]
            max_time, min_time = temp.datetime.max(), temp.datetime.min()

            eval_period = float((max_time - min_time).days) / self.eval_partition
            eval_date_start = max_time - DT.timedelta(days=eval_period)

            df.loc[df.org_id == org, var_name] = df.datetime[df.org_id == org] > eval_date_start

        return df

    def split_train_val_test_set(self, data, variable_feature=False, bb=False, by_time=True, var_name='new_eval',
                                 bb_feature=[]):
        data = data.sample(frac=1)  ## shuffle dataframe
        data['frd'] = pd.to_numeric(data['frd'])

        # ########## add weighted column ##########
        # #########################################
        # print('add weighted column')
        # no_non_fraud = data.loc[data['frd'] == 0].shape[0]
        # no_fraud = data.loc[data['frd'] == 1].shape[0]
        # print(round(no_non_fraud / no_fraud))
        # data.loc[data['frd'] == 1, 'weights_column'] = round(no_non_fraud / no_fraud)
        # data.loc[data['frd'] == 0, 'weights_column'] = 1

        if by_time:
            print('split by time')
            if 'eval' not in data.columns:
                data = self.create_eval_column(data, var_name='eval')
            trainval = data[data["eval"] == False]
            ot_eval = data[data["eval"] == True]

        train = trainval.sample(frac=0.8, random_state=1)

        val = trainval.drop(train.index)

        self.predictors = [rc for rc in data.columns if "tmxrc" in rc or "sumrc" in rc]

        if variable_feature:
            self.predictors += [var for var in data.columns if "intvar" in var]
        if bb:
            self.predictors += bb_feature

        self.response = "frd"
        return train, val, ot_eval