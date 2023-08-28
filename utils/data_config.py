def input_config(flagship_model):
    config = {"US_banking_login_test": {
        "org_list": ['5h8i3ud8', '4rvrfbxt', '60tfck6a', '8s1rqgxh', 'i8n5h0pw', '16xsqggn', '15saug00', 'qn4omaj3',
                     '89oebq5k', '5mem3i7z'],
        "nonfrd_ratio": [0.0002, 0.00012, 0.0013, 0.0006, 0.00006, 0.032, 0.0035, 0.00011, 0.000056, 0.0049],
        "frd_ratio": [0.08, 0.13, 0.61, 0.65, 0.88, 1, 1, 1, 1, 1],
        "date_range": [(20201201, 20210531)] * 10,
        "month_range": [(202012, 202105)] * 10,
        "event_types": ["login"] * 10,
        "primary_industry": ["banking"] * 10,
        "exclude_dates": [[20201210, 20210407]] + [[x for x in range(20201206, 20201215)] + [x for x in range(20210119, 20210131)] + [x for x in range(20210201, 20210213)] + [20210401]] + [[]] + [[x for x in range(20201205, 20201231)] + [x for x in range(20210101, 20210103)] + [x for x in range(20210320, 20210328)]] + [[20210416]] + [[20201231, 20210101, 20210419,20210528]] + [[]] + [[20201225,20201231,20210418,20210424,20210502,20210530]] + [[]] + [[x for x in range(20210107, 20210131)] + [x for x in range(20210201, 20210221)] + [20210301]],
        "exclude_months": [()]*10,
        "output_folder": "/data/zhanyi02/testing2/"},

        "Insurance_data_test": {
            "org_list": ['1r6sl4qo', '4y498eek', 'qujzweur', '5vqcv6vz', '11b6fun8', '80zkfs4u', '9pgs8tle', '8c4gjo5j',
                         '3zq73xrf', 'cow401me'],
            "nonfrd_ratio": [0.0002, 0.00012, 0.0013, 0.0006, 0.00006, 0.032, 0.0035, 0.00011, 0.000056, 0.0049],
            "frd_ratio": [0.08, 0.13, 0.61, 0.65, 0.88, 1, 1, 1, 1, 1],
            "date_range": [(20210101, 20210831)] * 10,
            "month_range": [(202101, 202108)] * 10,
            "event_types": ["login"] * 10,
            "primary_industry": ["banking"] * 10,
            "exclude_dates": []*10,
            "exclude_months": [()] * 10,
            "output_folder": "/data/zhanyi02/testing3/"},

        "US_ecommerce_login_test": {"org_list": ["0oammhpj", "5m17vtp6"],
                                    "nonfrd_ratio": [0.006, 0.001],
                                    "frd_ratio": [1, 1],
                                    "date_range": [(20210101, 20210328)] * 2,
                                    "month_range": [(202101, 202103)] * 2,
                                    "event_types": ["login"] * 2,
                                    "primary_industry": ["ecommerce"] * 2,
                                    "exclude_dates": [(20210201, 20210202), ()],
                                    "exclude_months": [(202102,), ()],
                                    "output_folder": "/data/Ping_data/testing/"},

        "US_ecommerce_payment": {
            "org_list": ["2gfnuum5", "yd92qw6q"],
            "nonfrd_ratio": [0.0029, 0.0062, 0.0375, 0.0102, 0.0137, 0.0477, 0.2226],
            "frd_ratio": [0.0279, 0.3348, 0.4755, 0.5353, 1.0000, 1.0000, 1.0000],
            "date_range": [(20200901, 20200902)] * 7,
            "month_range": [(202009, 202009)] * 7,
            "event_types": ["payment"] * 7,
            "primary_industry": ["ecommerce"] * 3 + ["fintech"] + ["ecommerce"] * 5,
            "exclude_dates": [()] * 7,
            "output_folder": "/data/Ping_data/test/"},

        "EU_ecommerce_payment_dqr": {
            "org_list": ['682l2fp6', 'nd1lmtff', 'coombeus', 'dm9utgtm', 'dbrnb56e', '5176c3mo', '363t8kgq',
                         '9x6slmeq', 'cx8xec1a', '5a2z3x6m', 'bjf37sdb'],
            "nonfrd_ratio": [0.006, 0.001] * 11,
            "frd_ratio": [1, 1] * 11,
            "date_range": [(20201001, 20210331)] * 11,
            "month_range": [(202010, 202103)] * 11,
            "event_types": ["payment"] * 11,
            "primary_industry": ["ecommerce"] * 11,
            "exclude_dates": [()] * 11,
            "output_folder": "/data/Ping_data/EU_ecommerce_payment/"},

        "EU_ecommerce_payment": {
            "org_list": ['dbrnb56e', '363t8kgq', 'coombeus', '5176c3mo', '9x6slmeq'],
            "nonfrd_ratio": [0.005, 0.01, 0.02, 1, 0.003],
            "frd_ratio": [1] * 5,
            "date_range": [(20201001, 20210331)] * 5,
            "month_range": [(202010, 202103)] * 5,
            "event_types": ["payment"] * 5,
            "primary_industry": ["ecommerce"] * 5,
            "exclude_dates": [()] * 3 + [(20210128, 20210210)] + [
                (20210327, 20210328, 20210329, 20210330, 20210331)],
            "output_folder": "/data/Ping_data/EU_ecommerce_payment/"},

        "US_banking_payment_dqr": {
            "org_list": ['5h8i3ud8', '2jtgv0aw', '1urfsei4', '4rvrfbxt', 'b37tmm2a', '16xsqggn', '5pwln7fl',
                         'c7f45hl1', '49jnml1b'],
            "nonfrd_ratio": [0.006, 0.001] * 9,
            "frd_ratio": [1, 1] * 9,
            "date_range": [(20201001, 20210331)] * 9,
            "month_range": [(202010, 202103)] * 9,
            "event_types": ["payment"] * 11,
            "primary_industry": ["banking"] * 9,
            "exclude_dates": [(20201201, 20201202)] * 9,
            "output_folder": "/data/zhanyi02/US_banking_payment/"},

        "TrustScore_2022": {
            "org_list": ['nd1lmtff','551fvs6v','5pbbq33n','c7f45hl1','m5o81ypt','vn0xwfzw','9ghwyvdk','hkekmym4','bzzzajtm','23o27znc','n3dmfcgy','4rvrfbxt','0oammhpj','nhs11h5x','551fvs6v','hgy2n0ks','0w57c49k','ummqowa2','2gfnuum5','dbrnb56e','yfy4aqbg','wfw5lfw8','4rvrfbxt','9ghwyvdk','coombeus'],
            "nonfrd_ratio": [0.05964, 0.1364, 0.03512, 0.08405, 0.01233, 0.08186, 0.04309, 0.9004, 0.05164, 0.00104, 0.0002, 0.00037, 0.00863, 0.00247, 0.00119, 0.00214, 0.02707, 0.01468, 0.02559, 0.00841, 0.03217, 0.01005, 0.00065, 0.00856, 0.02292, 0.05964, 0.1364, 0.03512, 0.08405, 0.01233, 0.08186, 0.04309, 0.9004, 0.05164, 0.00104, 0.0002, 0.00037, 0.00863, 0.00247, 0.00119, 0.00214, 0.02707, 0.01468, 0.02559, 0.00841, 0.03217, 0.01005, 0.00065, 0.00856, 0.02292],
            "frd_ratio": [0.14,1,0.4,0.57,0.15,1,0.34,1,1,0.41,0.57,1,1,1,0.71,0.11,0.16,0.89,0.28,1,0.96,1,1,1,1,1],
            "date_range": [(20220601, 20221231)] * 25,
            "month_range": [(202206, 202212)] * 25,
            "event_types": ["payment"] * 25,
            "primary_industry": ["banking"] * 25,
            "exclude_dates": [(20201201, 20201202)] * 25,
            "output_folder": "/data/zhanyi02/trustscore/trustscore2023/combined/trust_score_2023_merge.csv"},

    }

    if flagship_model in config:
        return config[flagship_model]
    else:
        print()
        print(" -- no data config --")
        print()
