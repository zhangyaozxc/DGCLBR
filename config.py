CONFIG = {
    'name': '@4879',
    # '/root/autodl-tmp/data'
    'path': './data',
    'gpu_id': "0",
    'model': 'mymodel',
    'dataset_name': 'Youshu',
    'topk': [20, 40, 80],

    ## optimal hyperparameters
    # 3.0e-4
    'lrs': [3.0e-4],
    'message_dropouts': [0.3],
    'node_dropouts': [0],
    # 5.0e-6
    'l2_regs': [5.0e-6],  # the l2 regularization weight: lambda_2
    "c_lambdas": [0.04],  # the contrastive loss weight: lambda_1
    "c_temps": [0.20],  # the temperature in the contrastive loss: tau

    ## hard negative sample and further train
    'sample': 'simple',
    'hard_window': [0.7, 1.0],  # top 30%
    'hard_prob': [0.3, 0.3],  # probability 0.8
    'conti_train': 'log/Youshu/',

    ## other settings
    'epochs': 100,
    'early': 50,
    'log_interval': 20,
    'test_interval': 1,
    'retry': 1,

    ## test path
    'test': ['log/Youshu']
}

