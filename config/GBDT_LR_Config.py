
stack_gbdt_params = {
                'num_trees':60,
                'num_leaves': 32,
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': {'binary_logloss', 'auc'},
                'max_depth': 6,
                'min_data_in_leaf': 450,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.95,
                'bagging_freq': 5,
                'lambda_l1': 1,
                'lambda_l2': 0.001,  # 越小l2正则程度越高
                'min_gain_to_split': 0.2,
                'verbose': 5,
                'is_unbalance': True,
                'isSave':True
            }

stack_lr_params = {
                "feature_size": 1920,   # num_trees*num_leaves
                "field_size": 60,      # num_trees
                "continuous_field_size": 0,
                "category_field_size": 60, # num_trees
                "learning_rate":0.001,
                "l2_reg":0.0001,
                "optimizer":"Adam",
                "num_epochs":50,
                "batch_size":512
            }