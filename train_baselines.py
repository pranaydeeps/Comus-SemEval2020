import lightgbm as lgb
from sklearn.metrics import *
from sklearn.datasets import load_svmlight_file


param = {'objective': 'multiclassova', 'num_class' : 3, 'metric': 'multi_error', 'verbose' : 2, 'is_unbalance' : True, 'num_leaves' : 45}

train_file = 'LexiconFeatures/data_train.txt'
train_lgb = lgb.Dataset(train_file)
test_file = 'LexiconFeatures/data_test.txt'
test_lgb = lgb.Dataset(test_file, reference=train_lgb)

qq_lgb = lgb.train(param,train_lgb, 10, feature_name=['f' + str(i + 1) for i in range(171597)])
preds = qq_lgb.predict('LexiconFeatures/data_test.txt').argmax(axis=1)


testdata = load_svmlight_file('LexiconFeatures/data_test.txt')
y_true = testdata[1]
print(classification_report(y_true, preds))
print(accuracy_score(y_true, preds))
print(confusion_matrix(y_true, preds))
