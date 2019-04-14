import os
from preprocessing.Feature.TaskFeature import BERT_MRPC_FeatureCreator

train_topics = list(range(2, 4))
dev_topics = [12, 18, 21, 23, 34, 35]
cross_document = True
positive_increase = 4
shuffle = True
BERT_feature = BERT_MRPC_FeatureCreator()
csv_dir = "./feature/3-tpc/"

for task, topics in (("train", train_topics), ("dev", dev_topics)):
    if not os.path.exists(csv_dir):
        os.mkdir(csv_dir)
    csv_path = csv_dir + "%s.tsv" % task
    BERT_feature.feature_csv(
        csv_path=csv_path, topics=topics, cross_document=cross_document,
        positive_increase=positive_increase, shuffle=shuffle)
