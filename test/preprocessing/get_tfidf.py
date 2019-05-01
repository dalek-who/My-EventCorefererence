#%%
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from preprocessing.Structurize.EcbClass import *

#%%
PCA_DIM = 100

#%%
ecb = EcbPlusTopView()
#%%
docs_list = []
for topic in ecb.document_view.topics_dict.values():
    for doc in topic.documents_dict.values():
        docs_list.append(doc)
#%%
nlp = spacy.load("en")
# 词形还原，去除停用词
corpus = []
for doc in docs_list:
    sentences_list = []
    for sentence in doc.all_sentences_dict.values():
        sentences_list.append(sentence.text())
    doc_text = " ".join(sentences_list)
    doc_spacy = nlp(doc_text)
    lemma_list = []
    for token in doc_spacy:
        lemma = token.lemma_
        if not token.is_stop:
            lemma_list.append(lemma)
    corpus.append(" ".join(lemma_list))

#%%
# TF-IDF
tfidfvec = TfidfVectorizer()
cop_tfidf = tfidfvec.fit_transform(corpus)
tfidf_weight = cop_tfidf.toarray()
#%%
# PCA降维
pca = PCA(n_components=PCA_DIM)
tfidf_weight_compressed = pca.fit_transform(tfidf_weight)

#%%
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

clf = KMeans(n_clusters=len(ecb.document_view.topics_dict))  # 景区 动物 人物 国家
s = clf.fit(tfidf_weight)