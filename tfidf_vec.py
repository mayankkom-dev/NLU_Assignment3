from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import nltk
nltk.download(['brown'])
from nltk.corpus import brown
import pickle

get_corpus = lambda x: brown.sents(categories=[f"{x}"])

print_length = lambda x: print(f'Total number of sentences {len(x)}')

def fitBrown_tfidfvect(corpus_cat):
  brown_cat = get_corpus(corpus_cat)
  vectorizer = TfidfVectorizer(min_df=5, strip_accents=True, preprocessor=lambda x: re.sub(r'[^\w\s\t\d]', '', ' '.join(x).lower()), stop_words='english')
  vect_cat = vectorizer.fit_transform(brown_cat)
  return vectorizer, vect_cat  

def find_topN(word, vectorizer, X_transf):
  v = vectorizer.get_feature_names_out()
  if word in v:
    idx = np.where(v==word)[0]
    cosine_similarities = linear_kernel(X_transf[:, idx].T, X_transf.T).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-11:-1][1:]
    return ', '.join(v[related_docs_indices]), list(zip(v[related_docs_indices], cosine_similarities[related_docs_indices]))
  return '', ''

def topN_TFiDF(top_n_df, vect, fitvect, suffix):
  top_n_df[f'topN-TF-{suffix}'], top_n_df[f'topN-TF-{suffix}-S'] = zip(*top_n_df['word1'].apply(find_topN, args = (vect,fitvect)))  
  return top_n_df
