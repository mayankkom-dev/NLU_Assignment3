from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import multiprocessing
import os, pickle
import nltk
nltk.download(['brown'])
from nltk.corpus import brown
import matplotlib.pyplot as plt

def train_generateTopn(top_n_dfWvec, brown_cat=['news', 'romance'], wins=[1, 2, 5, 10], vss=[10, 50, 100, 300], base_model="models"):
  
  def topNw2vec(word):
    if model.wv.__contains__(word):
      w_s = model.wv.most_similar(word)
      return ', '.join([w for w, s in w_s]), w_s
    return '', ''
  
  
  pretrainF = True
  if not os.path.exists(base_model):
    os.mkdir(base_model)
    pretrainF = False
  
  sent_stats = {}
  for cat in brown_cat:
    corpus = brown.sents(categories=cat)
    for win in wins:
      for vs in vss:
        
        if pretrainF: 
          model = Word2Vec.load(f"{base_model}/{cat}_w2vec_{win}_{vs}.model")
          try:
            with open(f"{base_model}/sent_stats.pkl", "rb") as fp:
              sent_stats = pickle.load(fp)
          except:
            sent_stats = {1: 4588, 2: 4504, 5: 4253, 10: 3759}
        else:
          corpus = [sent for sent in corpus if len(sent)>win] #[w.lower() for w in sent]
          print(f"Using {len(corpus)} for training with window size {win}")
          if win not in sent_stats: sent_stats[win]=len(corpus)
          model = Word2Vec(sentences=corpus, size=vs, window=win, min_count=5, workers=-1, seed=99, iter=5000)
          model.save(f"{base_model}/{cat}_w2vec_{win}_{vs}.model")
        
        
        top_n_dfWvec[f'{cat}_{win}_{vs}'], top_n_dfWvec[f'{cat}_{win}_{vs}_S'] = zip(*top_n_dfWvec['word1'].apply(topNw2vec))
  if not pretrainF:
    with open(f"{base_model}/sent_stats.pkl", "wb") as fp:
      pickle.dump(sent_stats, fp)
  return sent_stats, top_n_dfWvec

def plotBar_dict(stats, xlabel, ylabel, title):
  w_, l_ = zip(*stats.items())
  plt.bar(w_, l_)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title)
  plt.show()