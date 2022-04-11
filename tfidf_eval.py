import operator
import pytrec_eval
import numpy as np

def get_best_vec(df, models=None):
  if not models: models = [col for col in df.columns if len(col.split('_'))==4 ]
  score = {}
  qrel = {}
  run_all = {}
  for ix, (idx, row) in enumerate(df.iterrows()):
    # print(row['topNW_'])
    qrel[f'q{idx}'] = {w:1 for i, w in enumerate(row['topNW_'].split(','))}
    for m in models:
      # print(m)
      if m not in run_all: run_all[m] = {}
      
      if row[m]:
        # print(row[m], type(row[m]))
        run_all[m][f'q{idx}'] = {k:v for k, v in row[m]}
      else:
        run_all[m][f'q{idx}'] = {}

  evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg'})
  for m in models:
    ndcg_l = np.array([v_dic['ndcg'] for q, v_dic in evaluator.evaluate(run_all[m]).items() ])
    # print(m, ndcg_l)
    score[f'{m}'] = ndcg_l.mean()
  best_m = max(score.items(), key=operator.itemgetter(1))[0]
  return best_m, score

import matplotlib.pyplot as plt

def plot_avgnDCG(score):
  name, sc = zip(*score.items())
  name = ['-'.join(n.split('-')[1:]) if '-' in n else n for n in name]
  plt.bar(name, sc)
  plt.xlabel("TFIDF model for corpus")
  plt.xticks(rotation=90)
  plt.ylabel("avg nDCG")
  plt.title(f"Vectorizer performance on {','.join(name[:4])} corpus")
  plt.show()