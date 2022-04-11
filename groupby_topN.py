import pandas as pd
import matplotlib.pyplot as plt

def get_topNDf(df):
  df_group = df.groupby('word1')
  
  def get_topn(word):
    if word in df_group.groups:
      return ", ".join(df_group.get_group(word)['word2'].tolist())
    return None
  top_n_df = pd.DataFrame(df_group["word2"].count())
  top_n_df = top_n_df.rename(columns={'word2':'topN'})
  top_n_df.reset_index(inplace=True)
  top_n_df['topNW'] = top_n_df['word1'].apply(get_topn) 
  print(f"Total {top_n_df.shape[0]} unique word for our Golden Standard")
  return top_n_df

def pretty_plotHist(df, col, bins, title, xlabel, ylabel):
  df[col].hist(bins=bins)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()

def transitivityExp(top_n_df, df):
  df_group = df.groupby('word1')
  def expand_topN(row):
    word1, topN, topNW = list(row.to_dict().values())
    # print(word1, topN, topNW)
    syn_data = []
    
    def get_topn(word):
      if word in df_group.groups:
        return ", ".join(df_group.get_group(word)['word2'].tolist())
      return None
    
    if topN<10:
      topNW = topNW.split(',')
      for w in topNW:
        # print(w)
        n_w = get_topn(w.strip())
        if n_w:
          if n_w!=w:
            
            n_w = n_w.split(',')#[0]
            # print(type(n_w), n_w)
            # syn_data.append(n_w)
            syn_data.extend(n_w)
      
      
      t_ = 10-topN
      en = t_ if t_<len(syn_data) else len(syn_data)
      topNW.extend(syn_data[:en])
      topN = len(topNW)
      return topN, ", ".join(topNW)
    return topN, topNW

  top_n_df['topN_'], top_n_df['topNW_']  = zip(*top_n_df.apply(expand_topN, axis=1))
  return top_n_df