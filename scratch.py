# parameters = {'alpha':(0.025, 0.005), 'sg':(0, 1), 'negative':(2, 5)}
# # test model for w2vec hyper parameter tuning romance_10_10_S
# base_model = "drive/MyDrive/modelsOP"
# if not os.path.exists(base_model):
#   os.mkdir(base_model)
# win, vs = 3, 10
# corpus = brown.sents(categories="romance")
# corpus = [sent for sent in corpus if len(sent)>win] #[w.lower() for w in sent]
# print(f"Using {len(corpus)} for training with window size {win}")
# model = Word2Vec(alpha= 0.005, negative=2,sg=1, sentences=corpus, size=vs, window=win, min_count=5, workers=-1, seed=99, iter=5000)
# model.save(f"{base_model}/{cat}_w2vec_{win}_{vs}.model")
# top_n_dfWvec[f'{cat}_{win}_{vs}'], top_n_dfWvec[f'{cat}_{win}_{vs}_S'] = zip(*top_n_dfWvec['word1'].apply(topNw2vec))

# get_best_Wvec(top_n_dfWvec, models=[f'romance_{win}_{vs}_S']) # alpha 0.005 negative2 sg1
# get_best_Wvec(top_n_dfWvec, models=['romance_10_10_S']) # alpha 0.005 negative2
# get_best_Wvec(top_n_dfWvec, models=['romance_10_10_S']) # alpha 0.005