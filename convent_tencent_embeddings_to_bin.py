from gensim.models.keyedvectors import KeyedVectors


word2vec_file = "/home/lcw2/share/embeddings/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt"
model = KeyedVectors.load_word2vec_format(word2vec_file, binary=False)
model.save('./Tencent_AILab_ChineseEmbedding.bin')
