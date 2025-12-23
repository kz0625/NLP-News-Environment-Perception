# ===================== Configuration =====================
# 'Chinese' or 'English'
dataset_name='English'

# 'post' or 'article'
data_type='post'

# Config it as your local word-embeddings filepath
embedding_file='/data/cenjingyang/word2vec/glove.840B.300d.txt'

# ===================== Tokenization by Word Embeddings =====================
python get_words_${dataset_name}.py --dataset ${dataset_name} --data_type ${data_type} --embedding_file ${embedding_file}