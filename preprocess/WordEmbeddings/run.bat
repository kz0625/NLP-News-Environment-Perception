@echo off
rem ===================== Configuration =====================
rem 'Chinese' or 'English'
set dataset_name=English

rem 'post' or 'article'
set data_type=post

rem Config it as your local word-embeddings filepath
set embedding_file=data\cenjingyang\word2vec\glove.840B.300d.txt

rem ===================== Tokenization by Word Embeddings =====================
python get_words_%dataset_name%.py --dataset %dataset_name% --data_type %data_type% --embedding_file %embedding_file%

pause
