@echo off
:: ===================== Configuration =====================
set dataset_name=English
set pretrained_model=bert-base-english

:: ===================== Tokenization by BERT =====================
python get_bert_tokens.py --dataset %dataset_name% --pretrained_model %pretrained_model%
pause
