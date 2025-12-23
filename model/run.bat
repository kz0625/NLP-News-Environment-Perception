@echo off

rem ===================== Configuration =====================
rem 'Chinese' or 'English'
set dataset_name=English
rem BiLSTM, BERT, EANN, DeClarE, MAC
set model=EANN
rem Using News Environment or not (set it as 'true' or 'false')
set use_news_env=true
rem true是推理，false是训练
set evaluate = false

rem ===================== Training and Inferring =====================
set CUDA_VISIBLE_DEVICES=0

python -u main.py ^
--dataset %dataset_name% --model %model% --use_news_env %use_news_env% ^
--lr 5e-4 --batch_size 32 --epochs 50 ^
--evaluate %evaluate% ^
--save ckpts\%dataset_name%\%model%_%use_news_env%

pause
