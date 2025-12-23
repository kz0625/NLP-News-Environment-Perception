@echo off
rem ===================== Configuration =====================
rem 'Chinese' or 'English'
set dataset_name=English

rem ===================== Obtain the representations of posts and news =====================
set CUDA_VISIBLE_DEVICES=0
python get_repr.py --dataset %dataset_name%

pause
