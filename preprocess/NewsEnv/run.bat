@echo off
rem ===================== Configuration =====================
rem set dataset_name=Chinese
set dataset_name=English
set macro_env_days=3

rem ===================== Get the macro env and rank its internal items by similarities =====================
python get_env.py --dataset %dataset_name% --macro_env_days %macro_env_days%
