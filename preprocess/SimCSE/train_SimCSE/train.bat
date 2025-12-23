@echo off

REM Train Chinese data
REM echo Training on Chinese data...
REM set CUDA_VISIBLE_DEVICES=0
REM python train.py ^
REM    --train_file ./data/Chinese_train.txt ^
REM    --max_length 256 ^
REM    --pretrained bert-base-chinese ^
REM    --learning_rate 5e-6 ^
REM    --save_final True ^
REM    --tau 0.05

REM Train English data
echo Training on English data...
set CUDA_VISIBLE_DEVICES=0
python train.py ^
    --train_file ./data/English_train.txt ^
    --max_length 128 ^
    --pretrained bert-base-uncased ^
    --learning_rate 5e-6 ^
    --dropout_rate 0.1 ^
    --save_final True ^
    --tau 0.05

REM Train English data with data fraction
echo Training on English data with data fraction...
set CUDA_VISIBLE_DEVICES=0
python train.py ^
    --train_file ./data/English_train.txt ^
    --max_length 128 ^
    --pretrained bert-base-uncased ^
    --learning_rate 5e-6 ^
    --dropout_rate 0.1 ^
    --save_final True ^
    --tau 0.05 ^
    --PTM_root_path ./PretrainedLM ^
    --data_fraction 0.1

echo All training tasks are complete.
