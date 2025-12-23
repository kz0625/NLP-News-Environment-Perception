@echo off
rem ===================== Configuration =====================
rem 'Chinese' or 'English'
set dataset_name=English
set events_num=300

rem ===================== Event Clustering by Kmeans =====================
python event_clustering.py --dataset %dataset_name% --events_num %events_num%
pause
