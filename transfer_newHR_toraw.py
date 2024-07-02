
import matplotlib.pyplot as plt
import numpy as np
import os
import re

weedout_list = ['chao', 'ming', 'yongda']

'''setting'''
fix_alert_count = 0
recall_algo = False
flag_total = False
simulation = True
show_plot = True
use_BCG = 1 # 1: BCG, 0: ECG
resample_size = 30

data_dir = "DATA_driving_combine" #"Data_driving" 
output_dir = "Raw_modified"
#data_dir = "DATA_simulation"
subject_dirs = os.listdir(data_dir)

list_result = []
total_scores = []

if use_BCG == 1:
    print("train BCG as norHR")
else:
    print("train ECG as norHR")
print("resample size :" + str(resample_size))

for s_dir in subject_dirs:    
    if '.csv' in s_dir or 'total' in s_dir:
        continue  
    filepath = os.path.join(data_dir, s_dir)      
    if not os.path.exists(os.path.join(filepath, 'ECG')):
        continue
    if s_dir in weedout_list:
        continue
    print('process..  ' + s_dir)
    filepath = os.path.join(data_dir, s_dir)
    data_type_list = os.listdir(filepath)
    HRV_dir = os.path.join(filepath, 'HR_raw')
    RAW_dir = os.path.join(filepath, 'raw')

    HRV_file_list = os.listdir(HRV_dir) 
    for file in HRV_file_list:         
        HR = []
        status = []
        confidence_level = []
        with open(os.path.join(HRV_dir, file), 'r') as HRfile:
            for line in HRfile.readlines():
                line = line.split(',')
                HR.append(float(line[0]))
                status.append(float(line[1]))
                confidence_level.append(int(line[2]))
        
        Raw_file = file.replace('.csv', '.log')
        if not os.path.exists(os.path.join(RAW_dir, Raw_file)):
            continue
        copy_text_line = []
        with open(os.path.join(RAW_dir, Raw_file), 'r') as RAWfile:
            counter_samplerate = 0
            counter_second = 0
            for line in RAWfile.readlines():
                temp = line.split(',')
                if len(temp) > 2:                    
                    temp[5] = str(int(HR[counter_second]))
                    temp[7] = str(status[counter_second])
                    counter_samplerate += 1
                    if counter_samplerate == 64:
                        counter_samplerate = 0
                        counter_second += 1
                    if counter_second == len(HR):
                        break
                    temp = ','.join(temp)
                    copy_text_line.append(temp+ '\n')
                else:
                    copy_text_line.append(line)

        with open(os.path.join(output_dir, Raw_file), 'w') as outputfile:
            for line in copy_text_line:
                outputfile.write(line)
                



