# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 17:06:55 2020

@author: breez
"""


import HRV_algo.get_HRV as HRV
import PERCLOS_algo.get_PERCLOS as perclos
import numpy as np
import BCG_algo.python.src.read_data as rd
from BCG_algo.python.src.alg_freq_domain import Alg_freq_domain
import get_PVT as PVT
import os
import sys
import mariadb
import datetime 
import time

def convert(date_time): 
    format = '%Y-%m-%d-%H-%M' # The format 
    datetime_str = datetime.datetime.strptime(date_time, format) 
   
    return datetime_str 

def get_datetime(filename):
    info_list = (filename.split('.')[0]).split('_')
    for info in info_list:
        if '-' in info:
            return convert(info)
    

def get_filename_with_datetime(file_list, datetime):    
    for file in file_list:
        if datetime == get_datetime(file):
            return file
    return -1

def get_HRV_txt(text, is_PVT):
        
    ecg_list = []
    text = text.split('\n')

    #with open(os.path.join(ECG_dir, file), 'r') as m_input:
        #print('processing ', file)
    for line in text:
        if 'Time' in line :
            continue
        if 'State' in line :
            continue
        if ',' in line:
            bpm = line.split(',')[ecg_index]
        else: 
            if len(line.split('\t'))<5:
                continue
            bpm = line.split('\t')[ecg_index]

        ecg_list.append(float(bpm))
    HR, HR_t = HRV.ECG_to_HR(ecg_list, fs_ecg)
    if is_PVT:
        start = 60
    else:
        start = len(HR) - int(60 * window_size + 1)
    if start < 0 :
        start = 0
    bpm = HR[start: start + window_size*60]
    
    if len(bpm) < window_size*60:
        bpm = HR
    mean_HR, RMSSD, SDNN, VLF = HRV.Get_HRV(bpm)
    HRV_record = str(HRV.get_DF(bpm)) + ',' + str(mean_HR) + ',' + str(RMSSD) + ',' + str(SDNN) + ',' + str(VLF)  
    
    return HRV_record

def get_HRV_bcg_text(text, is_PVT, is_LHR):
    
    
    FFT_window_size=30
    sample_rate = 64
 
    pressure_data, acc_x, acc_y, acc_z, start_time = rd.read_pressure_acc_text(text)
    if USE_ACC_SUM:
        acc_y = (acc_x ** 2 + acc_y ** 2 + acc_z ** 2) ** 0.5
    elif USE_ACC_MAX:
        temp = np.array([acc_x, acc_y, acc_z])
        max_index = np.argmax(abs(np.mean(temp, axis=1)))
        acc_y = temp[max_index] * np.sign(np.mean(temp, axis=1)[max_index])
    
    algo = Alg_freq_domain(fs=sample_rate, fft_window_size=FFT_window_size)
    algo.USE_BDT = 0
    algo.USE_MANUALLY_WEIGHT = True
    
    algo.overlap_weight = overlap_weight
    if is_LHR:
        algo.overlap_weight = [1, 1, 1, 1, 1, 1, 1]        
    
    algo.main_func(pressure_data, acc_y)
    
    HR_list = algo.bpm
    
    HR = np.copy(HR_list)
    HR = HRV.moving_average(HR_list, 5)
    
    if is_PVT:
        start = 60
    else:
        start = len(HR) - int(60 * window_size + 1)
    if start < 0 :
        start = 0
    bpm = HR[start: start + window_size*60]
    
    if len(bpm) < window_size*60:
        bpm = HR
        
    mean_HR, RMSSD, SDNN, VLF = HRV.Get_HRV(bpm)
    HRV_record = str(HRV.get_DF(bpm)) + ',' + str(mean_HR) + ',' + str(RMSSD) + ',' + str(SDNN) + ',' + str(VLF)        
    
    return HRV_record 

def get_perclos_text(text, is_PVT, PERCLOS_threshold):
    text = text.split('\n')[:-1]
    ear_raw = []
    for line in text:
        line = "".join(filter(lambda ch: ch in '0123456789.', line))
        if line == '' :
            continue
        ear_raw.append(float(line))

    if PERCLOS_threshold ==0:        
        PERCLOS_threshold = perclos.get_threshold(ear_raw)

    ear = np.asarray(ear_raw)
    
    return str(perclos.compute_PERCLOS_overall(ear, PERCLOS_threshold)), str(perclos.compute_PERCLOS_localmax(ear, PERCLOS_threshold, FPS))

def get_PVT_text(text, filename):
    index_profile = 7
    text = text.split('\n')[:-1]
    data = np.zeros(len(data_head))
    raw_isi = [] 
    raw_rt = [] 
    raw_lor = [] 
    raw_cor = [] 
    data_start = False
    for  line in text:       
       
        if '.' in line:
            line = line.replace('.', ',')
        
        line_split = line.split(',')
        if data_start == False:
            for i in range(1, index_profile):
                if data_head[i] in line_split[0]:
                    if data[i] == 0:
                        data[i] = int(line_split[1])
                    else:
                        data[i] = (int(line_split[1])+data[i])/2
                    break
            if 'RT' in line:
                data_start = True
        else:
            raw_isi.append(int(line_split[1]))
            raw_lor.append(int(line_split[2]))
            raw_rt.append(int(line_split[3]))
            raw_cor.append(int(line_split[4]))
    
    # sleep_time = data[1]
    # wake_time = data[2]
    # sleep_2h = data[3]
    # HR = data[4]
    # SSS = data[5]
    # KSS = data[6]
    rt_used_mean, rt_used_sort_mean, rt_used_inv_mean, rt_used_lapes_len, rt_used_lapes_result, rt_corr_used_mean, rt_used_SNR = PVT.get_PVT_result_array(np.array(raw_isi), np.array(raw_lor), np.array(raw_rt), np.array(raw_cor))
    data[index_profile]   = rt_used_mean
    data[index_profile+1] = rt_used_sort_mean
    data[index_profile+2] = rt_used_inv_mean
    data[index_profile+3] = rt_used_lapes_len
    data[index_profile+4] = rt_used_lapes_result
    #data[index_profile+5] = rt_corr_used_mean
    #data[index_profile+6] = rt_used_SNR 
    consolidate_line = []        
    for i in range(len(data_head)):
        if i == 0:
            consolidate_line.append(filename)
        else:
            consolidate_line.append(str(data[i]))   
    return ','.join(map(str, consolidate_line))

### manual setting parameters
gener_all = True
where_subject = 'mads'


USE_ACC_SUM = False
USE_ACC_MAX = True

###constant
overlap_weight = [1, 1, 1, 1, 1, 0, 0]
ecg_index = 6
window_size = 5 # second
shift_size = 5 # second
fs_ecg = 100
PERCLOS_threshold = 0
FPS = 25
output_title = 'DF, HR, RMSSD, SDNN, VLF'
data_head = ['name', 'sleep_time', 'wake_time', 'sleep_2h', 'HR', 'SSS', 'KSS', 'RT', 'RT_fast', 'PVT_inverRT', 'lapse', 'lapse_percent']
    

### auto variables
list_main_id = []
list_subj_id = []
list_dattime = []
list_inofice = []

list_raw_data_pvt = []  
list_raw_data_DRV = []  


List_HRV_PVT = []
List_HRV_DRV = []    
List_HRV_bcg_PVT = []
List_HRV_bcg_DRV = []
List_PERCLOS_PVT = []
List_PERCLOS_DRV = []    
List_PERCLOS_max_PVT =[]
List_PERCLOS_max_DRV =[]
List_SDP = []
List_SDW = []
List_PVT = []
    

with open('LHR.txt', 'r', encoding='utf-8') as subject:
    LHR = (subject.read())
start_time = time.time()    
try:
    conn = mariadb.connect(
        user="root",
        password="biologue",
        host="127.0.0.1",
        port=3306,
        database="fatigue_demo"

    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)



if gener_all:
    cur = conn.cursor()    
    cur.execute("SELECT * FROM table_main", )
    items = [item for item in cur.fetchall()]
    list_main_id = [item[0] for item in items]
    list_subj_id = [item[2] for item in items]
    list_dattime = [item[3] for item in items]
    list_inofice = [item[5] for item in items]
else:
    #find subject
    cur = conn.cursor()    
    cur.execute("SELECT user_id FROM table_subject WHERE name LIKE %s", (where_subject,))
    list_subj_id_temp = [item[0] for item in cur.fetchall()]
    
    #find main id
    for subj_id in list_subj_id_temp:
        cur = conn.cursor()    
        cur.execute("SELECT  * FROM table_main WHERE user_id LIKE %s", (subj_id,))
        items = [item for item in cur.fetchall()]
        for item in items:
            list_main_id.append(item[0])
            list_subj_id.append(item[2])
            list_dattime.append(item[3])
            list_inofice.append(item[5])
            
for i in range(len(list_main_id)):
    print('subject id ' , list_subj_id[i], 'processing')
    PERCLOS_threshold = 0
    cur = conn.cursor()    
    cur.execute("SELECT  * FROM table_raw_pvt WHERE main_data_id LIKE %s", (list_main_id[i],))
    list_raw_data_pvt = [item for item in cur.fetchall()]
    
    cur = conn.cursor()    
    cur.execute("SELECT  * FROM table_raw_drive WHERE main_data_id LIKE %s", (list_main_id[i],))
    list_raw_data_drv = [item for item in cur.fetchall()]
    
# get info
    
    List_PVT = [get_PVT_text(item[3], list_subj_id[i]) for item in list_raw_data_pvt]
    
    for i in range(len(list_raw_data_drv) + len(list_raw_data_pvt)):
        if i % 2 == 1:
            index = int((i-1)/2)
            curr_ECG = list_raw_data_pvt[index][4]
            curr_BCG = list_raw_data_pvt[index][5]
            curr_EAR = list_raw_data_pvt[index][6]
        else:
            index = int((i-2)/2)
            curr_ECG = list_raw_data_drv[index][4]
            curr_BCG = list_raw_data_drv[index][5]
            curr_EAR = list_raw_data_drv[index][6]
            

    List_HRV_PVT = [get_HRV_txt(item[4], True ) for item in list_raw_data_pvt]
    List_HRV_DRV = [get_HRV_txt(item[4], False) for item in list_raw_data_drv]
 
    List_HRV_bcg_PVT = [get_HRV_bcg_text(item[5], True , False) for item in list_raw_data_pvt]
    List_HRV_bcg_DRV = [get_HRV_bcg_text(item[5], False, False) for item in list_raw_data_drv]
    
    List_PERCLOS_PVT = [get_perclos_text(item[6], True , PERCLOS_threshold) for item in list_raw_data_pvt]
    List_PERCLOS_DRV = [get_perclos_text(item[6], False, PERCLOS_threshold) for item in list_raw_data_drv]
    
    if int(list_inofice[i]) :   
        for item in list_raw_data_drv:
            text = item[3]
            text = text.split('\n')[:-1]
            raw_W = []
            raw_P = []

        
            for line in text:
                raw_W.append(int(line.split(',')[0]))
                raw_P.append(int(line.split(',')[1]))
            List_SDW.append(str(np.std(raw_W)))
            List_SDP.append(str(np.std(raw_P)))
    else:
        List_SDW = ['']*len(List_PERCLOS_DRV)
        List_SDP = ['']*len(List_PERCLOS_DRV)
            
            
# write file        
    filename = 'sub' + str(list_subj_id[i]) + '_' + list_dattime[i].strftime("%Y-%m-%d") + '_output.csv'
    filepath = os.path.join('DBresult', filename ) 
    PVT_title = ', '.join(map(str, data_head))
    with open(filepath, 'w', encoding='utf-8') as output:
        output.write(PVT_title + ', SDW, SDP, PERCLOS, PERCLOS_MAX,' + output_title + ',PERCLOS, PERCLOS_MAX,' + output_title + '\n')
        
        for i in range(len(List_HRV_PVT)): #len(List_HRV)):
            if i == 0:
                output.write(List_PVT[i] + ',,,' + str(List_PERCLOS_PVT[i][0]) + ',' + str(List_PERCLOS_PVT[i][1]) + ',' + List_HRV_PVT[i] + '\n')
            else:
                output.write(List_PVT[i] + ',' + List_SDW[i-1] + ',' + List_SDP[i-1] + ',' + str(List_PERCLOS_PVT[i][0]) + ',' + str(List_PERCLOS_PVT[i][1]) + ',' + List_HRV_PVT[i] + ',' + str(List_PERCLOS_DRV[i-1][0]) + ',' + str(List_PERCLOS_DRV[i-1][1]) + ',' + List_HRV_DRV[i - 1] + '\n')
                
        output.write(PVT_title + ',SDW, SDP,PERCLOS, PERCLOS_MAX,' + output_title  + ',PERCLOS, PERCLOS_MAX,' + output_title + '\n')
        for i in range(len(List_HRV_bcg_PVT)): #len(List_HRV)):
            if i == 0:
                output.write(List_PVT[i] + ',,,' + str(List_PERCLOS_PVT[i][0]) + ',' + str(List_PERCLOS_PVT[i][1]) + ',' + List_HRV_bcg_PVT[i] + '\n')
            else:
                output.write(List_PVT[i] + ',' + List_SDW[i-1] + ',' + List_SDP[i-1] + ',' + str(List_PERCLOS_PVT[i][0]) + ',' + str(List_PERCLOS_PVT[i][1]) + ',' + List_HRV_bcg_PVT[i] + ',' + str(List_PERCLOS_DRV[i-1][0]) + ',' + str(List_PERCLOS_DRV[i-1][1]) + ',' + List_HRV_bcg_DRV[i - 1] + '\n')
                
     
print("time elapsed: {:.2f}s".format(time.time() - start_time))
cur.close()
conn.close()