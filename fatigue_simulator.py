# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:28:08 2021

@author: breez
"""


import src.get_HRV as HRV
import src.get_PERCLOS as perclos
import src.read_data as rd
import src.get_PVT as PVT
from src.alg_freq_domain import Alg_freq_domain
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import datetime
from scipy.interpolate import interp1d

def get_statis(List):
    buffer = []
    
    for i in range(len(List)):
        temp = List[i].split(',')
        buffer.append(np.array([float(temp[i]) for i in range(1, len(temp))]))
        #output.write(List_ecg_before[i] + '\n')
    buffer = np.array(buffer)  
    mean = np.average(buffer, axis = 0)
    std = np.std(buffer, axis = 0)
    return mean, std

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))
            


def datetime_convert(date_time): 
    format = '%Y-%m-%d-%H-%M' # The format 
    
    datetime_str = datetime.datetime.strptime(date_time, format) 
   
    return datetime_str 

def ar(alist):
    return np.array(alist)

def norm_HRlist_model(HR_list):
    HR_list_sorted = sorted(HR_list)
    norm_list = []
    for i in range(100):
        
        norm_list.append(HR_list_sorted[int(i * len(HR_list_sorted) / 100)])
    return np.array(norm_list)

def simulate_bysubject(list_person_FI, final_threshold, proper_alert_count, gd_threshold,  plot = False):
    list_delaytime = []
    list_falsealert = []
    count_miss_alert = 0
    count_fals_alert = 0
    list_all_alert = []
    if FI_type == 0:
        final_threshold = 100 - final_threshold 
    for i in range(len(list_person_FI)):
        if FI_type == 0:
            list_fatigue = list_person_FI[i] <= final_threshold
        else:
            list_fatigue = list_person_FI[i] >= final_threshold

        
        list_alert_stage = []
        list_alert = []
        count = 0
        correct_alert = 0
        falsealert = 0
        for j in range(len(list_person_gd[i])):
            if list_fatigue[j] == True:
                count += 1
                if count >= proper_alert_count:
                    list_alert_stage.append(1)
                    list_alert.append(1)
                    if correct_alert == 0 and list_person_gd[i][j] >= gd_threshold:
                        list_delaytime.append(list_person_gd[i][j] - 1)
                        correct_alert = 1
                    elif list_person_gd[i][j] < int(gd_threshold):
                        falsealert += 1
                else:
                    list_alert_stage.append(0)
                
            else:
                count = 0
                list_alert_stage.append(0)
        list_all_alert.append(list_alert_stage) 
        if plot:
            plt.figure()
            plt.title(List_subject[i] + '_simulation')
            plt.plot(list_person_gd[i], list_fatigue, 'o', color = 'black')       
            plt.plot(ar(list_person_gd[i])[ar(list_alert_stage) == 1], list_alert, 'o', color = 'red')
            plt.savefig(os.path.join(figure_save_path_fatigue, List_subject[i] + '_fatigue.png'), format='png')
            plt.ylim([0, 3])
            plt.close()
        if correct_alert == 0:
            count_miss_alert += 1
        list_falsealert.append(falsealert)
        count_fals_alert += 1 if falsealert>0 else 0
    return count_miss_alert, count_fals_alert





''''Alg_freq_domain parameters setting'''
USE_ACC_SUM = True
FFT_window_size=32 
 
'''setting'''

FLAG_RECALL_ALGO = True
FLAG_TRAIN = False
FLAG_SIMULATION = False
FLAG_SHOW_PLOT = False
FLAG_FILTER_STATUS = False # filter the unstable HR by the status

use_BCG = 1 # 1: BCG, 0: ECG
use_raw = 1 # 0: use conbimation data 1: use the raw data
use_time = 1 # 1: time as golden, 0: KSS as golden, 2:lapse as golden
drive_ornot = 1 # 1: driving 0: doing PVT 
FI_type = 0 # 0:norHR 1:PERLCOS 2:model

data_dir = "DATA_KSS_GD" #DATA_driving_combine" #"Data_driving" 
subject_dirs = os.listdir(data_dir)

figure_save_path_fatigue = os.path.join(data_dir, '0_fatigue')
HR_save_all_path = os.path.join(data_dir, 'HR_BCG_all')
fix_alert_count = 3

#constant=====================================
DATA_MODE = ['ECG', 'BCG']
FILE_MODE = ['.txt', '.log']
CRITERIA_SPCIFICITY = [0.01, 0.102, 0.22, 0.32, 0.4]
WEEDOUT_LIST = ['chao', 'ming', 'yongda'] # ['shu','eric', 'roxus', 'kim', 'gui', 'grace', 'larry' , 'yangfei', 'dong', 'jie', 'he', 'ming']
RESAMPLE_SIZE = 30

#buffer=====================================
List_ecg_before = []
List_ecg_after = []
    
List_bcg_before = []
List_bcg_after = []


List_subject = []
List_allHR_ecg = []
List_allHR_bcg = []
List_restingHR = []
List_err_model = []
List_rvalue = []

list_norHR = []
list_KSS = []
list_lapsep = []
list_time = []
list_time_PERCLOS = []
list_PERCLOS = []
list_name = []

list_PERCLOS_bytime = []
list_norHR_bytime = []
list_norHR_bcg_bytime = []
list_Lapsep_by_time = []
list_KSS_by_time = []
list_person_gd = []
list_person_FI = []
list_person_st = []

model_sdr_mean = []
model_sdr_std = []
model_coef = []

if not os.path.exists(figure_save_path_fatigue):
    os.mkdir(figure_save_path_fatigue)

with open('LHR.d', 'r', encoding='utf-8') as subject:
    LHR = (subject.read())

with open('model.csv', 'r', encoding='utf-8') as input:
    model_sdr_mean = input.readline().split(',')
    model_sdr_mean = ar([float(i) for i in model_sdr_mean])
    model_sdr_std = input.readline().split(',')
    model_sdr_std = ar([float(i) for i in model_sdr_std])
    model_coef = input.readline().split(',')
    model_coef = ar([float(i) for i in model_coef])


if FI_type == 1:
    print("train PERCLOS")
elif FI_type == 0:
    if use_BCG == 1:
        print("train BCG as norHR")
    else:
        print("train ECG as norHR")
elif FI_type ==2:
    print("try model")

print("resample size :" + str(RESAMPLE_SIZE))


for s_dir in subject_dirs:    
    if '.csv' in s_dir or 'total' in s_dir:
        continue  
    filepath = os.path.join(data_dir, s_dir)      
    if not os.path.exists(os.path.join(filepath, 'ECG')):
        continue
    if s_dir in WEEDOUT_LIST:
        continue

    filepath = os.path.join(data_dir, s_dir)
    output_title = 'DF, HR, RMSSD, SDNN, VLF, norHR'
    data_head = ['name', 'sleep_time', 'wake_time', 'sleep_2h', 'HR', 'SSS', 'KSS', 'RT', 'RT_fast', 'PVT_inverRT', 'lapse', 'lapse_percent']
    PVT_title = ', '.join(map(str, data_head))

    data_type_list = os.listdir(filepath)

    HRV_dir = os.path.join(filepath, DATA_MODE[use_BCG])
    if use_raw: 
        HRV_dir = os.path.join(filepath, 'raw')
    PVT_dir = os.path.join(filepath, 'PVT')
    EAR_dir = os.path.join(filepath, 'EAR')
    
    
    HRV_file_list = os.listdir(HRV_dir)
    pvt_file_list = os.listdir(PVT_dir)
    EAR_file_list = os.listdir(EAR_dir)    
    
    '''parameters'''
    
    ecg_index = 6
    window_size_min = 5
    fs_ecg = 100    
    FPS = 25
    BCG_sample_rate = 64
    #training = True
    
    '''output list'''    
    List_HRV_PVT = []
    List_HRV_CCD = []
    List_PERCLOS_PVT = []
    List_PERCLOS_CCD = []    
    List_PERCLOS_max_PVT =[]
    List_PERCLOS_max_CCD =[]
    List_SDP = []
    List_SDW = []
    List_result_PVT = []
    List_time_PVT = []
    List_state = []
    List_time_PERCLOS = []
    List_count_HRV = []
    List_all_HR = []
    
    # HRV
    
    flag_first = 1
    starttime = 0
    for file in HRV_file_list:
        ecg_list = []
        count_HRV = 0
        
        if 'CCD' in file and 'combine' not in data_dir:
            continue
        if use_raw == 1:
            if '.log' not in file:
                continue
            if use_BCG == 1 and 'ECG' in file:  
                continue
            elif use_BCG == 0 and 'ECG' not in file:
                continue  
            currtime = datetime.datetime(1, 1, 1, 0, 0)
            starttime = currtime
        else:            
            date_string = file.replace(FILE_MODE[use_BCG], '').split('_')[1]            
            if flag_first:
                starttime = datetime_convert(date_string)
                if FI_type == 1:
                    break
        
            currtime = datetime_convert(date_string)
        '''get HR'''
        if use_BCG: 
            HR_path = os.path.join(filepath, 'HR_BCG')
            if use_raw:
                HR_path = os.path.join(filepath, 'HR_raw')
            new_file_name = file.split('.')[0] + '.csv'   
            if (not os.path.exists(HR_path) or not os.path.exists(os.path.join(HR_path, new_file_name))) or FLAG_RECALL_ALGO:             
                pressure_data, acc_x, acc_y, acc_z, start_time = rd.read_pressure_acc(HRV_dir, file)
                print('processing ', file.encode("utf-8"))

                for i in range(len(acc_y)):
                    acc_y[i] = 65535 - acc_y[i] if  acc_y[i]> 50000 else acc_y[i]
                    acc_z[i] = 65535 - acc_z[i] if  acc_z[i]> 50000 else acc_z[i]
                    acc_x[i] = 65535 - acc_x[i] if  acc_x[i]> 50000 else acc_x[i]
                if USE_ACC_SUM:
                    acc_y = (acc_x ** 2 + acc_y ** 2 + acc_z ** 2) ** 0.5
                else:
                    temp = ar([acc_x, acc_y, acc_z])
                    max_index = np.argmax(abs(np.mean(temp, axis=1)))
                    acc_y = temp[max_index] * np.sign(np.mean(temp, axis=1)[max_index])
                
                algo = Alg_freq_domain(fs=BCG_sample_rate, fft_window_size=FFT_window_size)
    
                algo.main_func(pressure_data, acc_y)
                
                HR_list = algo.bpm
                HR = np.copy(HR_list)
                HR = HRV.moving_average(HR_list, 5)
                status = algo.status
                confidence_level = algo.confidence_level
                
                if not os.path.exists(HR_path):
                    os.mkdir(HR_path)
                if not os.path.exists(HR_save_all_path):
                    os.mkdir(HR_save_all_path)
                
                with open(os.path.join(HR_path, new_file_name), 'w') as HRfile:
                    with open(os.path.join(HR_save_all_path, new_file_name), 'w') as HRallfile:
                        for i in range(len(HR)):
                            HRfile.write(str(HR[i]) + ',' + str(status[i]) + ',' + str(int(confidence_level[i])) + '\n')
                            HRallfile.write(str(HR[i]) + ',' + str(status[i]) + ',' + str(int(confidence_level[i])) + '\n')
            else:
                HR = []
                status = []
                confidence_level = []
                with open(os.path.join(HR_path, new_file_name), 'r') as HRfile:
                    for line in HRfile.readlines():
                        line = line.split(',')
                        HR.append(float(line[0]))
                        status.append(float(line[1]))
                        confidence_level.append(int(line[2]))
                
        else:  
            HR_path = os.path.join(filepath, 'HR_ECG')
            new_file_name = file.split('.')[0] + '.csv'
            if (not os.path.exists(HR_path) or not os.path.exists(os.path.join(HR_path, new_file_name))) or FLAG_RECALL_ALGO:
                with open(os.path.join(HRV_dir, file), 'r' , encoding="utf-8") as m_input:
                    print('processing ', file.encode("utf-8"))
                    for line in m_input.readlines():
                        if 'Time' in line :
                            if use_raw and starttime == 0:
                                date_string = line.replace('Start Time: ', '')
                                date_string = re.split(":|/|-", date_string)
                                date_string = '-'.join(date_string[:5])
                                starttime = datetime_convert(date_string)
                                currtime = datetime_convert(date_string)
                                continue
                            else:
                                continue
                        if 'State' in line :
                            continue
                        if ',' in line:
                            bpm = line.split(',')[ecg_index]
                        else: 
                            bpm = line.split('\t')[ecg_index]
            
                        ecg_list.append(float(bpm))
                
                HR, HR_t = HRV.ECG_to_HR(ecg_list, fs_ecg) 
                if not os.path.exists(HR_path):
                    os.mkdir(HR_path)
                with open(os.path.join(HR_path, new_file_name), 'w') as HRfile:
                    for hr, hr_t in zip(HR, HR_t):
                        HRfile.write(str(hr_t) + ',' + str(hr) + '\n')
            else:
                HR = []
                HR_t = []
                with open(os.path.join(HR_path, new_file_name), 'r') as HRfile:
                    for line in HRfile.readlines():
                        element = line.split(',')
                        HR.append(float(element[1]))
                        HR_t.append(float(element[0]))
        '''get HRV'''
        if len(HR) == 0:

            count_HRV = 1
            List_HRV_PVT.append([0, 0, 0, 0, 0, 0])
            
        else:
            #resample HR serials for training the threshold
            
            HR_resmple = []
            temp = []
            start = 0
            temp_status = []
            for i in range(len(HR)):
                temp.append(HR[i])
                if use_BCG:
                    temp_status.append(status[i])
                    if len(temp) == RESAMPLE_SIZE :
                        if np.mean(temp_status) < 3.5:
                            HR_resmple.append(np.median(temp))
                        temp = []
                        temp_status = []
                else:
                    if HR_t[i] - start > RESAMPLE_SIZE:
                        HR_resmple.append(np.median(temp))
                        start = HR_t[i]
                        temp = []
                
            if use_BCG:
                List_all_HR = [*List_all_HR, *HR_resmple]
                #List_all_HR = [*List_all_HR, *HR_resmple[: int(60 * 10 / resample_size)]] #use idele part of HR
            else:
                List_all_HR = [*List_all_HR, *HR_resmple]
                
            start = 0
            delta_time = currtime - starttime
            while start < len(HR):   
                bpm = []
                sec = 0
                while start + sec < len(HR)-1:
                    if use_BCG and FLAG_FILTER_STATUS:
                        if status[start + sec] != 6 :
                            bpm.append(HR[start + sec])
                    else:
                        bpm.append(HR[start + sec])
                    if use_BCG:    
                        if len(bpm) == window_size_min * 60:
                            break
                    else:
                        if HR_t[start + sec] - HR_t[start] >= window_size_min * 60:
                            break
                    sec += 1
                                       
                if len(bpm) >= window_size_min * 60 * 0.8:                                       
                    delta_time += datetime.timedelta(seconds = sec) if use_BCG else datetime.timedelta(seconds = HR_t[start + sec] - HR_t[start])
                    start += sec
                else:
                    break
            
                mean_HR, RMSSD, SDNN, LFHF, VLF = HRV.get_HRV(bpm)
                DF = HRV.get_DF(bpm)
                skewness = HRV.get_skweness(bpm)
                kurtosis = HRV.get_kurtosis(bpm)
                count_HRV += 1  
                
                List_time_PVT.append(delta_time.seconds / 3600)
                if delta_time - (currtime - starttime) <= datetime.timedelta(minutes = 10):
                    List_state.append(0)
                else:
                    List_state.append(1) 
                
                List_HRV_PVT.append([mean_HR, RMSSD, SDNN, skewness, kurtosis, 0])#mean_HR, RMSSD, SDNN, VLF = HRV.Get_HRV(bpm)
                
                
        List_count_HRV.append(count_HRV)
        flag_first = 0
    
    '''EAR'''
    if FI_type == 1:
        FPS_ear = 14.7
        flag_first = 1
        PERCLOS_threshold = 0
        for file in EAR_file_list:
            ear_raw = []
            is_PVT = True
            print('processing ', file.encode("utf-8"))
            
            date_string = file.replace('.csv', '').split('_')[1]  
            currtime = datetime_convert(date_string)
            
            #read data
            with open(os.path.join(EAR_dir, file), 'r', encoding='utf-8') as lines:
                
                for line in lines.readlines():
                    line = "".join(filter(lambda ch: ch in '0123456789.', line))
                    if line == '' :
                        continue
                    ear_raw.append(float(line))
            if PERCLOS_threshold ==0 or 'PVT' in file:        
                PERCLOS_threshold = perclos.get_threshold(ear_raw)
            delta_time = currtime - starttime
            start = 0
            while start < len(ear_raw):   
                ear = []
                sec = 0
                while start + sec < len(ear_raw):
                    
                    ear.append(ear_raw[start + sec])
                        
                    if len(ear) == window_size_min * 60 * FPS_ear:
                        break
                    sec += 1
                    
                    
                if len(ear) >= window_size_min * 60 * FPS_ear:
                    start += sec 
                    delta_time += datetime.timedelta(seconds = sec / FPS_ear)
                else:
                    break
            
                ear = np.asarray(ear_raw)
                List_PERCLOS_max_PVT.append(perclos.compute_PERCLOS_localmax(ear, PERCLOS_threshold, FPS_ear))
                List_time_PERCLOS.append(delta_time.seconds / 3600)
                if 'PVT' in file:
                    List_state.append(0)
                else:
                    List_state.append(1) 
                
        
                flag_first = 0
  
  
    '''PVT'''
    flag_first = 1  
    related_threshold = 0
    ind = 0
    list_PVT_time = [0]
    list_PVT_result = [0]
    
    for file in pvt_file_list:
        with open(os.path.join(PVT_dir, file), 'r') as line:
            pvt_data = line.read()
        file = file.replace('PVT_', '')
        date_string = file.replace('.csv', '').split('_')[1] 
        currtime = datetime_convert(date_string)
        curr_deltatime = currtime - starttime
        

        if use_time == 0 or use_time == 1:
            list_PVT_time.append((curr_deltatime.seconds - 600)/3600)
            list_PVT_time.append(curr_deltatime.seconds / 3600)
            KSS1, KSS2 = PVT.get_KSS_text(pvt_data)
            list_PVT_result.append(KSS1)
            list_PVT_result.append(KSS2)
        elif use_time == 2:
            list_PVT_time.append(curr_deltatime.seconds / 3600)
            result_PVT, related_threshold = PVT.get_PVT_text(pvt_data, s_dir, related_threshold = related_threshold)
            list_PVT_result.append(float(result_PVT.split(',')[10]))
     

        flag_first = 0
        ind += 1
    f = interp1d(list_PVT_time, list_PVT_result, kind = 'linear')

    '''generate HR serial for training HR threshold for fatigue'''
    if FI_type == 1:
        if min(List_time_PERCLOS) < list_PVT_time[0]:
            List_time_PERCLOS[np.argmin(List_time_PERCLOS)] = list_PVT_time[0]
        list_PVT_result_interpolation = f(List_time_PERCLOS)
    else:
        if List_time_PVT[0] < list_PVT_time[0]:
            List_time_PVT[0] = list_PVT_time[0]
        list_PVT_result_interpolation = f(List_time_PVT)

    if len(List_all_HR) > 360 :
        List_all_HR = List_all_HR[:360]
    List_subject.append(s_dir)
    List_allHR_ecg.append(List_all_HR)
    List_restingHR.append(min(List_all_HR[int(60 * 2 / RESAMPLE_SIZE): int(60 * 10 / RESAMPLE_SIZE)]))
    List_all_HR = ar(sorted(List_all_HR))  
    norm_all_HR = norm_HRlist_model(List_all_HR)


    y = np.arange(1,101)    
    R_value = np.corrcoef(ar(norm_all_HR), y)       
    paras = np.polyfit(np.log(y), norm_all_HR, 1)
    yfit = np.polyval(paras,np.log(y))
    error = np.mean((yfit - norm_all_HR)**2)
    List_err_model.append(error)
    List_rvalue.append(R_value[1,0])


    temp_HR = []
    for i in range(len(List_HRV_PVT)):
        temp_HR.append(List_HRV_PVT[i][0])

        
    person_norHR = []  
    FI_model = [] 

    for i in range(len(List_HRV_PVT)): #len(List_HRV)):

        normHR = np.argmin(abs(norm_all_HR - List_HRV_PVT[i][0]))
        
        
        List_HRV_PVT[i][0] = normHR            
        list_norHR.append(normHR)
        person_norHR.append(normHR)
        if use_time == 0:
            list_KSS.append(list_PVT_result_interpolation[i])
        else:
            list_lapsep.append(list_PVT_result_interpolation[i])


        list_time.append(List_time_PVT[i])        
        list_name.append(s_dir)
        if FI_type == 2:
            FI = List_HRV_PVT[i][:-1] - model_sdr_mean
            FI = FI / model_sdr_std
            FI = FI * model_coef
            FI_model.append(sum(FI))

        
    for i  in range(len(List_time_PERCLOS)):
        list_time_PERCLOS.append(List_time_PERCLOS[i])
        list_PERCLOS.append(float(List_PERCLOS_max_PVT[i]))   
        list_KSS.append(list_PVT_result_interpolation[i])
    
    
    if FI_type == 1:
        list_person_FI.append(List_PERCLOS_max_PVT)
        list_person_gd.append(List_time_PERCLOS if use_time == 1 else list_PVT_result_interpolation)
        list_person_st.append(List_state)

    elif FI_type == 0:
        
        list_person_FI.append(person_norHR)
        list_person_gd.append(List_time_PVT if use_time == 1 else list_PVT_result_interpolation.tolist())
        list_person_st.append(List_state)
    
    elif FI_type == 2:       
        
        list_person_FI.append(FI_model)
        list_person_gd.append(List_time_PVT if use_time == 1 else list_PVT_result_interpolation.tolist())
        list_person_st.append(List_state)


if FLAG_TRAIN : 
    
    '''training and get ROC performance'''
    #classify_threshold = [ 5, 10, 15, 25]
    if use_time == 1:
        classify_threshold = [1.3, 2, 2.6]
    elif use_time == 1:
        classify_threshold = [5, 6]
    elif use_time == 2:
        classify_threshold = [10, 15]
    
    arr_Gld = ar([item for sublist in list_person_gd for item in sublist])
    arr_FI = ar([item for sublist in list_person_FI for item in sublist])
    if FI_type == 0:
        arr_FI = 100 - arr_FI 
    arr_Gld = arr_Gld[arr_FI != -1]
    arr_FI = arr_FI[arr_FI != -1]
    collect_specificity = []
    collect_sensitivity = []
    collect_percision = []
    collect_pos_count = []
    #for golden in classify_threshold:
    golden = classify_threshold[0]
    with open(data_dir + '/ROC_output.csv', 'w', encoding='utf-8') as output:
        class2 = arr_FI[arr_Gld >= golden]
        class1 = arr_FI[arr_Gld < golden]
        
        class1 = ar(class1)
        class2 = ar(class2)
        
        offset = min(arr_FI)
        multiple = max(arr_FI) - offset
        T = 0
        list_sensitivity = []
        list_specificity = []
        list_percision = []
        collect_threshold = []
        for i in range(200):
            T = (i / 200 * multiple ) + offset
            sensitivity = len(class2[ar(class2)>=T])/ len(class2)
            specificity = len(class1[ar(class1)<T])/ len(class1)
            percision = (len(class2[ar(class2)>=T]) + len(class1[ar(class1)<T])) / len(arr_FI)
            list_percision.append(percision)
            list_sensitivity.append(sensitivity)
            list_specificity.append(1-specificity)
            collect_threshold.append(T)
            output.write(str(T)+ ',' + str(1-specificity) + ',' + str(sensitivity) + '\n')

            
        
        list_specificity = ar(list_specificity)
        list_sensitivity = ar(list_sensitivity)
        collect_specificity.append(list_specificity)
        collect_sensitivity.append(list_sensitivity)
        # collect_percision.append(list_percision)
        # collect_pos_count.append(len(class2)/ len(arr_FI))
        AUC = 0
        for i in range(len(list_specificity)-1):
            AUC += (list_sensitivity[i] + list_sensitivity[i+1]) * 0.5 * abs(list_specificity[i+1] - list_specificity[i])
    
        print('As golden =', golden ,' AUC =', AUC)
    collect_threshold = ar(collect_threshold)

    inver_specificity = collect_specificity[0]
    sensitivity = collect_sensitivity[0]
    threshold = collect_threshold
    count_alert = []
    goal_indexes = []
    threshold_list = []
    for i in range(len(CRITERIA_SPCIFICITY)):
        goal_ind = np.argmin(abs(CRITERIA_SPCIFICITY[i] - inver_specificity))
        
        n_miss, n_flasealert = simulate_bysubject(list_person_FI, threshold[goal_ind], i+1 , classify_threshold[0])
        score = n_miss + n_flasealert
        if n_miss > 0 :
            n_miss, n_flasealert = simulate_bysubject(list_person_FI, threshold[goal_ind -1], i+1 , classify_threshold[0])
            new_score = n_miss + n_flasealert
            while new_score <= score:
                goal_ind -= 1
                score = new_score
                n_miss, n_flasealert = simulate_bysubject(list_person_FI, threshold[goal_ind -1], i+1 , classify_threshold[0])
                new_score = n_miss + n_flasealert
                if score <= 1:
                    break
        else:
            n_miss, n_flasealert =simulate_bysubject(list_person_FI, threshold[goal_ind+1], i+1 , classify_threshold[0])
            new_score = (n_miss + n_flasealert)
            while new_score <= score:
                goal_ind += 1
                score = new_score
                n_miss, n_flasealert = simulate_bysubject(list_person_FI, threshold[goal_ind +1], i+1 , classify_threshold[0])
                new_score = n_miss + n_flasealert
                if score <= 1:
                    break


        goal_indexes.append(goal_ind)
        count_alert.append(score)
        threshold_list.append(threshold[goal_ind])
    
    #proper_alert_count = np.argmax(count_alert) + 1
    proper_alert_count = np.argmin(count_alert) + 1
    if fix_alert_count > 0 :
        proper_alert_count = fix_alert_count
    
    if FI_type == 0:
        final_threshold = 100 - threshold[goal_indexes[proper_alert_count -1]] if use_time else threshold[goal_indexes[proper_alert_count -1]]
    else:
        final_threshold = threshold[goal_indexes[proper_alert_count -1]]

    print('proper threshold =', round(final_threshold, 2) ,' sensitivity =',round(sensitivity[goal_indexes[proper_alert_count -1]], 2), 'count = ', proper_alert_count )
    
    
    if FLAG_SIMULATION:        
        list_delaytime = []
        list_falsealert = []
        count_miss_alert = 0
        list_all_alert = []

        for i in range(len(list_person_FI)):
            list_delaytime.append(-1)
            plt.figure()
            plt.title(List_subject[i] + '_simulation')
            # if FI_type == 1 or use_time == 0:
            #     list_fatigue = list_person_FI[i] >= final_threshold
            if FI_type == 0:
                list_fatigue = list_person_FI[i] <= final_threshold
            else:
                list_fatigue = list_person_FI[i] >= final_threshold
            plt.plot(list_person_gd[i], list_fatigue, 'o', color = 'black')
            
            list_alert_stage = []
            
            list_alert = []
            count = 0
            correct_alert = 0
            falsealert = 0
            for j in range(len(list_person_gd[i])):                
                if list_fatigue[j] == True:
                    count += 1
                    if count >= proper_alert_count:
                        list_alert_stage.append(1)
                        list_alert.append(1)
                        if correct_alert == 0 and list_person_gd[i][j] >= int(classify_threshold[0]):
                            if list_delaytime[-1] == -1:
                                list_delaytime[-1] = list_person_gd[i][j] - int(classify_threshold[0])
                            correct_alert = 1
                        elif list_person_gd[i][j] < int(classify_threshold[0]):
                            falsealert += 1
                    else:
                        list_alert_stage.append(0)
                    
                else:
                    count = 0
                    list_alert_stage.append(0)
            list_all_alert.append(list_alert_stage)        
            plt.plot(ar(list_person_gd[i])[ar(list_alert_stage) == 1], list_alert, 'o', color = 'red')
            plt.savefig(os.path.join(figure_save_path_fatigue, List_subject[i] + '_fatigue.png'), format='png')
            plt.ylim([0, 3])
            plt.close()
            if correct_alert == 0:
                count_miss_alert += 1
            list_falsealert.append(falsealert)
        list_delaytime = ar(list_delaytime)
        print('amount of miss alert = ', count_miss_alert)
        print('amount of false alert = ', sum(list_falsealert), 'mean = ', round(sum(list_falsealert) / len(list_falsealert), 2))
        print('mean delay of alert is ', round(60 * np.mean(list_delaytime[list_delaytime > 0])), 'minutes' )
        print('longest delay of alert is ', round(60 * max(list_delaytime)), 'minutes' )
        
        '''save the modeling feature every samples  refer the worknotebook 20210409'''
        with open('fatigue_model_feature.csv', 'w') as csvfile:
            csvfile.write('name, alert, delay, max_HR, difference, diff_ratio, restingHR, error_mnodel, R_value\n')
            for i in range(len(list_delaytime)):
                line = List_subject[i] + ',' + str(sum(list_all_alert[i])) + ',' + str(list_delaytime[i])
                max_HR = max(List_allHR_ecg[i])
                difference = List_restingHR[i] - min(List_allHR_ecg[i])
                diff_ratio = difference / List_restingHR[i]

                line += ',' + str(max_HR)
                line += ',' + str(difference)
                line += ',' + str(diff_ratio)
                line += ',' + str(List_restingHR[i])
                line += ',' + str(List_err_model[i])
                line += ',' + str(List_rvalue[i])
                csvfile.write( line + '\n')

        '''plot simulation plot distrobution'''
        list_count_checkpoint = []
        plt.figure()
        plt.title('checkpoint_distribution')
        for i in range(len(List_subject)):
            
            gd = ar(list_person_gd[i])[ar(list_person_st[i]) == drive_ornot]
            alert = ar(list_person_gd[i])[np.logical_and(ar(list_person_st[i]) == drive_ornot , ar(list_all_alert[i]) == 1)]

        
            list_count_checkpoint.append(len(list_person_gd[i]))
            x = [i+1] * len(gd)
            x = [List_subject[i]] * len(gd)
            
            plt.plot(x,gd, 's', color = 'black')
            if len(alert) > 0:
                
                #plt.plot([i+1] * len(alert), alert, 's', color = 'red')
                plt.plot([List_subject[i]] * len(alert), alert, 's', color = 'red')
        #plt.ylim([13, 16])
        plt.xticks(rotation=45)
        if FLAG_SHOW_PLOT:
            plt.show()
        plt.savefig(os.path.join(figure_save_path_fatigue, 'checkpoint_distribution_fatigue.png'), format='png')
        
        plt.close()    
        print('mean check point = ', round(np.mean(list_count_checkpoint),1), ' , std = ', round(np.std(list_count_checkpoint), 2))
        
        
        
            