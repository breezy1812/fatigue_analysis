# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:09:58 2020

@author: breez
"""

import src.get_HRV as HRV
import src.get_PERCLOS as perclos
import src.read_data as rd
import src.get_PVT as PVT
# from src.alg_freq_domain import Alg_freq_domain
from src.alg_freq_domain import Alg_freq_domain
# import WHEEL_algo.wheel_algo as wheel
import matplotlib.pyplot as plt
from scipy import optimize as op
from scipy import integrate
import numpy as np
import os

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
            

''''Alg_freq_domain parameters setting'''
USE_ACC_SUM = True
MANUALLY_WEIGHT = True
overlap_weight = [1, 1, 1, 1, 1, 1, 1]
FFT_window_size=30 

'''setting'''
flag_total = False
head_end_only = False
include_last_two = False
calib_KSS = False
gen_ECG = True

weedout_list = ['chao', 'ming', 'yongda', 'fei','grace','he','jie','sheng','yong','yongda'] # ['shu','eric', 'roxus', 'kim', 'gui', 'grace', 'larry' , 'yangfei', 'dong', 'jie', 'he', 'ming']

with open('LHR.txt', 'r', encoding='utf-8') as subject:
    LHR = (subject.read())

data_dir =  "Data_driving2" 
subject_dirs = os.listdir(data_dir)
List_ecg_before = []
List_ecg_after = []
    
List_bcg_before = []
List_bcg_after = []


List_subject = []
List_allHR_ecg = []
List_allHR_bcg = []

list_norHR = []
list_KSS = []
list_lapsep = []
list_time = []
list_PERCLOS = []
list_name = []

list_PERCLOS_bytime = []
list_norHR_bytime = []
list_skewness_bytime = []
list_kurtosis_bytime = []
list_norHR_bcg_bytime = []
list_Lapsep_by_time = []
list_RT250_by_time = []
list_KSS_by_time = []
list_HR_bytime = []
list_RMSSD_bytime = []
list_SDNN_bytime = []

figure_save_path_ecg = os.path.join(data_dir, '0_ECG_histogram')
figure_save_path_bcg = os.path.join(data_dir, '0_BCG_histogram')
if not os.path.exists(figure_save_path_ecg):
    os.mkdir(figure_save_path_ecg)
if not os.path.exists(figure_save_path_bcg):
    os.mkdir(figure_save_path_bcg)



    
flag_total = True
for s_dir in subject_dirs:    
    if '.csv' in s_dir or 'total' in s_dir:
        continue  
    filepath = os.path.join(data_dir, s_dir)      
    if not os.path.exists(os.path.join(filepath, 'ECG')):
        continue
    if s_dir in weedout_list:
        continue


# s_dir = 'ming'
# if 1:
    filepath = os.path.join(data_dir, s_dir)
    output_title = 'DF, HR, RMSSD, SDNN, VLF, norHR'
    data_head = ['name', 'sleep_time', 'wake_time', 'sleep_2h', 'HR', 'SSS', 'KSS', 'RT', 'RT_fast', 'PVT_inverRT', 'lapse', 'lapse_percent']
    PVT_title = ', '.join(map(str, data_head))
    in_office = False
    data_type_list = os.listdir(filepath)

    ECG_dir = os.path.join(filepath, 'ECG')
    BCG_dir = os.path.join(filepath, 'BCG')
    PVT_dir = os.path.join(filepath, 'PVT')
    EAR_dir = os.path.join(filepath, 'EAR')
    
    ECG_file_list = os.listdir(ECG_dir)
    BCG_file_list = os.listdir(BCG_dir)
    pvt_file_list = os.listdir(PVT_dir)
    EAR_file_list = os.listdir(EAR_dir)
    
    if 'WHE' in data_type_list:
        WHE_dir = os.path.join(filepath, 'WHE')
        WHE_file_list = os.listdir(WHE_dir)
        in_office = True
    
    #parameters
    
    ecg_index = 6
    window_size_min = 5
    fs_ecg = 100    
    FPS = 25
    BCG_sample_rate = 64
    training = True
    #output list
    
    List_HRV_PVT = []
    List_HRV_CCD = []
    
    List_HRV_bcg_PVT = []
    List_HRV_bcg_CCD = []
    List_PERCLOS_PVT = []
    List_PERCLOS_CCD = []    
    List_PERCLOS_max_PVT =[]
    List_PERCLOS_max_CCD =[]
    List_SDP = []
    List_SDW = []
    List_result_PVT = []
    
    List_all_HR_ECG = []
    List_all_HR_BCG = []
    
    # ECG
    flag_first = 1
    for file in ECG_file_list:
        is_PVT = True
        ecg_list = []
        
        if 'CCD' in file:
            continue
        
        if flag_first == 0 and ECG_file_list.index(file) != len(ECG_file_list) - 1  and head_end_only:
            if include_last_two and ECG_file_list.index(file) == len(ECG_file_list) - 2:
                   flag_first = 0
            else:
                continue   

            
        with open(os.path.join(ECG_dir, file), 'r') as m_input:
            print('processing ', file)
            for line in m_input.readlines():
                if 'Time' in line :
                    continue
                if 'State' in line :
                    continue
                if ',' in line:
                    bpm = line.split(',')[ecg_index]
                else: 
                    bpm = line.split('\t')[ecg_index]
    
                if abs(float(bpm))  < 10 and len(ecg_list) < 10:
                    ecg_index = 5
                else:
                    ecg_list.append(float(bpm))
        
        HR, HR_t = HRV.ECG_to_HR(ecg_list, fs_ecg)
        if len(HR_t) == 0:
            DF = 0
            mean_HR = 0
            RMSSD = 0
            SDNN = 0
            VLF = 0
            if is_PVT:
                List_HRV_PVT.append([DF, mean_HR, RMSSD, SDNN, VLF, 0])#mean_HR, RMSSD, SDNN, VLF = HRV.Get_HRV(bpm)
            else:
                List_HRV_CCD.append([DF, mean_HR, RMSSD, SDNN, VLF, 0])
        else:
        
            if is_PVT:
                List_all_HR_ECG = [*List_all_HR_ECG, *HR]             
                
                # hist_ecg = np.zeros(200)
                # for i in range(len(List_all_HR_ECG)):
                #     hist_ecg[int(List_all_HR_ECG[i])] += 1  
                # y = np.array(hist_ecg / len(List_all_HR_ECG))*100
                # x = np.array(range(200))    
                
                # if training and len(List_all_HR_ECG) > 19*60:
                #     mean = sum(x * y) / sum(y)                 #note this correction
                #     sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))       #note this correction
                #     popt,pcov = op.curve_fit(Gauss,x,y,p0=[1,mean,sigma])
                #     new_y = Gauss(x,*popt)
                #     training = False
            if is_PVT:                
                bpm = HR[: np.argmin(abs(HR_t - (window_size_min * 60))) ]
            else:
                start = HR_t[-1] - int(60 * (window_size_min)) 
                start = np.argmin(abs(HR_t - start))
                bpm = HR[start:]

            if len(bpm) < window_size_min * 60:
                bpm = HR
            mean_HR, RMSSD, SDNN, LFHF, VLF = HRV.get_HRV(bpm)
            skewness = HRV.get_skweness(bpm)
            kurtosis = HRV.get_kurtosis(bpm)
            DF = HRV.get_DF(bpm)
            
            if is_PVT:
                List_HRV_PVT.append([skewness, mean_HR, RMSSD, SDNN, kurtosis, 0])#mean_HR, RMSSD, SDNN, VLF = HRV.Get_HRV(bpm)
            else:
                List_HRV_CCD.append([skewness, mean_HR, RMSSD, LFHF, kurtosis, 0])
            
 
        flag_first = 0
        
            
    # BCG
    flag_first = 1
    for file in BCG_file_list:
        
        FFT_window_size=30        
        is_PVT = True
        BCG_list = []
        print('processing ', file)
        
        if 'CCD' in file:
            continue
    
        pressure_data, acc_x, acc_y, acc_z, start_time = rd.read_pressure_acc(BCG_dir, file)
        if USE_ACC_SUM:
            acc_y = (acc_x ** 2 + acc_y ** 2 + acc_z ** 2) ** 0.5
        else:
            temp = np.array([acc_x, acc_y, acc_z])
            max_index = np.argmax(abs(np.mean(temp, axis=1)))
            acc_y = temp[max_index] * np.sign(np.mean(temp, axis=1)[max_index])
        
        algo = Alg_freq_domain(fs=BCG_sample_rate, fft_window_size=FFT_window_size)
        
        # algo.USE_MANUALLY_WEIGHT = MANUALLY_WEIGHT
        # if MANUALLY_WEIGHT == True:
        #     algo.overlap_weight = overlap_weight
        #     if s_dir in LHR:
        #         algo.overlap_weight = [1, 1, 1, 1, 1, 1, 1]        
        
        algo.main_func(pressure_data, acc_y)
        
        HR_list = algo.bpm
        if is_PVT:
            List_all_HR_BCG = [*List_all_HR_BCG , *HR_list]
        if flag_first == 0 and BCG_file_list.index(file) != len(BCG_file_list) - 1 and head_end_only:
            if include_last_two and BCG_file_list.index(file) == len(BCG_file_list) - 2:
                flag_first = 0
            else:
                continue
            
        HR = np.copy(HR_list)
        #HR = HRV.moving_average(HR_list, 5)
        if is_PVT:
            List_all_HR_BCG = [*List_all_HR_BCG, *HR]
        
        HR_t = np.linspace(0, len(HR_list)-1, len(HR_list))
        if is_PVT:
            start = 0
        else:
            start = len(HR) - int(60 * (window_size_min))
        if start < 0 :
            start = 0
        bpm = HR[start: start + window_size_min * 60]
        
        if len(bpm) < window_size_min * 60:
            bpm = HR
            
        mean_HR, RMSSD, SDNN, LFHF, VLF = HRV.get_HRV(bpm)
        skewness = HRV.get_skweness(bpm)
        kurtosis = HRV.get_kurtosis(bpm)
        
        flag_first = 0
        # else:
        #     refer_HR = (mean_HR - refer_HR)
        #HRV_record = str(HRV.get_DF(bpm)) + ',' + str(mean_HR) + ',' + str(RMSSD) + ',' + str(SDNN) + ',' + str(VLF) + ',' + str((mean_HR - refer_HR))              
  

        if is_PVT:
            List_HRV_bcg_PVT.append([skewness, mean_HR, RMSSD, SDNN, kurtosis, 0])
        else:
            List_HRV_bcg_CCD.append([HRV.get_DF(bpm), mean_HR, RMSSD, SDNN, VLF, 0])
    
    '''PVT'''
    flag_first = 1  
    related_threshold = 0
    for file in pvt_file_list:
        with open(os.path.join(PVT_dir, file), 'r') as line:
            pvt_data = line.read()
        #print('processing ', file)
        
        result_PVT, related_threshold = PVT.get_PVT_text(pvt_data, s_dir, related_threshold = related_threshold)
        #result_PVT = PVT.get_PVT_text(pvt_data, s_dir, optimal_threshold = True)
        if flag_first == 0 and pvt_file_list.index(file) != len(pvt_file_list) - 1 and head_end_only:
            if include_last_two and pvt_file_list.index(file) == len(pvt_file_list) - 2:
                flag_first = 0
            else:
                continue
        List_result_PVT.append(result_PVT)  
        flag_first = 0
    
    '''EAR'''
    FPS_ear = 14.7
    flag_first = 1
    PERCLOS_threshold = 0
    for file in EAR_file_list:
        ear_raw = []
        is_PVT = True
        print('processing ', file)
        
    
        
        #read data
        with open(os.path.join(EAR_dir, file), 'r', encoding='utf-8') as lines:
            
            for line in lines.readlines():
                line = "".join(filter(lambda ch: ch in '0123456789.', line))
                if line == '' :
                    continue
                ear_raw.append(float(line))
        if PERCLOS_threshold ==0 or 'PVT' in file:        
            PERCLOS_threshold = perclos.get_threshold(ear_raw)
            #FPS_ear = len(ear_raw) / (10 * 60)
            
        if 'PVT' in file:
            if len(ear_raw) > FPS_ear*10*60:
                ear_raw = ear_raw[int(FPS_ear*10*60) * -1:]
        
        #ear_raw = ear_raw[:int( len(ear_raw)/2)]     
    
        if flag_first == 0 and EAR_file_list.index(file) != len(EAR_file_list) - 1 and head_end_only:
            if include_last_two and EAR_file_list.index(file) == len(EAR_file_list) - 2:
                flag_first = 0
            else:
                continue
            
        #PERCLOS_threshold = perclos.get_threshold(ear_raw)
        
        ear = np.asarray(ear_raw)
        
        if is_PVT:
            List_PERCLOS_PVT.append(str(perclos.compute_PERCLOS_overall(ear, PERCLOS_threshold)))
            List_PERCLOS_max_PVT.append(str(perclos.compute_PERCLOS_localmax(ear, PERCLOS_threshold, FPS_ear)))
        else:
            List_PERCLOS_CCD.append(str(perclos.compute_PERCLOS_overall(ear, PERCLOS_threshold)))
            List_PERCLOS_max_CCD.append(str(perclos.compute_PERCLOS_localmax(ear, PERCLOS_threshold, FPS_ear)))
        
        # if flag_first == 1 and head_end_only:
        #     List_PERCLOS_PVT.pop(1)
        #     List_PERCLOS_max_PVT.pop(1)
        flag_first = 0
  
  
    
    with open(filepath + '_output.csv', 'w', encoding='utf-8') as output:
        List_subject.append(s_dir)
        List_all_HR_BCG = np.array(sorted(List_all_HR_BCG))
        List_all_HR_ECG = np.array(sorted(List_all_HR_ECG))
        
        hist_ecg = np.zeros(100)
        hist_bcg = np.zeros(100)
        bins = range(0,150,2)
        plt.figure()
        plt.title(s_dir + '_ECG')
        plt.hist(List_all_HR_ECG, bins = bins)
        plt.xlim([40, 120])
        plt.savefig(os.path.join(figure_save_path_ecg, s_dir + '_raw_filtered.png'), format='png')
        plt.close()
        
        plt.figure()
        plt.title(s_dir + '_BCG')
        plt.hist(List_all_HR_BCG, bins = bins)
        plt.xlim([40, 120])
        plt.savefig(os.path.join(figure_save_path_bcg, s_dir + '_raw_filtered.png'), format='png')
        plt.close()
        # for i in range(len(List_all_HR_ECG)):
        #     hist_ecg[int(List_all_HR_ECG[i]/2)] += 1   
        # for i in range(len(List_all_HR_BCG)):
        #     hist_bcg[int(List_all_HR_BCG[i]/2)] += 1
        # List_allHR_ecg.append(np.array(hist_ecg))
        # List_allHR_bcg.append(np.array(hist_bcg))
        
              
        output.write(PVT_title + ',' + output_title + '\n')
        for i in range(len(List_HRV_PVT)): #len(List_HRV)):
            normHR = np.argmin(abs(List_all_HR_ECG - List_HRV_PVT[i][1])) *100 / len(List_all_HR_ECG)
            normHR_bcg = np.argmin(abs(List_all_HR_BCG - List_HRV_bcg_PVT[i][1])) *100 / len(List_all_HR_BCG)
        
            temp_KSS = float(List_result_PVT[i].split(',')[6])
            if calib_KSS : #and float(List_result_PVT[-1].split(',')[6]) < 4:
                temp_KSS += 1.5
            
            List_HRV_PVT[i][-1] = normHR
            List_HRV_bcg_PVT[i][-1] = normHR_bcg
            list_norHR.append(normHR_bcg)
            list_KSS.append(temp_KSS)
            list_lapsep.append(float(List_result_PVT[i].split(',')[11]))
            list_time.append(i)
            list_PERCLOS.append(float(List_PERCLOS_max_PVT[i]))
            list_name.append(s_dir)
            if i == 0:
                List_ecg_before.append(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_PVT[i])))
            else:
                List_ecg_after.append(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_PVT[i])))
            output.write(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_PVT[i])) + '\n')
            # else:
            #     output.write(List_SDW[i-1] + ',' + List_SDP[i-1] + ',' + List_PERCLOS_PVT[i] + ',' + List_PERCLOS_max_PVT[i] + ',' + List_HRV_PVT[i] + ',' + List_PERCLOS_CCD[i - 1] + ',' + List_PERCLOS_max_CCD[i - 1] + ',' + List_HRV_CCD[i - 1] + '\n')
        list_PERCLOS_bytime.append(np.array(List_PERCLOS_PVT))
        if gen_ECG:        
            
            list_norHR_bytime.append(np.array([List_HRV_PVT[i][-1] for i in range(len(List_HRV_PVT)) ]))
            list_skewness_bytime.append(np.array([List_HRV_PVT[i][0] for i in range(len(List_HRV_PVT))]))
            list_kurtosis_bytime.append(np.array([List_HRV_PVT[i][-2] for i in range(len(List_HRV_PVT))]))
            list_HR_bytime.append(np.array([List_HRV_PVT[i][1] for i in range(len(List_HRV_PVT))]))
            list_RMSSD_bytime.append(np.array([List_HRV_PVT[i][2] for i in range(len(List_HRV_PVT))]))
            list_SDNN_bytime.append(np.array([List_HRV_PVT[i][3] for i in range(len(List_HRV_PVT))]))
        else:
            list_norHR_bcg_bytime.append(np.array([List_HRV_bcg_PVT[i][-1] for i in range(len(List_HRV_bcg_PVT))  ]))
            list_norHR_bytime.append(np.array([List_HRV_bcg_PVT[i][-1] for i in range(len(List_HRV_PVT)) ]))
            list_skewness_bytime.append(np.array([List_HRV_bcg_PVT[i][0] for i in range(len(List_HRV_PVT))]))
            list_kurtosis_bytime.append(np.array([List_HRV_bcg_PVT[i][-2] for i in range(len(List_HRV_PVT))]))
            list_HR_bytime.append(np.array([List_HRV_bcg_PVT[i][1] for i in range(len(List_HRV_PVT))]))
            list_RMSSD_bytime.append(np.array([List_HRV_bcg_PVT[i][2] for i in range(len(List_HRV_PVT))]))
            list_SDNN_bytime.append(np.array([List_HRV_bcg_PVT[i][3] for i in range(len(List_HRV_PVT))]))


        list_RT250_by_time.append(np.array([float(List_result_PVT[i].split(',')[11]) for i in range(len(List_result_PVT)) ]))
        list_Lapsep_by_time.append(np.array([float(List_result_PVT[i].split(',')[10]) for i in range(len(List_result_PVT)) ]))
        list_KSS_by_time.append(np.array([float(List_result_PVT[i].split(',')[6]) for i in range(len(List_result_PVT))]))
        output.write(PVT_title + ',' +output_title + '\n')
        for i in range(len(List_HRV_bcg_PVT)): #len(List_HRV)):
            List_HRV_bcg_PVT[i][-1] = np.argmin(abs(List_all_HR_BCG - List_HRV_bcg_PVT[i][1])) *100 / len(List_all_HR_BCG)
            if i == 0:
                List_bcg_before.append(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_bcg_PVT[i])))
            else:
                List_bcg_after.append( List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_bcg_PVT[i])))
            output.write(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_bcg_PVT[i])) + '\n')
            
        
list_PERCLOS_bytime = np.array(list_PERCLOS_bytime)
list_norHR_bytime = np.array(list_norHR_bytime)
list_norHR_bcg_bytime = np.array(list_norHR_bcg_bytime)
list_Lapsep_by_time = np.array(list_Lapsep_by_time)
list_KSS_by_time = np.array(list_KSS_by_time)
list_skewness_bytime = np.array(list_skewness_bytime)
list_kurtosis_bytime = np.array(list_kurtosis_bytime)
list_HR_bytime = np.array(list_HR_bytime)
list_RMSSD_bytime = np.array(list_RMSSD_bytime)
list_SDNN_bytime = np.array(list_SDNN_bytime)
list_RT250_by_time = np.array(list_RT250_by_time)
if flag_total : 
    if head_end_only:
        with open(data_dir + '/total_output.csv', 'w', encoding='utf-8') as output:
            output.write('Subject, ' + ', '.join(map(str, List_subject)) + '\n') 
            output.write('norHR_ecg_before, ' + ', '.join(map(str, [line.split(',')[-1] for line in List_ecg_before])) + '\n') 
            output.write('norHR_ecg_after, ' + ', '.join(map(str, [line.split(',')[-1] for line in List_ecg_after])) + '\n') 
            output.write('norHR_bcg_before, ' + ', '.join(map(str, [line.split(',')[-1] for line in List_bcg_before])) + '\n') 
            output.write('norHR_bcg_after, ' + ', '.join(map(str, [line.split(',')[-1] for line in List_bcg_after])) + '\n') 
            
            output.write('KSS_before, ' + ', '.join(map(str, [line.split(',')[6] for line in List_bcg_before])) + '\n') 
            output.write('KSS_after, ' + ', '.join(map(str, [line.split(',')[6] for line in List_bcg_after])) + '\n') 
            output.write(PVT_title + ',' + output_title + '\n')
            
            mean, std = get_statis(List_ecg_before)
            output.write('mean ecg_before, ' + ', '.join(map(str, mean)) + '\n')  
            output.write('std ecg_before, ' + ', '.join(map(str, std)) + '\n')  
            output.write('\n')
            
            mean, std = get_statis(List_ecg_after)
            output.write('mean ecg_after, ' + ', '.join(map(str, mean)) + '\n')  
            output.write('std ecg_after, ' + ', '.join(map(str, std)) + '\n')  
            output.write('\n')
            
            mean, std = get_statis(List_bcg_before)
            output.write('mean bcg_before, ' + ', '.join(map(str, mean)) + '\n')  
            output.write('std bcg_before, ' + ', '.join(map(str, std)) + '\n')  
            output.write('\n')
            
            mean, std = get_statis(List_bcg_after)
            output.write('mean bcg_after, ' + ', '.join(map(str, mean)) + '\n')  
            output.write('std bcg_after, ' + ', '.join(map(str, std)) + '\n')  
            output.write('\n')
    else:
        with open(data_dir + '/total_output.csv', 'w', encoding='utf-8') as output:
            for i in range(len(List_subject)):
                output.write( List_subject[i] + ',' + ', '.join(map(str, list_PERCLOS_bytime[i])) + '\n') 
            output.write('\n')
            for i in range(len(List_subject)):
                output.write( List_subject[i] + ',' + ', '.join(map(str, list_skewness_bytime[i])) + '\n')
            output.write('\n')
            for i in range(len(List_subject)):
                output.write( List_subject[i] + ',' + ', '.join(map(str, list_kurtosis_bytime[i])) + '\n')
            output.write('\n')
            for i in range(len(List_subject)):
                output.write( List_subject[i] + ',' + ', '.join(map(str, list_Lapsep_by_time[i])) + '\n')
            output.write('\n')
            for i in range(len(List_subject)):
                output.write( List_subject[i] + ',' + ', '.join(map(str, list_KSS_by_time[i])) + '\n')
        with open(data_dir + '/train_ECG_lapse_5min_norHR.csv', 'w', encoding='utf-8') as output:
            
            output.write(', HR, RMSSD, SDNN, skwenwss, kurtosis \n')
            for i in range(len(List_subject)):
                for j in range(len(list_skewness_bytime[i])):
                    out = []
                    label = 1 if j > 1 else 0
                    label = 1 if list_Lapsep_by_time[i][j] >= 10 else 0
                    out.append(label)
                    out.append(list_norHR_bytime[i][j])                    
                    out.append(list_RMSSD_bytime[i][j])
                    out.append(list_SDNN_bytime[i][j])
                    out.append(list_skewness_bytime[i][j])
                    out.append(list_kurtosis_bytime[i][j])
                    output.write(', '.join(map(str, out)) + '\n')

   