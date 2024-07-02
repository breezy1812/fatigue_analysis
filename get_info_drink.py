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
from src.alg_freq_domain_origin import Alg_freq_domain
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
USE_ACC_SUM = False
MANUALLY_WEIGHT = False
overlap_weight = [1, 1, 1, 1, 1, 0, 0]
flag_total = False
first_five = True
# all_day = True
# head_end_only = True
# include_last_two = False


with open('LHR.txt', 'r', encoding='utf-8') as subject:
    LHR = (subject.read())

data_dir = "DATA_drink2" #"Data_driving" 
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


# s_dir = 'han'
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
    #EAR_dir = os.path.join(filepath, 'EAR')
    
    ECG_file_list = os.listdir(ECG_dir)
    BCG_file_list = os.listdir(BCG_dir)
    pvt_file_list = os.listdir(PVT_dir)
    #EAR_file_list = os.listdir(EAR_dir)
    
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
    
                ecg_list.append(float(bpm))
        
        HR, HR_t = HRV.ECG_to_HR(ecg_list, fs_ecg, 5)
        if is_PVT:
            List_all_HR_ECG = [*List_all_HR_ECG, *HR]
            
            
            hist_ecg = np.zeros(200)
            for i in range(len(List_all_HR_ECG)):
                hist_ecg[int(List_all_HR_ECG[i])] += 1  
            y = np.array(hist_ecg / len(List_all_HR_ECG))*100
            x = np.array(range(200))    
            
            if training and len(List_all_HR_ECG) > 19*60:
                mean = sum(x * y) / sum(y)                 #note this correction
                sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))       #note this correction
                popt,pcov = op.curve_fit(Gauss,x,y,p0=[1,mean,sigma])
                new_y = Gauss(x,*popt)
                
                training = False
                #v, err = integrate.quad(Gauss, 0, 50, args = (popt[0], popt[1], popt[2]))
            # if ~training:
            #     if min(HR) < popt[1] - 4*popt[2]:
            #         popt[2] = (popt[1] - min(HR))/4
                
            
            
        # if flag_first == 0 and ECG_file_list.index(file) != len(ECG_file_list) - 1  and head_end_only:
        #     if include_last_two and ECG_file_list.index(file) == len(ECG_file_list) - 3:
        #         flag_first = 0
        #     else:
        #         continue
            
        if first_five:
            start = 0
        else:
            start = len(HR) - int(60 * (window_size_min))
        if start < 0 :
            start = 0
        bpm = HR[start: start + window_size_min * 60]
        
        if len(bpm) < window_size_min * 60:
            bpm = HR
        mean_HR, RMSSD, SDNN, VLF = HRV.get_HRV(bpm)
        flag_first = 0
        # else:
        #     refer_HR = ((mean_HR - refer_HR)/refer_HR) * 100
        #HRV_record = str(HRV.get_DF(bpm)) + ',' + str(mean_HR) + ',' + str(RMSSD) + ',' + str(SDNN) + ',' + str(VLF) + ',' + str((mean_HR - refer_HR))    
        #List_HRV_PVT.append(HRV_record)#mean_HR, RMSSD, SDNN, VLF = HRV.Get_HRV(bpm)
        #List_HRV_CCD.append(HRV_record)
        if is_PVT:
            List_HRV_PVT.append([HRV.get_DF(bpm), mean_HR, RMSSD, SDNN, VLF, 0])#mean_HR, RMSSD, SDNN, VLF = HRV.Get_HRV(bpm)
        else:
            List_HRV_CCD.append([HRV.get_DF(bpm), mean_HR, RMSSD, SDNN, VLF, 0])
            
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
        
        algo.USE_MANUALLY_WEIGHT = MANUALLY_WEIGHT
        if MANUALLY_WEIGHT == True:
            algo.overlap_weight = overlap_weight
            if s_dir in LHR:
                algo.overlap_weight = [1, 1, 1, 1, 1, 1, 1]        
        
        algo.main_func(pressure_data, acc_y)
        
        HR_list = algo.bpm
        if is_PVT:
            List_all_HR_BCG = [*List_all_HR_BCG , *HR_list]
            
        # if flag_first == 0 and BCG_file_list.index(file) != len(BCG_file_list) - 1 and head_end_only:
        #     if include_last_two and BCG_file_list.index(file) == len(BCG_file_list) - 3:
        #         flag_first = 0
        #     else:
        #         continue
            
        HR = np.copy(HR_list)
        HR = HRV.moving_average(HR_list, 5)
        HR_t = np.linspace(0, len(HR_list)-1, len(HR_list))
        if first_five:
            start = 60
        else:
            start = len(HR) - int(60 * (window_size_min+1))
        if start < 0 :
            start = 0
        bpm = HR[start: start + window_size_min * 60]
        
        if len(bpm) < window_size_min * 60:
            bpm = HR
            
        mean_HR, RMSSD, SDNN, VLF = HRV.get_HRV(bpm)
        
        flag_first = 0
        # else:
        #     refer_HR = (mean_HR - refer_HR)
        #HRV_record = str(HRV.get_DF(bpm)) + ',' + str(mean_HR) + ',' + str(RMSSD) + ',' + str(SDNN) + ',' + str(VLF) + ',' + str((mean_HR - refer_HR))              
  

        if is_PVT:
            List_HRV_bcg_PVT.append([HRV.get_DF(bpm), mean_HR, RMSSD, SDNN, VLF, 0])
        else:
            List_HRV_bcg_CCD.append([HRV.get_DF(bpm), mean_HR, RMSSD, SDNN, VLF, 0])
    
    '''PVT'''
    flag_first = 1    
    for file in pvt_file_list:
        with open(os.path.join(PVT_dir, file), 'r') as line:
            pvt_data = line.read()
        print('processing ', line)
        result_PVT = PVT.get_PVT_text(pvt_data, s_dir)
        # if flag_first == 0 and pvt_file_list.index(file) != len(pvt_file_list) - 1 and head_end_only:
        #     if include_last_two and pvt_file_list.index(file) == len(pvt_file_list) - 3:
        #         flag_first = 0
        #     else:
        #         continue
        List_result_PVT.append(result_PVT)  
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

        
        output.write(PVT_title + ',' + output_title + '\n')
        for i in range(len(List_HRV_PVT)): #len(List_HRV)):
            normHR = np.argmin(abs(List_all_HR_ECG - List_HRV_PVT[i][1])) *100 / len(List_all_HR_ECG)
            
            
            
            List_HRV_PVT[i][-1] = normHR
            list_norHR.append(100-normHR)
            list_KSS.append(float(List_result_PVT[i].split(',')[6]))
            list_lapsep.append(float(List_result_PVT[i].split(',')[11]))
            list_time.append(i)
            #list_PERCLOS.append(float(List_PERCLOS_max_PVT[i]))
            
            if i == 0:
                List_ecg_before.append(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_PVT[i])))
            else:
                List_ecg_after.append(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_PVT[i])))
            output.write(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_PVT[i])) + '\n')
            # else:
            #     output.write(List_SDW[i-1] + ',' + List_SDP[i-1] + ',' + List_PERCLOS_PVT[i] + ',' + List_PERCLOS_max_PVT[i] + ',' + List_HRV_PVT[i] + ',' + List_PERCLOS_CCD[i - 1] + ',' + List_PERCLOS_max_CCD[i - 1] + ',' + List_HRV_CCD[i - 1] + '\n')
                
        
        output.write(PVT_title + ',' +output_title + '\n')
        for i in range(len(List_HRV_bcg_PVT)): #len(List_HRV)):
            List_HRV_bcg_PVT[i][-1] = np.argmin(abs(List_all_HR_BCG - List_HRV_bcg_PVT[i][1])) *100 / len(List_all_HR_BCG)
            if i == 0:
                List_bcg_before.append(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_bcg_PVT[i])))
            else:
                List_bcg_after.append( List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_bcg_PVT[i])))
            output.write(List_result_PVT[i] + ',' + ', '.join(map(str, List_HRV_bcg_PVT[i])) + '\n')
            
# List_allHR_ecg = np.array(List_allHR_ecg)     
# List_allHR_bcg = np.array(List_allHR_bcg)   
if flag_total : 
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

    