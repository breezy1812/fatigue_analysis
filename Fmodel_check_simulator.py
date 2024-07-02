
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import datetime

from numpy.core.numeric import NaN
import src.get_PVT as PVT
from scipy.interpolate import interp1d
import src.get_PERCLOS as perclos
import src.get_HRV as HRV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# from src.LogisticRession import LogisticRegressionGD


def ar(alist):
    return np.array(alist)

def norm_HRlist_model(HR_list):
    HR_list_sorted = sorted(HR_list)
    norm_list = []
    for i in range(100):
        
        norm_list.append(HR_list_sorted[int(i * len(HR_list_sorted) / 100)])
    return np.array(norm_list)

def get_correlation(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)
    C1 = np.dot(x, x.conj()) 
    C2 = np.dot(y, y.conj()) 
    C3 = np.dot(x, y.conj()) 
    result = [0,0,0]
    for i in range(len(x)):
        result[0] += x[i]**2
        result[1] += y[i]**2
        result[2] += x[i]*y[i]

    cor = C3 / (C1 * C2) **0.5
    cor2 = result[2] / (result[0] * result[1]) **0.5
    return cor

def resample_HR(HR, status):
    HR_resmple = []
    temp = []
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
    return HR_resmple

def Fmodel_compute(List_all_HR):
    restingHR = (min(List_all_HR[int(60 * 2 / RESAMPLE_SIZE): int(60 * 10 / RESAMPLE_SIZE)]))
    d = restingHR - min(List_all_HR)
    d_ratio = round(d / restingHR, 4)
    List_all_HR = ar(sorted(List_all_HR))  
    personal_model = norm_HRlist_model(List_all_HR)
    y = np.arange(1,101)            
    R_value = get_correlation(ar(personal_model), y)
    fmodel_index = round(R_value + 0.004 * d, 3)

    return fmodel_index, d_ratio, personal_model

def datetime_convert(date_time): 
    format = '%Y-%m-%d-%H-%M' # The format 
    
    datetime_str = datetime.datetime.strptime(date_time, format) 
   
    return datetime_str 

# setting ====================================================

FLAG_SHOW_PLOT = True
FLAG_FILTER_STATUS = False # filter the unstable HR by the status
FLAG_USE_OUTSIDE_THRESHOLD = False
FLAG_PROTOTYPE = True #七月先行版 
FLAG_use_PERCLOS = False
use_BCG = 1 # 1: BCG, 0: ECG
use_KSS = True
window_size_min = 5
number_collect_fatigue = 3
RESAMPLE_SIZE = 30
GLOBAL_THRESHOLD_PERSENTAGE = 40
regression_mode = 1

# constant ====================================================
weedout_list = ['chao', 'ming', 'yongda']
DATA_DIR = "DATA_driving_combine" #"DATA_simulation" 
# DATA_DIR = "DATA_simulation"
# DATA_DIR = "DATA_6cycle"
DATA_DIR = "DATA_KSS_GD"
DATA_DIR_TEST = "DATA_KSS_GD" #"DATA_KSS_test"
subject_dirs = os.listdir(DATA_DIR)
# buffer ====================================================
list_result = []
total_scores = []
List_subject = []
list_time_gd = []
list_person_hr = []
list_threshold = []
list_KSS = []
list_gd_train = []
list_PERCLOS = []

if use_BCG == 1:
    print("train BCG as norHR")
else:
    print("train ECG as norHR")
print("resample size :" + str(RESAMPLE_SIZE))

for s_dir in subject_dirs:    
    if '.csv' in s_dir or 'total' in s_dir:
        continue  
    filepath = os.path.join(DATA_DIR, s_dir)      
    if not os.path.exists(os.path.join(filepath, 'ECG')):
        continue
    if s_dir in weedout_list:
        continue
    List_subject.append(s_dir)    
    filepath = os.path.join(DATA_DIR, s_dir)
    data_type_list = os.listdir(filepath)
    HRV_dir = os.path.join(filepath, 'HR_raw')
    PVT_dir = os.path.join(filepath, 'PVT')
    raw_dir = os.path.join(filepath, 'raw')
    
    if not os.path.exists(HRV_dir):
        print('there is no ' + HRV_dir)
        continue
    HRV_file_list = os.listdir(HRV_dir)
    raw_file_list = os.listdir(raw_dir)  

    '''parameters'''    
    List_all_HR = []  
    List_time = []
    List_HR = []

    List_SDNN = []
    List_DF = []
    List_RMSSD = []
    List_pNN50 = []
    List_PERCLOS_max_PVT = []

    for file in HRV_file_list:
        if use_BCG:            
            HR = []
            status = []
            confidence_level = []
            with open(os.path.join(HRV_dir, file), 'r') as HRfile:
                for line in HRfile.readlines():
                    line = line.split(',')
                    HR.append(float(line[0]))
                    status.append(float(line[1]))
                    confidence_level.append(int(line[2]))
        else:
            HR = []
            HR_t = []
            with open(os.path.join(HRV_dir, file), 'r') as HRfile:
                for line in HRfile.readlines():
                    element = line.split(',')
                    HR.append(float(element[1]))
                    HR_t.append(float(element[0]))

    
        
        if len(HR) == 0:
            continue
            
        
        '''resample'''
        if use_BCG:
            HR_resmple = resample_HR(HR, status)
            List_all_HR = [*List_all_HR, *HR_resmple]
        else:
            HR_resmple = []
            temp = []
            start = 0                
            for i in range(len(HR)):
                temp.append(HR[i])                    
                if HR_t[i] - start > RESAMPLE_SIZE:
                    HR_resmple.append(np.median(temp))
                    start = HR_t[i]
                    temp = []
            List_all_HR = [*List_all_HR, *HR_resmple]

        if len(List_all_HR) > 360 :
            List_all_HR = List_all_HR[:360]        

        start = 0
        starttime = datetime.datetime(1, 1, 1, 0, 0)
        currtime = datetime.datetime(1, 1, 1, 0, 0)
        delta_time = currtime - starttime
        while start < len(HR):   
            bpm = []
            sec = 0
            while start + sec < len(HR)-1:
                if use_BCG :
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
            rri = (60/np.array(bpm))  * 1000
            mean_HR = np.mean(bpm)  
            SDNN = np.std(rri)
            RMSSD = np.sqrt(np.mean((rri[1:] - rri[:-1])**2))
            DF = HRV.get_DF(bpm)           

            List_time.append(delta_time.seconds / 3600)
            List_HR.append(mean_HR)
            List_SDNN.append(SDNN)
            List_RMSSD.append(RMSSD)
            List_DF.append(DF)
            
            
        list_time_gd.append(List_time)
        if regression_mode == 3 :
            list_person_hr.append([List_HR, List_SDNN, List_RMSSD, List_DF])
        else:
            list_person_hr.append([List_HR, List_SDNN, List_RMSSD])

        '''Get KSS from PVT data'''
        flag_first = 1  
        related_threshold = 0
        ind = 0
        starttime = 0
        list_PVT_time = []
        list_PVT_result = []
        pvt_file_list = os.listdir(PVT_dir)

        
        
        for file in pvt_file_list:
            with open(os.path.join(PVT_dir, file), 'r') as line:
                pvt_data = line.read()
            #print('processing ', file)
            file = file.replace('PVT_', '')
            date_string = file.replace('.csv', '').split('_')[1] 
            currtime = datetime_convert(date_string)
            if starttime == 0:
                starttime = currtime
            curr_deltatime = currtime - starttime
            

            # if use_time == 0 or use_time == 1:
            list_PVT_time.append(curr_deltatime.seconds /3600)
            list_PVT_time.append((curr_deltatime.seconds + 600) / 3600)
            KSS1, KSS2 = PVT.get_KSS_text(pvt_data)
            list_PVT_result.append(KSS1)
            list_PVT_result.append(KSS2)            

            flag_first = 0
            ind += 1
        interpolation = interp1d(list_PVT_time, list_PVT_result, kind = 'linear')


        if List_time[0] < list_PVT_time[0]:
            List_time[0] = list_PVT_time[0]
        for i in range(len(List_time)):
            if List_time[i] > list_PVT_time[-1]:
                List_time[i] = list_PVT_time[-1]
        list_PVT_result_interp = interpolation(List_time)
        list_PVT_result_interp = [np.round(i,1) for i in list_PVT_result_interp]
        list_KSS.append(list_PVT_result_interp)

        '''generate golden serial for train'''
        gd_kss = [0] * len(list_PVT_result_interp)
        for i in range(1, len(gd_kss)):
            if list_PVT_result_interp[i] >= 7:
                gd_kss[i] = 1  
        list_gd_train.append(gd_kss)             


        '''model compute'''
        fmodel_index, d_ratio, personal_model = Fmodel_compute(List_all_HR)
        personal_threshold = personal_model[GLOBAL_THRESHOLD_PERSENTAGE]            
        list_threshold.append(personal_threshold)
        if fmodel_index > 1 and d_ratio > 0.1:
            result = (s_dir + '\t' + str(fmodel_index) + '\t' + str(d_ratio) + '\t Pass' + '\t' + str(personal_threshold) )
            
        else:
            result = (s_dir + '\t' + str(fmodel_index) + '\t' + str(d_ratio) + '\t ----' + '\t' + str(personal_threshold) )
            
        list_result.append(result)
        total_scores.append(fmodel_index + d_ratio*10)


    if FLAG_use_PERCLOS:
        FPS_ear = 14.7
        flag_first = 1
        PERCLOS_threshold = 0
        ear_filename = ''
        for file in raw_file_list:
            if 'avi' in file or 'mp4' in file:
                ear_filename = file

        ear_raw = []         
        
        #read EAR data
        with open(os.path.join(raw_dir, ear_filename), 'r', encoding='utf-8') as lines:
            
            for line in lines.readlines():
                line = "".join(filter(lambda ch: ch in '0123456789.', line))
                if line == '' :
                    continue
                ear_raw.append(float(line))      
            PERCLOS_threshold = perclos.get_threshold(ear_raw)
            
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
            else:
                break        
            ear = np.asarray(ear)
            List_PERCLOS_max_PVT.append(perclos.compute_PERCLOS_localmax(ear, PERCLOS_threshold, FPS_ear))
        list_PERCLOS.append(List_PERCLOS_max_PVT)


for _,x in sorted(zip(total_scores, list_result)):
    print(x)

'''training model'''
if FLAG_PROTOTYPE:
    scale_paras = len(list_person_hr[0])
    threshold_train_KSS = 4 
    X = []
    Y = []
    for i in range(len(list_gd_train)):
        for n in range(scale_paras):
            list_person_hr[i][n] = list_person_hr[i][n] / np.mean(list_person_hr[i][n][:3])
        for j in range(len(list_person_hr[i][0])):
            if list_KSS[i][j] > threshold_train_KSS and j * window_size_min > 30:
                x = []
                for n in range(scale_paras):
                    x.append(list_person_hr[i][n][j])
                X.append(x)
                Y.append(list_gd_train[i][j])
    X = np.array(X)
    Y = np.array(Y)

    section = np.linspace(1, 10, 10 )

    plt.figure()
    for i in range(scale_paras):
        plt.subplot(scale_paras,1,i+1)
        if i == 0:
            sect = ((section/5) + 8) / 10
        else:
            sect = section / 10


        hist_fatigue = [0] *10
        hist_nonfatigue = [0] *10    
        for j in range(len(Y)):
            if i == 0:
                A = int(((X[j][i] * 10) - 8) * 5)            
            else:
                A = int(np.floor(X[j][i] * 10))             
            
            A = 9 if X[j][i] > 1 else A
            A = 0 if A < 0 else A
            if Y[j] == 1:
                hist_fatigue[A] += 1
                # plt.plot(1,X[j][i], 'o', color = 'black', markersize=2)
            else:
                hist_nonfatigue[A] += 1
                # plt.plot(0,X[j][i], 'o', color = 'black', markersize=2)
        plt.plot(sect,hist_fatigue, color = 'red')
        plt.plot(sect,hist_nonfatigue, color = 'blue')
    

    if regression_mode == 1:
        lrGD=LogisticRegression(tol=0.001, max_iter=800, random_state=1)
        lrGD.fit(X,Y)
        train_interc = lrGD.intercept_[0] + 0.5
        train_parameters = lrGD.coef_[0] 
    elif regression_mode == 2:
        reg = LinearRegression().fit(X, Y)
        train_interc = reg.intercept_
        train_parameters = reg.coef_
    elif regression_mode == 3:
        pipe_svc = make_pipeline(StandardScaler(),
                          SVC(random_state=1))

        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

        param_grid = [
            {'svc__C': param_range, 
            'svc__gamma': param_range, 
            'svc__kernel': ['rbf']}]

        svm = GridSearchCV(
            estimator=pipe_svc, 
            param_grid=param_grid, 
            scoring='accuracy', 
            refit=True,
            cv=5,
            n_jobs=-1
            )
        svm = svm.fit(X, Y)
        train_parameters = svm.best_params_
        train_interc = 0

    print('parameters in modes is ' + ','.join(map(str, train_parameters)) + ',' + str(train_interc))

if FLAG_USE_OUTSIDE_THRESHOLD:
    filepath = os.path.join(DATA_DIR, 'threshold_mannual.csv')
    if os.path.exists(filepath):
        list_thrs_manu = []
        with open(filepath, 'r', encoding='utf-8-sig') as fp:
            while(1):
                temp = fp.readline()
                if temp == '':
                    break
                name = temp.split(',')[0]
                i  = List_subject.index(name)
                list_threshold[i] = float(temp.split(',')[1])
        
subject_dirs = os.listdir(DATA_DIR_TEST)
List_subject = []
list_time_gd = []
list_person_hr = []
list_threshold = []
list_KSS = []
list_gd_train = []
list_PERCLOS = []

for s_dir in subject_dirs:    
    if '.csv' in s_dir or 'total' in s_dir:
        continue      
    if s_dir in weedout_list:
        continue
    
    filepath = os.path.join(DATA_DIR_TEST, s_dir)         
    data_type_list = os.listdir(filepath)
    HRV_dir = os.path.join(filepath, 'HR_raw')
    PVT_dir = os.path.join(filepath, 'PVT')
    raw_dir = os.path.join(filepath, 'raw')
    
    if not os.path.exists(HRV_dir):
        print('there is no ' + HRV_dir)
        continue
    HRV_file_list = os.listdir(HRV_dir)
    raw_file_list = os.listdir(raw_dir)  
    List_subject.append(s_dir) 
    '''parameters'''    
    List_all_HR = []  
    List_time = []
    List_HR = []

    List_SDNN = []
    List_DF = []
    List_RMSSD = []
    List_pNN50 = []
    List_PERCLOS_max_PVT = []

    for file in HRV_file_list:
        if use_BCG:            
            HR = []
            status = []
            confidence_level = []
            with open(os.path.join(HRV_dir, file), 'r') as HRfile:
                for line in HRfile.readlines():
                    line = line.split(',')
                    HR.append(float(line[0]))
                    status.append(float(line[1]))
                    confidence_level.append(int(line[2]))
        else:
            HR = []
            HR_t = []
            with open(os.path.join(HRV_dir, file), 'r') as HRfile:
                for line in HRfile.readlines():
                    element = line.split(',')
                    HR.append(float(element[1]))
                    HR_t.append(float(element[0]))

            
        if len(HR) == 0:
            continue
            
        
        '''resample'''
        if use_BCG:
            HR_resmple = resample_HR(HR, status)
            List_all_HR = [*List_all_HR, *HR_resmple]
        else:
            HR_resmple = []
            temp = []
            start = 0                
            for i in range(len(HR)):
                temp.append(HR[i])                    
                if HR_t[i] - start > RESAMPLE_SIZE:
                    HR_resmple.append(np.median(temp))
                    start = HR_t[i]
                    temp = []
            List_all_HR = [*List_all_HR, *HR_resmple]

        if len(List_all_HR) > 360 :
            List_all_HR = List_all_HR[:360]        

        start = 0
        starttime = datetime.datetime(1, 1, 1, 0, 0)
        currtime = datetime.datetime(1, 1, 1, 0, 0)
        delta_time = currtime - starttime
        while start < len(HR):   
            bpm = []
            sec = 0
            while start + sec < len(HR)-1:
                if use_BCG :
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
            rri = (60/np.array(bpm))  * 1000
            mean_HR = np.mean(bpm)  
            SDNN = np.std(rri)
            RMSSD = np.sqrt(np.mean((rri[1:] - rri[:-1])**2))
            DF = HRV.get_DF(bpm)           

            List_time.append(delta_time.seconds / 3600)
            List_HR.append(mean_HR)
            List_SDNN.append(SDNN)
            List_RMSSD.append(RMSSD)
            List_DF.append(DF)
            
            
        list_time_gd.append(List_time)
        if regression_mode == 3 :
            list_person_hr.append([List_HR, List_SDNN, List_RMSSD, List_DF])
        else:
            list_person_hr.append([List_HR, List_SDNN, List_RMSSD])

        '''Get KSS from PVT data'''
        flag_first = 1  
        related_threshold = 0
        ind = 0
        starttime = 0
        list_PVT_time = []
        list_PVT_result = []
        pvt_file_list = os.listdir(PVT_dir)        
        
        for file in pvt_file_list:
            with open(os.path.join(PVT_dir, file), 'r') as line:
                pvt_data = line.read()
            #print('processing ', file)
            file = file.replace('PVT_', '')
            date_string = file.replace('.csv', '').split('_')[1] 
            currtime = datetime_convert(date_string)
            if starttime == 0:
                starttime = currtime
            curr_deltatime = currtime - starttime
            

            # if use_time == 0 or use_time == 1:
            list_PVT_time.append(curr_deltatime.seconds /3600)
            list_PVT_time.append((curr_deltatime.seconds + 600) / 3600)
            KSS1, KSS2 = PVT.get_KSS_text(pvt_data)
            list_PVT_result.append(KSS1)
            list_PVT_result.append(KSS2)            

            flag_first = 0
            ind += 1
        interpolation = interp1d(list_PVT_time, list_PVT_result, kind = 'linear')


        if List_time[0] < list_PVT_time[0]:
            List_time[0] = list_PVT_time[0]
        for i in range(len(List_time)):
            if List_time[i] > list_PVT_time[-1]:
                List_time[i] = list_PVT_time[-1]
        list_PVT_result_interp = interpolation(List_time)
        list_PVT_result_interp = [np.round(i,1) for i in list_PVT_result_interp]
        list_KSS.append(list_PVT_result_interp)

        '''generate golden serial for train'''
        gd_kss = [0] * len(list_PVT_result_interp)
        for i in range(1, len(gd_kss)):
            if list_PVT_result_interp[i] >= 7:
                gd_kss[i] = 1  
        list_gd_train.append(gd_kss)             


    if FLAG_use_PERCLOS:
        FPS_ear = 14.7
        flag_first = 1
        PERCLOS_threshold = 0
        ear_filename = ''
        for file in raw_file_list:
            if 'avi' in file or 'mp4' in file:
                ear_filename = file

        ear_raw = []         
        
        #read EAR data
        with open(os.path.join(raw_dir, ear_filename), 'r', encoding='utf-8') as lines:
            
            for line in lines.readlines():
                line = "".join(filter(lambda ch: ch in '0123456789.', line))
                if line == '' :
                    continue
                ear_raw.append(float(line))      
            PERCLOS_threshold = perclos.get_threshold(ear_raw)
            
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
            else:
                break        
            ear = np.asarray(ear)
            List_PERCLOS_max_PVT.append(perclos.compute_PERCLOS_localmax(ear, PERCLOS_threshold, FPS_ear))
        list_PERCLOS.append(List_PERCLOS_max_PVT)
if FLAG_PROTOTYPE:
    scale_paras = len(list_person_hr[0])
    for i in range(len(list_gd_train)):
        for n in range(scale_paras):
            list_person_hr[i][n] = list_person_hr[i][n] / np.mean(list_person_hr[i][n][:3])


if FLAG_SHOW_PLOT:
    list_count_checkpoint = []
    fig_all = plt.figure()
    plt.title('checkpoint_distribution', figure = fig_all)
    
    list_sens = []
    list_sepc = []
    list_fp = []
    list_delay_time = []
    serial_KSS_count = [0] * 11
    serial_FP_count  = [0] * 11
    serial_TP_count  = [0] * 11
    Totao_C = [0] * 4
    for i in range(len(List_subject)):
        
        list_bool_gd = list_gd_train[i]
        list_alert_stage = []
        list_FI_view = []
        C_TP = 0
        C_FN = 0
        C_FP = 0
        C_TN = 0
        
        C_control = 1
        alarm_count = 0
        time_fatigue_gd = list_time_gd[i][-1] + 1
        time_fatigue_pd = list_time_gd[i][-1] + 1
        
        if FLAG_PROTOTYPE: 
            HR_serial = list_person_hr[i]
        else:
            threshold = list_threshold[i]
            HR_serial = list_person_hr[i][0]

        
        for j in range(len(list_time_gd[i])): 
             
            fatigue_condition = 0
            serial_KSS_count[int(list_KSS[i][j])] += 1
            isfatigue = 0

            '''check fatigue state'''
            if FLAG_PROTOTYPE: 
                if  regression_mode == 3:
                    temp = [HR_serial[k][j] for k in range(scale_paras)]
                    isfatigue = svm.predict([temp])[0] > 0.5
                    number_collect_fatigue = 1
                else:
                    for k in range(scale_paras):                    
                        isfatigue += HR_serial[k][j] * train_parameters[k]                
                    isfatigue += train_interc
                    list_FI_view.append(isfatigue)
                    if regression_mode == 1:
                        isfatigue = isfatigue >= 0.1
                        number_collect_fatigue = 4
                    elif regression_mode == 2:
                        isfatigue = isfatigue > 0.5
                        number_collect_fatigue = 4
            else:
                isfatigue = HR_serial[j] <= threshold
            
            if FLAG_use_PERCLOS:
                number_collect_fatigue = 1
                threshold = 0.15
                if j < len(list_PERCLOS[i]):
                    isfatigue = list_PERCLOS[i][j] > threshold
                else:
                    isfatigue = False


            '''there were no alarm in 1 hour after start driving '''
            if j * window_size_min <= 60:
                isfatigue = False

            '''check alarm'''
            if isfatigue:
                alarm_count += 1
                if alarm_count >= number_collect_fatigue and j * window_size_min > 30:
                    fatigue_condition = 1                                      
            else:
                alarm_count = 0
            list_alert_stage.append(fatigue_condition)

            
            if  j >= len(list_time_gd[i]):
                continue
            '''if TP happend, stop counting FP TP and FN'''
            if C_TP == 1: 
                C_control = 0   

            '''check performance'''
            if fatigue_condition:
                if list_KSS[i][j] >= 7:
                    time_fatigue_pd = min(list_time_gd[i][j], time_fatigue_pd)  
                    serial_TP_count[int(np.floor(list_KSS[i][j]))] += 1       
                    C_TP += C_control
                    Totao_C[0] += 1
                    if list_KSS[i][j] >= 8 :
                        time_fatigue_gd = min(list_time_gd[i][j], time_fatigue_gd)
                elif list_KSS[i][j] < 6:                    
                    serial_FP_count[int(np.floor(list_KSS[i][j]))] += 1
                    C_FP += 1
                else:
                    time_fatigue_pd = min(list_time_gd[i][j], time_fatigue_pd) # KSS > 6 start refer as the previous alerm
                    
            else:
                if j < len(list_KSS[i]) - 1 :
                    if list_KSS[i][j] >= 8 and list_KSS[i][j + 1] >= 8 :
                        # During the extra testing interval, if the participants provide a self-assessment above or 
                        # equal to the drowsiness threshold again, the reading shall be treated as a false negative
                        time_fatigue_gd = min(list_time_gd[i][j], time_fatigue_gd)
                        C_FN += C_control
                        Totao_C[1] += 1
                    else:
                        C_TN += 1

                else:
                    if list_KSS[i][j] < 6:                        
                        C_TN += 1





            # if list_KSS[i][j] >= 7 and fatigue_condition:    
            #     # if the participant is at a KSS level of 7 or above
            #     if list_KSS[i][j] >= 8 :
            #         time_fatigue_gd = min(list_time_gd[i][j], time_fatigue_gd)
            #     time_fatigue_pd = min(list_time_gd[i][j], time_fatigue_pd)  
            #     serial_TP_count[int(np.floor(list_KSS[i][j]))] += 1       
            #     C_TP += C_control
            # elif j < len(list_KSS[i]) - 1 and j * window_size_min > 30 :                
            #     if list_KSS[i][j] >= 8 : 
                    
            #         time_fatigue_gd = min(list_time_gd[i][j], time_fatigue_gd)
            #         C_FN += C_control
            #     elif fatigue_condition :
                    
            #         if list_KSS[i][j] < 6:
            #             serial_FP_count[int(np.floor(list_KSS[i][j]))] += 1
            #             C_FP += C_control
            #         else:
            #             time_fatigue_pd = min(list_time_gd[i][j], time_fatigue_pd) # KSS > 6 start refer as the previous alerm
            #             serial_TP_count[int(np.floor(list_KSS[i][j]))] += 1 
        
        if  time_fatigue_gd < list_time_gd[i][-1] + 1 and time_fatigue_pd < list_time_gd[i][-1] + 1:
            deltatime_fatigue = (time_fatigue_gd - time_fatigue_pd) * 60
            list_delay_time.append(deltatime_fatigue)
            list_fp.append(C_FP)
        elif time_fatigue_pd < list_time_gd[i][-1] + 1:
            deltatime_fatigue = (list_time_gd[i][-1] - time_fatigue_pd) * 60
            list_delay_time.append(deltatime_fatigue)
            list_fp.append(C_FP)
        elif time_fatigue_gd < list_time_gd[i][-1] + 1 :
            deltatime_fatigue = (time_fatigue_gd - list_time_gd[i][-1]) * 60
            list_delay_time.append(deltatime_fatigue)
            list_fp.append(C_FP)
        else:
            deltatime_fatigue = NaN

        KSS_at_time_fatigue_pd = list_KSS[i]        
        
        if C_TP + C_FN > 0:
            print( List_subject[i] + '\t' + ",sensitivity," + str(C_TP / (C_TP + C_FN)) + ",delay time," + str(deltatime_fatigue))
            list_sens.append(C_TP / (C_TP + C_FN))
        else:
            print( List_subject[i] + '\t' + ",sensitivity," + "null" + ",delay time," + str(deltatime_fatigue))
        
        if C_TN + C_FP > 0:
            list_sepc.append(C_TN / (C_TN + C_FP))
            
        
        
        Totao_C[2] += C_TN
        Totao_C[3] += C_FP
        
        if use_KSS:
            gd = ar(list_KSS[i])
            alert = ar(list_KSS[i])[ar(list_alert_stage) == 1]
        else:
            gd = ar(list_time_gd[i])#[ar(list_time_gd[i]) >= 1]
            alert = ar(list_time_gd[i])[ ar(list_alert_stage) == 1]
    
        list_count_checkpoint.append(len(list_time_gd[i]))
        x = [i+1] * len(gd)
        x = [List_subject[i]] * len(gd)
        
        
        plt.plot(x,gd, 's', color = 'black', figure = fig_all)
        if len(alert) > 0:            
            #plt.plot([i+1] * len(alert), alert, 's', color = 'red')
            plt.plot([List_subject[i]] * len(alert), alert, 's', color = 'red', figure = fig_all)



    for i in range(len(serial_FP_count)):
        serial_FP_count[i] = serial_FP_count[i] / serial_KSS_count[i] if serial_KSS_count[i] > 0 else 0
        serial_TP_count[i] = serial_TP_count[i] / serial_KSS_count[i] if serial_KSS_count[i] > 0 else 0

    print('mesn of sensitivity = ' + str(np.mean(list_sens)))
    cor_sense = np.mean(list_sens) - 1.645 * np.sqrt(np.std(list_sens) / len(list_sens))
    print('mesn2 of sensitivity = ' + str(cor_sense))
    print('testing sensitivity = ' + str(Totao_C[0] / (Totao_C[0]+Totao_C[1])))
    print('testing specificity = ' + str(Totao_C[2] / (Totao_C[2]+Totao_C[3])))
    print('mean number of false positive = ' + str(np.mean(list_fp)))
    print('Golden Probability = ' + ','.join(map(str, serial_KSS_count)))
    print('FP Probability = ' + ','.join(map(str, serial_FP_count)))
    print('TP Probability = ' + ','.join(map(str, serial_TP_count)))
    print('mean of delay time = ' + str(np.mean(list_delay_time)))
    plt.xticks(rotation=45)
        
    if regression_mode == 1:
        sub = 8
        HR_serial = list_person_hr[sub]
        fig_one = plt.figure()
        plt.title('fatigue_history', figure = fig_one)
        X = list_time_gd[sub]
        for i in range(3):
            temp = abs(HR_serial[i] * train_parameters[i])
            temp = temp - temp[1]
            temp = temp / max(temp)
            plt.plot(X,temp , 's', figure = fig_one)
        # for k in range(scale_paras):                    
        #     isfatigue += HR_serial[k] * train_parameters[k]
        # plt.plot(X,isfatigue , 's', figure = fig_one)


    # plt.figure()
    # plt.title('checkpoint_distribution')
    # count = 0
    # list_sens = []
    # for i in range(len(List_subject)):
    #     threshold = list_threshold[i]
    #     list_alert_stage = []

    #     if use_KSS:
    #         gd = ar(list_KSS[i])
    #     else:
    #         gd = ar(list_time_gd[i])
        
    #     HR_serial = list_person_hr[i][0]
    #     SDNN_serial = list_person_hr[i][1]
    #     DF_serial = list_person_hr[i][2]
    #     ref_HR = np.mean(HR_serial[:3])
    #     # plt.plot(gd, HR_serial / ref_HR)
    #     plt.plot(gd, SDNN_serial,'o')
    #     # plt.plot(gd, DF_serial,'o')


    plt.show()    
    plt.close()   
    