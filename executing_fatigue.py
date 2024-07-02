
import matplotlib.pyplot as plt
import numpy as np
import os
from src.alg_fatigue import Alg_fatigue
import datetime
import matplotlib.dates as mdates

# setting ====================================================
use_BCG = 1 # 1: BCG, 0: ECG
window_size_min = 5
RESAMPLE_SIZE = 30
GLOBAL_THRESHOLD_PERSENTAGE = 40
regression_mode = 2
KSS_min_traning = 2

# constant ====================================================
weedout_list = ['chao', 'ming', 'yongda']
DATA_DIR = "DATA_driving_combine" #"DATA_simulation" 
DATA_DIR = "DATA_KSS_GD"
DATA_DIR_TEST = "DATA_KSS_GD" #"DATA_KSS_test"
parafile_path = "src//parameters.txt"

# buffer ====================================================
list_result = []
total_scores = []
List_subject = []
List_gd_train = []
List_time_gd = []
List_person_hr = []
List_person_hr_norm = []
list_threshold = []
list_PERCLOS = []

def main():

    print('************************************************************')
    print('\ttraining data' + DATA_DIR)
    print('\ttesting data' + DATA_DIR_TEST)
    print('************************************************************')
    subject_dirs = os.listdir(DATA_DIR)
    scale_paras = 4 if regression_mode == 3 else 3
    alg_ft = Alg_fatigue()
    for s_dir in subject_dirs:    
        if '.csv' in s_dir or 'total' in s_dir:
            continue  
        filepath = os.path.join(DATA_DIR, s_dir)      
        if not os.path.exists(os.path.join(filepath, 'ECG')):
            continue
        if s_dir in weedout_list:
            continue
        List_subject.append(s_dir) 

        data_type_list = os.listdir(filepath)
        HRV_dir = os.path.join(filepath, 'HR_raw')
        PVT_dir = os.path.join(filepath, 'PVT')
        
        if not os.path.exists(HRV_dir):
            print('there is no ' + HRV_dir)
            return

        
        HR, status, confidence_level = alg_ft.get_HR_list(HRV_dir)
        list_HR, list_SDNN, list_RMSSD, list_DF, time_list = alg_ft.get_HRV_list(HR, status, window_size_min)
        if regression_mode == 3 :
            list_person_hr_test = [list_HR, list_SDNN, list_RMSSD, list_DF]
        else:
            list_person_hr_test = [list_HR, list_SDNN, list_RMSSD]
        List_person_hr.append(list_person_hr_test)
        for n in range(scale_paras):
            passornot, list_person_hr_test[n] = alg_ft.get_normalization_paras(list_person_hr_test[n])
            if not passornot:
                print("fatigue learning false!")
                return

        List_person_hr_norm.append(list_person_hr_test)        
        list_kss = alg_ft.get_KSS_from_PVT(PVT_dir)
        List_gd_train.append(list_kss)
        List_time_gd.append(time_list)

    X = []
    Y = []
    for i in range(len(List_gd_train)):
        for j in range(len(List_person_hr_norm[i][0])):
            x = []
            for n in range(scale_paras):
                x.append(List_person_hr_norm[i][n][j])
            if List_gd_train[i][j] >= 6.5:
                Y.append(1)
            else:
                Y.append(0)
            X.append(x)
                
    X = np.array(X)
    Y = np.array(Y)
    print("ratio of Y data = " + str(np.mean(Y)))
    alg_ft.training_model_fatigue(X, Y, regression_mode)
    alg_ft.save_model_file(parafile_path)
    list_FI_view, list_alert_stage = alg_ft.testing_model_fatigue(X)

    list_testing_result = [1 if i > 0.5 else 0 for i in list_FI_view]
    print(Y)
    print(np.array(list_testing_result))
    correct_count = 0
    for i in range(len(Y)):
        if list_testing_result[i] == Y[i]:
            correct_count += 1
    print("percision = " + str(correct_count / len(Y)))

    print("=======test one sample===========================================")
    subject = "mads"
    subject_ind = List_subject.index(subject)
    print("testing " + List_subject[subject_ind])
    alg_ft_test = Alg_fatigue("src//parameters.txt")
    testing_X = []
    kss = List_gd_train[subject_ind]
    raw_dir = os.path.join(os.path.join(DATA_DIR, subject), 'raw')
    start_time = alg_ft_test.get_time_from_raw(raw_dir)
    HR_list = List_person_hr[subject_ind][2]
    for j in range(len(List_person_hr_norm[subject_ind][0])):        
        x = []
        for n in range(scale_paras):
            x.append(List_person_hr_norm[subject_ind][n][j])
        testing_X.append(x)
    list_FI_view, list_alert_stage = alg_ft_test.testing_model_fatigue(testing_X, List_time_gd[subject_ind])
    list_para_view = 1 - List_person_hr_norm[subject_ind][1]
    list_para_view2 = []  

    time_list = []
    for time in List_time_gd[subject_ind]:
        time = start_time + datetime.timedelta(hours = time)
        time_list.append(time)

    for i in range(len(list_para_view)):
        list_para_view2.append(np.mean(list_para_view[i-2:i+3])) 
    print(list_alert_stage)
    plt.title('fatigue_plot_' + List_subject[subject_ind])
    plt.subplot(2,1,1)
    plt.title("Fatigue Index")
    for i in range(len(list_para_view)):
        if list_alert_stage[i] == 1:
            plt.plot([time_list[i]], [list_FI_view[i]], 's', color = 'red')
        else:
            plt.plot([time_list[i]], [list_FI_view[i]], 's', color = 'black')
    plt.subplot(2,1,2)
    plt.tight_layout()
    plt.title("Heart Rate")
    plt.plot(time_list, HR_list, 's', color = 'black')



    #=======test all samples============================================
    subject_dirs = os.listdir(DATA_DIR_TEST)
    List_subject_test = []
    alg_ft_test = Alg_fatigue("src//parameters.txt")
    fig_all = plt.figure()
    plt.title('checkpoint_distribution', figure = fig_all)
    for s_dir in subject_dirs:          
        if '.csv' in s_dir or 'total' in s_dir:
            continue  
        filepath = os.path.join(DATA_DIR, s_dir)      

        if s_dir in weedout_list:
            continue
        List_subject_test.append(s_dir) 

        HRV_dir = os.path.join(filepath, 'HR_raw')
        raw_dir = os.path.join(filepath, 'raw')
        PVT_dir = os.path.join(filepath, 'PVT')
        
        if not os.path.exists(HRV_dir):
            print('there is no ' + HRV_dir)
            continue
        HR, status, confidence_level = alg_ft_test.get_HR_list(HRV_dir)
        start_time = alg_ft_test.get_time_from_raw(raw_dir)
        list_HR, list_SDNN, list_RMSSD, list_DF, list_time_gd = alg_ft_test.get_HRV_list(HR, status, window_size_min)
        if regression_mode == 3 :
            list_person_hr_test = [list_HR, list_SDNN, list_RMSSD, list_DF]
        else:
            list_person_hr_test = [list_HR, list_SDNN, list_RMSSD]
        
        for n in range(scale_paras):
            passornot, list_person_hr_test[n] = alg_ft_test.get_normalization_paras(list_person_hr_test[n])
            if not passornot:
                break
        list_kss = alg_ft_test.get_KSS_from_PVT(PVT_dir)
        testing_X = []
        for j in range(len(list_person_hr_test[0])):        
            x = []
            for n in range(scale_paras):
                x.append(list_person_hr_test[n][j])
            testing_X.append(x)

        list_FI_view, list_alert_stage = alg_ft_test.testing_model_fatigue(testing_X, list_time_gd)

        x = [s_dir] * len(list_alert_stage)
        for i in range(len(list_FI_view)):
            if list_alert_stage[i] == 1:
                plt.plot(s_dir,list_kss[i] , 's', color = 'red', figure = fig_all)
            else:
                plt.plot(s_dir,list_kss[i] , 's', color = 'black', figure = fig_all)
    plt.xticks(rotation=45, figure = fig_all)


    plt.show()    
    plt.close()   
if __name__ == '__main__':
    main()
    