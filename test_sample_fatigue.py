from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import os
from src.alg_fatigue import Alg_fatigue
from src.alg_freq_domain import Alg_freq_domain
import datetime
import matplotlib.dates as mdates
import python.src.read_data as rd

# constant ====================================================
# weedout_list = ['chao', 'ming', 'yongda']
regression_mode = 2
window_size_min = 5
data_from = 0  # 0:MCU  1:python 2:ECG
# DATA_DIR_TEST = "DATA_KSS_GD" #"DATA_KSS_test"
parafile_path = "src//parameters1.txt"
sample_rate = 64
down_scale = 1
FFT_window_size=32

filepath = "testdata/202208_DDAW_validation"
filename = "2022_09_26_19_05_55(主駕).log"

ground_truth_filename = filename.replace('.log', '.csv')
if '.txt' in filename:
    ground_truth_filename = filename.replace('.txt', '.csv')

# buffer ====================================================

def main():
    scale_paras = 4 if regression_mode == 3 else 3
    pressure_data, acc_x, acc_y, acc_z, raw_status, raw_hr,timestamp, start_time = rd.read_pressure_acc_result(os.path.join(filepath, "raw"), filename, down_scale)

    alg_ft_test = Alg_fatigue(parafile_path)
    algo_HeartR = []
    algo_status = []
    algo_time = []
    delta_time = 0
    
    ## downsample from raw data whose samplinrate 64 to 1 Hz
    if data_from == 0:
        for i in range(len(raw_hr)):
            if i % sample_rate == 0 :
                delta_time += 1
                algo_HeartR.append(raw_hr[i])
                algo_status.append(raw_status[i])
                algo_time.append(delta_time / 3600)
    elif data_from == 1 :
        algo = Alg_freq_domain(fs=sample_rate, fft_window_size=FFT_window_size)
        time = np.linspace(0, len(pressure_data) / sample_rate, len(pressure_data))
        for i in range(len(acc_y)):
            acc_y[i] = 65535 - acc_y[i] if  acc_y[i]> 50000 else acc_y[i]
            acc_z[i] = 65535 - acc_z[i] if  acc_z[i]> 50000 else acc_z[i]
            acc_x[i] = 65535 - acc_x[i] if  acc_x[i]> 50000 else acc_x[i]
        algo.main_func(pressure_data, acc_x, acc_y, acc_z)
        algo_HeartR = algo.bpm
        algo_status = algo.status
        algo_time = [i/ 3600 for i in range(len(algo_HeartR))]
    elif data_from == 2 :


        if os.path.isfile(os.path.join(filepath, "ground_truth_bpm", ground_truth_filename)):
            algo_HeartR, start_time = rd.read_csv_with_time(os.path.join(filepath, "ground_truth_bpm", ground_truth_filename))
    
        
    ## get KSS 
    KSS_time = []
    KSS_score = []
    with open(os.path.join(filepath, "KSS", ground_truth_filename), 'r', encoding='utf-8') as kssfile:
        time = kssfile.readline()

        if "time" in time:
            time = time.split(",")
            start_time.extend(list(map(int, time[1:])))
        elif 'Start_Time' in time:
            time = time.split(':')[1]
            start_time.extend(list(map(int, time.split('-'))))      
              
        for line in kssfile.readlines():   
            cont = line.split(",")
            if len(cont) < 2:
                break         
            KSS_score.append(int(cont[1]))
            time = cont[0].split(":")
            curr_time = np.copy(start_time)
            curr_time[3] = time[0]
            curr_time[4] = time[1]
            KSS_time.append(datetime.datetime(int(curr_time[0]), int(curr_time[1]), int(curr_time[2]), int(curr_time[3]), int(curr_time[4])))
    KSS_score = np.array(KSS_score)
    KSS_time = np.array(KSS_time)


    ## compute fatigue index and alarm
    list_HR, list_SDNN, list_RMSSD, list_DF, list_time_gd = alg_ft_test.get_HRV_list(algo_HeartR, algo_status, window_size_min)
    if regression_mode == 3 :
        list_person_hr_test = [list_HR, list_SDNN, list_RMSSD, list_DF]
    else:
        list_person_hr_test = [list_HR, list_SDNN, list_RMSSD]
    
    passornot, list_person_hr_test, ind_learn_successful = alg_ft_test.dsp_learn_vigor(list_person_hr_test)
    if not passornot:
        print("fatigue learning false!")
        return
        
    
    testing_X = []
    for j in range(len(list_person_hr_test[0])):        
        x = []
        for n in range(scale_paras):
            x.append(list_person_hr_test[n][j])
        testing_X.append(x)

    list_FI_view, list_alert_stage = alg_ft_test.testing_model_fatigue(testing_X, list_time_gd, start_alarm = ind_learn_successful)
    
    ## generate the datetime serial
    start_time = datetime.datetime(int(start_time[0]), int(start_time[1]), int(start_time[2]), int(start_time[3]), int(start_time[4]))
    time_list = []
    for time in list_time_gd:
        time = start_time + datetime.timedelta(hours = time)
        time_list.append(time) 
        
    # compute the sensitivity  
    TP = 0
    FN = 0
    ind = 0
    last_detect = 0
    time_diff = datetime.timedelta(hours = 0)
    for i in range(len(KSS_score)):
        while ind < len(list_alert_stage)-1:
            if time_list[ind + 1] > KSS_time[i]:
                detect_result = list_alert_stage[last_detect : ind]
                break
            else:
                ind += 1

        if KSS_score[i] >= 8:
            if 1 in detect_result:
                alarm_ind = list_alert_stage.index(1)
                time_diff = time_list[alarm_ind] - KSS_time[KSS_score == 8][0]
                TP = 1
                break
            else:
                FN += 1
        elif KSS_score[i] == 7:
            if 1 in detect_result:
                alarm_ind = list_alert_stage.index(1)
                time_diff = time_list[alarm_ind] - KSS_time[KSS_score == max(KSS_score)][0]
                TP = 1
                break
    if (FN + TP) > 0:
        print("sensitivity = " + str(int(100 * TP / (FN+TP))) + "%")
        print("delay  " + str(int(time_diff.total_seconds() / 60)) + "  minutes")
        
    with open(os.path.join(filepath, "fatigue", ground_truth_filename), 'w', encoding='utf-8') as fatiguiefile:
        for i in range(len(time_list)):
            line = time_list[i].strftime("%H:%M:%S")
            line += "," + str(list_FI_view[i])
            line += "," + str(list_alert_stage[i]) + '\n'
            fatiguiefile.writelines(line)
    
        
    plt.title(filename)
    ax1 = plt.subplot(2,1,1)
    plt.ylabel("Fatigue index")
    ax2 = ax1.twinx()
    plt.title("Fatigue Index & Alert Status")
    plt.xlim(KSS_time[0], KSS_time[-1])
    # list_para_view = 1 - list_person_hr_test[1]
    list_alert_stage = np.array(list_alert_stage)
    time_list = np.array(time_list)
    list_FI_view = np.array(list_FI_view)
    L1 = ax1.plot(time_list[list_alert_stage == 1], list_FI_view[list_alert_stage == 1], 's', color = 'red', label = "FI with alarm")
    L2 = ax1.plot(time_list[list_alert_stage == 0], list_FI_view[list_alert_stage == 0], 's', color = 'black', label = "FI without alarm")
    L0 = ax1.plot(time_list[:ind_learn_successful + 1], list_FI_view[:ind_learn_successful + 1], 's', color = 'gray', label = "FI within learning")
    plt.legend(handles = L0 + L1 + L2, loc='upper center', fontsize = 8)
    
    # L3 = ax2.plot(time_list, list_HR, 'o', color = 'blue', label = "Heart Rate")
    # ax2.set_ylim(60,80)
    # ax2.ylabel("Heart rate")
    # plt.legend(handles = L0 + L1 + L2 + L3, loc='upper center', fontsize = 8)

    
    plot2 = plt.subplot(2,1,2)

    plot2.set_xlabel("Date time")
    plot2.set_ylabel("KSS score")
    for i in range(len(KSS_score)):
        if i > 0:
            x1 = [KSS_time[i-1], KSS_time[i]]
            y1 = np.array([KSS_score[i] - 0.5, KSS_score[i]- 0.5])
            y2 = KSS_score[i] + 0.5
            
            
            if KSS_score[i] >= 8 :
                plot2.fill_between(x1, y1, y2=y2, color='red', alpha=0.6)
            elif KSS_score[i] == 7 :
                plot2.fill_between(x1, y1, y2=y2, color='orange', alpha=0.6)
            else:
                plot2.fill_between(x1, y1, y2=y2, color='green', alpha=0.6)
    
    plot2.set_ylim(3,8)
    plot2.set_xlim(KSS_time[0], KSS_time[-1])
    plt.show()    
    plt.close()  
    
    
    
            

if __name__ == '__main__':
    main()
    