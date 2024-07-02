import numpy as np
import os
import sys
import time
import datetime
import time
import json
import requests
import bcrypt
import random

CURRENT_PATH = os.getcwd() + "/DATA_KSS_GD"
url = "http://192.168.0.86:8811/"

url_bcg = url +"bcgs"
url_algo = url +"algos"
headers = {"Authorization": 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJmcmVzaCI6dHJ1ZSwiaWF0IjoxNjMwOTEyNjc5LCJqdGkiOiIwN2M1MjQ1My1lOTQ0LTQ3MTYtYjYwYi1iOTA5YTAwNDQ3NzkiLCJ0eXBlIjoiYWNjZXNzIiwic3ViIjoiYmlvbG9ndWUiLCJuYmYiOjE2MzA5MTI2NzksImV4cCI6MTYzODY4ODY3OX0.4v8a_rAlhMX51S-zxmigZ8U2fKPXAZ-rCa9B4Mo5rhU'}
Sdatetime = datetime.datetime(2021,10,11,12,0,0)
userId = "6167db1dff0f5a5b47d24c7d"
carId = "615d49845d420bdd51eaad72"

algo_fields = {
    "userid": userId,
    "carid": carId,
    "hwversion": "0.0.0",
    "swversion": "0.0.0",
    "timestamp": time.time(),
    "timestampm":0,
    "fatigue": 0,
    "hr": [],
    "confidence": [],
    "resp": [],
    "status": []
}


bcg_fields = {
    'userid': userId,
    'carid': carId,
    "bcg": [],
    "datalen": 0,
    "accx": [],
    "accy": [],
    "accz": [],
    "device": 0,
    "ecgid": "0",
    "algoidlist": [],
    "algoidlistlen": 0,
    "timestampst": 0,
    "timestampend": 0,
    "timestamp": 0
}

# GEN_USER_TO_DB = ["danny"]
# GEN_USER_TO_DB = ["bo", "danny", "howard"]
GEN_USER_TO_DB = ["bo", "danny", "howard", "jerry", "jimmy", "kim", "mads", "shu", "roxus", "yong", "zhang"]
SEARCH_TIMESTAMP = {}

BCG_FS = 64.0
BCG_PERIOD = 1 / BCG_FS

BCG_SAVING_CNTS = BCG_FS*60*1  # 1 minute

FILES_CODE = 0
BCG_FILES_CODE = 1
ECG_FILES_CODE = 2

# BCG format
# - 0: X
# - 1: data
# - 2: accx
# - 3: accy
# - 4: accz
# - 5: HR
# - 6: Resp
# - 7: Status

# ECG format
# - 0: X
# - 1: X
# - 2: X
# - 3: X
# - 4: X
# - 5: ECG
# - 6: X

def remove_last_newLine(list_data):
    if "\n" in list_data[len(list_data)-1]:
        list_data[len(list_data)-1] = list_data[len(list_data)-1].replace("\n", "")


for user in GEN_USER_TO_DB:
    Sdatetime += datetime.timedelta(days=1)
    USER_ID = None
    CARINFO_ID = None
    print(f"===== Read the User and CarInfo from database =====")
    

    print(f"===== Write to database --> Algo[{user}] =====")
    raw_data_path = CURRENT_PATH + "/" + user + "/" + "raw"
    for directory, _x, files in os.walk(raw_data_path):
        for f in files:
            if ".log" in f and "ECG" not in f:
                print(f"Write the file to DB [{f}]")
                dataF = open(directory + "/" + f, "r")
                raw_data = dataF.readlines()
                cnt = 0
                data_cnt = 0
                id_cnt = 0
                unix_time = None
                ecg = []
                bcg_id = []
                bcg = []
                accx = []
                accy = []
                accz = []
                t_st = 0
                for x in raw_data:
                    if "Start Time" in x:
                        temp = x.split(" ")
                        remove_last_newLine(temp)
                        unix_time = datetime.datetime.strptime(temp[2],"%Y/%m/%d-%H:%M:%S:%f").timestamp()*1000
                        if "ecg" in f.lower():
                            FILES_CODE = ECG_FILES_CODE
                            print("ECG file detect")
                        else:
                            FILES_CODE = BCG_FILES_CODE
                            print("BCG file detect")
                        print("start time: ", unix_time)
                        SEARCH_TIMESTAMP[user] = unix_time
                        cnt = 0
                        data_cnt = 0
                        id_cnt = 0
                    elif "State Change" in x:
                        print("State change")
                    elif "End Time" in x:
                        temp = x.split(" ")
                        remove_last_newLine(temp)
                        end_time = datetime.datetime.strptime(temp[2],"%Y/%m/%d-%H:%M:%S:%f").timestamp()*1000
                        print("end time: ", end_time)
                    else:
                        x = x.replace("\"", "")
                        data_list = x.split(",")
                        remove_last_newLine(data_list)

                        # update the MCU timestamp
                        if ECG_FILES_CODE == FILES_CODE:
                            pass

                        elif BCG_FILES_CODE == FILES_CODE:

                            cnt += 1
                            if data_cnt == 0:
                                t_st = unix_time
                            unix_time += BCG_PERIOD
                            data_cnt += 1


                            bcg.append(data_list[1])
                            accx.append(data_list[2])
                            accy.append(data_list[3])
                            accz.append(data_list[4])
                            if (cnt % BCG_SAVING_CNTS) == 0:
                                collect_bcg = bcg_fields.copy()


                                collect_bcg["bcg"] = bcg
                                collect_bcg["accx"] = accx
                                collect_bcg["accy"] = accy
                                collect_bcg["accz"] = accz
                                collect_bcg["datalen"] = data_cnt
                                collect_bcg["timestamp"] = t_st
                                collect_bcg["timestampst"] = t_st
                                collect_bcg["timestampend"] = unix_time

                                # y = requests.post(url_bcg, json=collect_bcg, headers=headers)
                                # print(y)
                                # print(t_st)
                                # print(unix_time)

                                bcg = []
                                accx = []
                                accy = []
                                accz = []
                                bcg_id = []
                                id_cnt = 0
                                data_cnt = 0


    # print(SEARCH_TIMESTAMP)
    raw_data_path = CURRENT_PATH + "/" + user + "/" + "HR_raw"
    for directory, _x, files in os.walk(raw_data_path):
        for f in files:
            if ".csv" in f:
                
                print(directory + "/" + f)
                dataF = open(directory + "/" + f, "r")
                raw_data = dataF.readlines()
                cnt = 0
                l = 0

                hr = []
                status = []
                confidence_label = []
                Sdatetime_t = Sdatetime + datetime.timedelta(minutes=random)
                for data in raw_data:
                    
                    temp_cnt = 0
                    for d in data.split(","):
                        if temp_cnt == 0:
                            hr.append(float(d))
                        elif temp_cnt == 1:
                            status.append(float(d))
                        elif temp_cnt == 2:
                            confidence_label.append(float(d))
                        
                        temp_cnt += 1
                    cnt += 1
                    if cnt == 60:
                        l += 1
                        
                        Sdatetimestamp = int((Sdatetime_t).timestamp() * 1000 )

                        algo = algo_fields.copy()
                        algo["timestamp"] = Sdatetimestamp
                        algo["timestampm"] = Sdatetimestamp
                        algo["hr"] = hr
                        algo["status"] = status
                        algo["confidence"] = confidence_label
                        Sdatetime_t += datetime.timedelta(minutes=1)
                        y = requests.post(url_algo, json=algo, headers=headers)
                        #print(y)
                        #print(algo)
                        
                        cnt = 0
                        hr = []
                        status = []
                        confidence_label = []


                print("sum: ", l)
