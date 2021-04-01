#importing the necessary libraries
from features import Features
import pandas as pd
import time
import pickle
from os import listdir
from typing import Dict
import json
from datetime import datetime



def get_files(file_path: str) -> Dict[str, str]:
    """
    Function to get the list of files inside a file path
    :parameter file_path will contain the file path of the folder
    :attrib onlyfiles is a list of all the files in file_path
    :attrib files is a dictionary which will have the sound file as key,
    and the file path as the value
    This function returns the files as a dictionary
    """
    onlyfiles = [f for f in listdir(file_path)]
    files = {}
    for i in onlyfiles:
        files[i] = file_path + i
    return files


def machine(machine_type: str) ->str:
    """
    Function machine gets the type model depending on the machine type
    :parameter machine_type will contain the type of machine
    """
    if machine_type.lower() == "fan":
        model = "Models/fan_model.sav"
        return model
    elif machine_type.lower() == "valve":
        model = "Models/valve_model.sav"
        return model
    elif machine_type.lower() == "pump":
        model = "Models/pump_model.sav"
        return model
    elif machine_type.lower() == "slider":
        model = "Models/slider_model.sav"
        return model


def predict(df, model: str) -> int:
    """
    Function predict will predict the data
    :parameter df contains the file with the features
    :parameter model contains the model to be used in predicting
    :attrib fm will contain the loaded prediction
    :attrib prediction will contain the prediction from the model
    This function returns the variable if 1, means abnormal, if 0 means normal
    
    """
    with open(model, 'rb') as f:
        fm = pickle.load(f)
        prediction = fm.predict(df)
        #print(fm.score(df, df))
        #print(f"Machine is {prediction}.")
    
    if prediction == 1:
        print("Machine is Abnormal.")
        return 1
            #send an alert    
    
    elif prediction == 0:
        print("Machine is Normal.")
        return 0
    
    
def results(pred: Dict[str, int], file_path: str, anomalies: str, machine_type: str):
    """
    Function to get the results and save it as a json file
    :parameter pred contains the prediction in a dictionary format
    :parameter file_path contains the file path of the folder
    :parameter anomalies contains the number of anomalies
    :parameter machine_type contains the type of machine
    It creates a json file and saves the json file in the Test_Predictions_Output
    """
    now = datetime.now()
    date = now.strftime("%d/%m/%Y %H:%M:%S")
    result = {
        'Type of Machine': machine_type,
        'File Path' : file_path,
        'Date and Time' : date,
        'Predictions' : pred,
        'Total Anomalies' : anomalies
        }

    with open(f'Demo/Test_Predictions_output/{machine_type}_result.json', 'w') as json_file:
        json.dump(result, json_file)
        print("Results are now in your file!")
   

def get_prediction(dict_files: Dict[str,str], model: str) -> Dict[str, int]:
    """
    Function to get the prediction of the files
    :parameter dict_files will contain the file paths 
    :parameter model will contain the specified model for prediction
    :attrib pred will contain a dictionary with the prediction
    This function will return the prediction in a dictionary

    """
    pred = {}
    for key, value in dict_files.items(): 
        print(value)           
        data = Features.get_features(value)
        df = pd.DataFrame([data])
        print(f"Predicting for {key}...")
        prediction = predict(df, model) 
        pred[key] = prediction     
    return pred


def count_anomalies(pred: Dict[str, int]) -> int:
    """
    Function to calculate the anomalies in the prediction
    :parameter pred contains the prediction in a dictionary format
    :attrib count will contain the total number of anomalies
    This function returns the count as in integer
    """
    count = 0 
    for i in pred.values():
        if i == 1:
            count +=1
    return count


"""
This is where the program starts
:attrib machine_type will contain the input type of machine
:attrib file_path will contain the file path of the sound files
:attrib start_time will contain the time the program started
:attrib model will contain the model to predict the file
:attrib dict_files will contain the files in a dictionary
:attrib pred will contain the prediction in a dictionary
:attrib count will contain the total number of abnormal in the set
:attrib end_time will contain the time the program ended
"""


machine_type = input("Enter the machine type: ")
file_path = input("Enter the file path: ")
start_time = time.time()
model = machine(machine_type)
dict_files = get_files(file_path)
pred = get_prediction(dict_files, model)
count = count_anomalies(pred)
        
anomalies = str(count) +  " out of " + str(len(pred.values())) + " samples."
print(f"Predicted Anomalies: {anomalies}")
results(pred, file_path, anomalies, machine_type)
end_time = time.time()
print(f"Program runs for {end_time - start_time} seconds.")
    


    
