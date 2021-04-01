#importing the necessary libraries
import pandas as pd
from zipfile import ZipFile
import time
from typing import List


def get_filenames(zip_file: str) -> List:
    """
    Function to get all the files inside the zip files
    :attrib file_names will contain all the filepath in a list
    This function returns the file_names attrib as a list
    """
    file_names = []
    with ZipFile(zip_file, 'r') as zipObj:
        listOfiles = zipObj.namelist()
        for elem in listOfiles:
            if "wav" in elem:
                file_names.append(elem)
    return file_names

def create_df(zip_file: str, file_names: List):
    """
    Function that will create a dataframe with all the file paths
    :attrib df will contain the created dataframe
    This function returns the attrib df

    """
    df = pd.DataFrame(file_names, columns=["File Path"])
    df['File Path'] = df['File Path'].apply(lambda x: zip_file.split(".")[0] + "/" + x) 
    df['Type of SNR'] = df['File Path'].apply(lambda x: x.split("/")[1])
    df['Type of Machine'] = df['File Path'].apply(lambda x: x.split("/")[2])
    df['Model Number'] = df['File Path'].apply(lambda x: x.split("/")[3])
    df['Status'] = df['File Path'].apply(lambda x: x.split("/")[4])
    df['File Name'] = df['File Path'].apply(lambda x: x.split("/")[5])
    return df


"""
This is where the program starts
:attrib start_time will calculate the time the program started
:attrib zip_file will get the file path from the input
:attrib file_names will contain a list of the files on the zip file
:attrib df will contain the dataframe with the files paths
:attrib end_time will calculate the time the program ended
This program creates the dataframe and saves it as a csv file
"""

start_time = time.time()
zip_file = input("Enter file path: ")
file_names = get_filenames(zip_file)
print(file_names)
df = create_df(zip_file, file_names)
df.to_csv(f'{zip_file.split(".")[0]}_filepath.csv', index=False)
end_time = time.time()
print(f"Successfully created the filepath csv for {zip_file.split('.')[0]}")
print(f"Program runs for {end_time - start_time} seconds.")

