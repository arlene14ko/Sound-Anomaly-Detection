#importing the necessary libraries
from features import Features
import time

"""
This is where the preprocessing program starts
:attrib start_time will calculate the time the program started
:attrib filepath will get the file path from the input
:attrib df will contain the preprocessed data with all the features
:attrib end_time will calculate the time the program ended
This program creates the dataframe and saves it as a csv file
"""

start_time = time.time()
filepath = input("Enter csv filepath: ")

df = Features.preprocessing(filepath)
df.to_csv(f'{filepath.split(".")[0]}_features1.csv', index=False)

end_time = time.time()
print(f'Successfully got the features for {filepath.split(".")[0]} and saved it as a csv.')
print(f"Program runs for {end_time - start_time} seconds.")
