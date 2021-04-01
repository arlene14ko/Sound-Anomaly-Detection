#importing the necessary libraries
import pandas as pd
import numpy as np
import librosa
from typing import Dict

class Features:
    """
    CLass Features will contain all the functions to get the features of the sound file
    """
    
    
    def get_y_sr(filename: str):
        """
        Function to load an audio file as a floating point time series.
        :parameter filename is required, it is the file path of the sound file
        :attrib y and SR will contain the loaded value of the sound file
        This function returns the attrib y and attrib SR
        """
        y, sr = librosa.load(filename)
        return  y, sr
    
    
    def chroma_stft(y):
        """
        Function to compute a chromagram from a waveform 
        :parameter y contains the loaded sound file
        This function returns computed chromogram
        """
        return librosa.feature.chroma_stft(y=y[0], sr=y[1])
            
    
    def chroma_cqt(y):
        """
        Function to get the constant-Q chromagram
        :parameter y contains the loaded sound file
        This function returns constant-Q chromogram in an array
        """
        return librosa.feature.chroma_cqt(y=y[0], sr=y[1])
    
    
    def chroma_cens(y):
        """
        Function that computes the chroma variant “Chroma Energy Normalized” (CENS)
        :parameter y contains the loaded sound file
        This function returns computed chroma energy normalized
        """
        return librosa.feature.chroma_cens(y=y[0], sr=y[1])
    
    
    def melspectogram(y):
        """
        Function that computes a mel-scaled spectrogram
        :parameter y contains the loaded sound file
        This function returns computed mel-scaled spectrogram.
        """
        return librosa.feature.melspectrogram(y=y[0], sr=y[1])
    
    
    def mfcc(y):
        """
        Function that computes a mel-scaled spectrogram
        :parameter y contains the loaded sound file
        This function returns computed mel-scaled spectrogram.
        """
        return librosa.feature.mfcc(y=y[0], sr=y[1])
        
    
    def rms(y): 
        """
        Function to compute the root-mean-square (RMS) value for each frame
        :parameter y contains the loaded sound file
        This function returns the computed RMS
        """
        return librosa.feature.rms(y=y)
    
    def spec_centroid(y):
        """
        Function to compute the spectral centroid
        :parameter y contains the loaded sound file
        This function returns the computed spectral centroid
        """
        return librosa.feature.spectral_centroid(y=y[0], sr=y[1])
    
    
    def spec_bw(y):
        """
        Function to compute the spectral bandwidth
        :parameter y contains the loaded sound file
        This function returns the computed spectral bandwidth
        """
        return librosa.feature.spectral_bandwidth(y=y[0], sr=y[1])
    
    
    def spec_contrast(y):
        """
        Function to compute the spectral contrast
        :parameter y contains the loaded sound file
        This function returns the computed spectral contrast
        """
        S = np.abs(librosa.stft(y[0]))
        contrast = librosa.feature.spectral_contrast(S=S, sr=y[1])
        return contrast
    
    
    def spec_flatness(y):
        """
        Function to compute the spectral flatness
        :parameter y contains the loaded sound file
        This function returns the computed spectral flatness
        """
        return librosa.feature.spectral_flatness(y=y)
    
    
    def spec_rolloff(y):
        """
        Function to compute the rolloff frequency
        :parameter y contains the loaded sound file
        This function returns the rolloff frequency
        """
        return librosa.feature.spectral_rolloff(y=y[0], sr=y[1])
    
    
    def poly_0(y):
        """
        Function to fit a degree-0 polynomial (constant) to each frame
        :parameter y contains the loaded sound file
        This function returns the degree-0 polynomial
        """
        S = np.abs(librosa.stft(y))
        return librosa.feature.poly_features(S=S, order=0)
    
    
    def poly_1(y):
        """
        Function to fit a linear polynomial to each frame
        :parameter y contains the loaded sound file
        This function returns the linear polynomial
        """
        S = np.abs(librosa.stft(y))
        return librosa.feature.poly_features(S=S, order=1)
    
    
    def poly_2(y):
        """
        Function to fit a quadratic to each frame
        :parameter y contains the loaded sound file
        This function returns the quadratic
        """
        S = np.abs(librosa.stft(y))
        return librosa.feature.poly_features(S=S, order=2)
    
    
    def tonnetz(x):
        """
        Function to compute the tonal centroid features (tonnetz)
        :parameter y contains the loaded sound file
        This function returns the computed tonnetz
        """
        y = librosa.effects.harmonic(x[0])
        return librosa.feature.tonnetz(y=y, sr=x[1])
    
    def zcr(y):
        """
        Function to compute the zero-crossing rate of an audio time series.
        :parameter y contains the loaded sound file
        This function returns the computed zero-crossing rate
        """
        return librosa.feature.zero_crossing_rate(y)
    
    
    def get_features(filename: str) -> Dict[str, str]:
        """
        Function to get the calculated features of the sound file
        :parameter filename will contain the file path of the sound file
        :attrib y will contain the loaded sound file
        :attrib SR will contain the sample rate of the sound file
        :attrib x will contain the attrib y and SR in a list
        :attrib chroma_stft will contain the computed chroma_stft
        :attrib chroma_cqt will contain the chroma_cqt(x)
        :attrib chroma_cens will contain the chroma_cens
        :attrib melspectrogram contain the computed melspectrogram
        :attrib mfcc will contain the mfcc
        :attrib rms will contain the rms
        :attrib spec_centroid will contain the computed spectral centroid(x)
        :attrib spec_bw will contain the computed spectral bandwidth
        :attrib spec_contrast contain the spectral contrast
        :attrib flatness contain the spectral flatness
        :attrib spec_rolloff contains the spectral rolloff
        :attrib poly_0 contains the 0 degree polynomial
        :attrib poly_1 contains the linear polynomial
        :attrib poly_2 contains the quadratic polynomial
        :attrib tonnetz contains the computed tonnetz
        :attrib zcr contains the zero crossing rate
        :attrib data will contain a dictionary with all the features
        This function will return the data as a dictionary with all the features
        """
        print("Loading the data....")
        y, SR = Features.get_y_sr(filename)
        x = [y, SR]
        chroma_stft = Features.chroma_stft(x)
        chroma_cqt = Features.chroma_cqt(x)
        chroma_cens = Features.chroma_cens(x)
        melspectrogram = Features.melspectogram(x)
        mfcc = Features.mfcc(x)
        rms = Features.rms(y)
        spec_centroid = Features.spec_centroid(x)
        spec_bw = Features.spec_bw(x)
        spec_contrast = Features.spec_contrast(x)
        flatness = Features.spec_flatness(y)    
        spec_rolloff = Features.spec_rolloff(x)
        poly_0 = Features.poly_0(y)
        poly_1 = Features.poly_1(y)
        poly_2 = Features.poly_2(y)
        tonnetz = Features.tonnetz(x)
        zcr = Features.zcr(y)
    
  
        data = {"y_mean" : y.mean(), 
                "y_max" : y.max(), "y_min" : y.min(),
                "chroma_stft min" : chroma_stft.min(),
                "chroma_stft mean" : chroma_stft.mean(),
                "chroma_stft max" : chroma_stft.max(),
                "chroma_cqt min" : chroma_cqt.min(),
                "chroma_cqt mean" : chroma_cqt.mean(),
                "chroma_cqt max" : chroma_cqt.max(),
                "chroma_cens min" : chroma_cens.min(),
                "chroma_cens mean" : chroma_cens.mean(),
                "chroma_cens max" : chroma_cens.max(),
                "melspectrogram min" : melspectrogram.min(),
                "melspectrogram mean" : melspectrogram.mean(),
                "melspectrogram max" : melspectrogram.max(),
                "mfcc min" : mfcc.min(), "mfcc mean" : mfcc.mean(),
                "mfcc max" : mfcc.max(), "rms min" : rms.min(),
                "rms mean" : rms.mean(), "rms max" : rms.max(),
                "spec_centroid min" : spec_centroid.min(),
                "spec_centroid mean" : spec_centroid.mean(),
                "spec_centroid max" : spec_centroid.max(),
                "spec_bw min" : spec_bw.min(), 
                "spec_bw mean" : spec_bw.mean(),
                "spec_bw max" : spec_bw.max(), 
                "spec_contrast min" : spec_contrast.min(),
                "spec_contrast mean" : spec_contrast.mean(),
                "spec_contrast max" : spec_contrast.max(),
                "flatness min" : flatness.min(),
                "flatness mean" : flatness.mean(),
                "flatness max" : flatness.max(),
                "spec_rolloff min" : spec_rolloff.min(),
                "spec_rolloff mean" : spec_rolloff.mean(),
                "spec_rolloff max" : spec_rolloff.max(),
                "poly_0 min" : poly_0.min(), "poly_0 mean" : poly_0.mean(),
                "poly_0 max" : poly_0.max(), "poly_1 min" : poly_1.min(),
                "poly_1 mean" : poly_1.mean(), "poly_1 max" : poly_1.max(),
                "poly_2 min" : poly_2.min(), "poly_2 mean" : poly_2.mean(),
                "poly_2 max" : poly_2.max(), 
                "tonnetz min" : tonnetz.min(),
                "tonnetz mean" : tonnetz.mean(),
                "tonnetz max" : tonnetz.max(),
                "zcr min" : zcr.min(), "zcr mean" : zcr.mean(),
                "zcr max" : zcr.max(), "y std" : y.std(), 
                "chroma_stft std" : chroma_stft.std(),
                "chroma_cqt std": chroma_cqt.std(), 
                "chroma_cens std" : chroma_cens.std(), 
                "melspectrogram std" : melspectrogram.std(), "mfcc std" : mfcc.std(),
                "rms std" : rms.std(), "spec_centroid std" : spec_centroid.std(),
                "spec_bw std" : spec_bw.std(), 
                "spec_contrast std" : spec_contrast.std(),
                "flatness std": flatness.std(),
                "spec_rolloff std" : spec_rolloff.std(),
                "poly_0 std" : poly_0.std(),
                "poly_1 std" : poly_1.std(),
                "poly_2 std" : poly_2.std(),
                "tonnetz std": tonnetz.std(), 
                "zcr std" : zcr.std()
            }
        return data
    
    
    def preprocessing(filepath):
        """
        Function to create the dataframe with the features
        :parameter filepath will contain the file path of all the sound files
        :attrib df will contain the created dataframe
        This function will return the features in a  dataframe

        """
        
        df = pd.read_csv(filepath)
        print("Loading the data....")
        df['y and SR'] = df['File Path'].apply(Features.get_y_sr)
        
        df['y'] = df['y and SR'].apply(lambda x: x[0])
        df['SR'] = df['y and SR'].apply(lambda x: x[1])
        
        print("Getting the data....")
        df['y mean'] = df['y'].apply(lambda x : x.mean())
        df['y max'] = df['y'].apply(lambda x: x.max())
        df['y min'] = df['y'].apply(lambda x: x.min())
        
        
        print("Getting the chromagram stft....")
        df['chroma_stft'] = df['y and SR'].apply(Features.chroma_stft)
        df['chroma_stft min'] = df['chroma_stft'].apply(lambda x: x.min())
        df['chroma_stft mean'] = df['chroma_stft'].apply(lambda x: x.mean())
        df['chroma_stft max'] = df['chroma_stft'].apply(lambda x: x.max()) 
        
        
        print("Getting the chroma cqt....")
        df['chroma_cqt'] = df['y and SR'].apply(Features.chroma_cqt)
        df['chroma_cqt min'] = df['chroma_cqt'].apply(lambda x: x.min())
        df['chroma_cqt mean'] = df['chroma_cqt'].apply(lambda x: x.mean())
        df['chroma_cqt max'] = df['chroma_cqt'].apply(lambda x: x.max())
        
        print("Getting the chroma cens....")
        df['chroma_cens'] = df['y'].apply(Features.chroma_cens)
        df['chroma_cens min'] = df['chroma_cens'].apply(lambda x: x.min())
        df['chroma_cens mean'] = df['chroma_cens'].apply(lambda x: x.mean())
        df['chroma_cens max'] = df['chroma_cens'].apply(lambda x: x.max())
        
        print("Getting the melspectogram....")
        df['melspectogram'] = df['y and SR'].apply(Features.melspectogram)
        df['melspectogram min'] = df['melspectogram'].apply(lambda x: x.min())
        df['melspectogram mean'] = df['melspectogram'].apply(lambda x: x.mean())
        df['melspectogram max'] = df['melspectogram'].apply(lambda x: x.max())
        
        print("Getting the MFCC....")
        df['mfcc'] = df['y and SR'].apply(Features.mfcc)
        df['mfcc min'] = df['mfcc'].apply(lambda x: x.min())
        df['mfcc mean'] = df['mfcc'].apply(lambda x: x.mean())
        df['mfcc max'] = df['mfcc'].apply(lambda x: x.max())
        
        print("Getting the RMS....")
        df['rms'] = df['y'].apply(Features.rms)
        df['rms min'] = df['rms'].apply(lambda x: x.min())
        df['rms mean'] = df['rms'].apply(lambda x: x.mean())
        df['rms max'] = df['rms'].apply(lambda x: x.max())
        
        print("Getting the Spectral Centroid....")
        df['spec_centroid'] = df['y and SR'].apply(Features.spec_centroid)
        df['spec_centroid min'] = df['spec_centroid'].apply(lambda x: x.min())
        df['spec_centroid mean'] = df['spec_centroid'].apply(lambda x: x.mean())
        df['spec_centroid max'] = df['spec_centroid'].apply(lambda x: x.max())
         
        print("Getting the Spectral Bandwidth....")  
        df['spec_bw'] = df['y and SR'].apply(Features.spec_bw)
        df['spec_bw min'] = df['spec_bw'].apply(lambda x: x.min())
        df['spec_bw mean'] = df['spec_bw'].apply(lambda x: x.mean())
        df['spec_bw max'] = df['spec_bw'].apply(lambda x: x.max())
        
        print("Getting the Spectral Contrast....")
        df['spec_contrast'] = df['y and SR'].apply(Features.spec_contrast)
        df['spec_contrast min'] = df['spec_contrast'].apply(lambda x: x.min())
        df['spec_contrast mean'] = df['spec_contrast'].apply(lambda x: x.mean())
        df['spec_contrast max'] = df['spec_contrast'].apply(lambda x: x.max())
        
        print("Getting the Spectral Flatness....")
        df['flatness'] = df['y'].apply(Features.spec_flatness)
        df['flatness min'] = df['flatness'].apply(lambda x: x.min())
        df['flatness mean'] = df['flatness'].apply(lambda x: x.mean())
        df['flatness max'] = df['flatness'].apply(lambda x: x.max())
        
        print("Getting the Spectral Roll off....")
        df['rolloff'] = df['y and SR'].apply(Features.spec_rolloff)
        df['rolloff min'] = df['rolloff'].apply(lambda x: x.min())
        df['rolloff mean'] = df['rolloff'].apply(lambda x: x.mean())
        df['rolloff max'] = df['rolloff'].apply(lambda x: x.max())
        
        print("Fitting a 0-degree polynomial....")
        df['poly_0'] = df['y'].apply(Features.poly_0)
        df['poly_0 min'] = df['poly_0'].apply(lambda x: x.min())
        df['poly_0 mean'] = df['poly_0'].apply(lambda x: x.mean())
        df['poly_0 max'] = df['poly_0'].apply(lambda x: x.max())
        
        print("Fitting the linear polynomial....")
        df['poly_1'] = df['y'].apply(Features.poly_1)
        df['poly_1 min'] = df['poly_1'].apply(lambda x: x.min())
        df['poly_1 mean'] = df['poly_1'].apply(lambda x: x.mean())
        df['poly_1 max'] = df['poly_1'].apply(lambda x: x.max())
        
        print("Fit a quadratic ....")
        df['poly_2'] = df['y'].apply(Features.poly_2)
        df['poly_2 min'] = df['poly_2'].apply(lambda x: x.min())
        df['poly_2 mean'] = df['poly_2'].apply(lambda x: x.mean())
        df['poly_2 max'] = df['poly_2'].apply(lambda x: x.max())
        
        print("Getting the tonnetz....")
        df['tonnetz'] = df['y and SR'].apply(Features.tonnetz)
        df['tonnetz min'] = df['tonnetz'].apply(lambda x: x.min())
        df['tonnetz mean'] = df['tonnetz'].apply(lambda x: x.mean())
        df['tonnetz max'] = df['tonnetz'].apply(lambda x: x.max())
        
        print("Getting the Zero Crossing Rate....")
        df['zero_crossing_rate'] = df['y'].apply(Features.zcr)
        df['zero_crossing_rate min'] = df['zero_crossing_rate'].apply(lambda x: x.min())
        df['zero_crossing_rate mean'] = df['zero_crossing_rate'].apply(lambda x: x.mean())
        df['zero_crossing_rate max'] = df['zero_crossing_rate'].apply(lambda x: x.max())
           
        
        df['y std'] = df['y'].apply(lambda x: x.std())
        df['chroma_stft std'] = df['chroma_stft'].apply(lambda x: x.std())
        df['chroma_cqt std'] = df['chroma_cqt'].apply(lambda x: x.std())
        df['chroma_cens std'] = df['chroma_cens'].apply(lambda x: x.std())
        df['melspectogram std'] = df['melspectogram'].apply(lambda x: x.std())
        df['mfcc std'] = df['mfcc'].apply(lambda x: x.std())
        df['rms std'] = df['rms'].apply(lambda x: x.std())
        df['spectral_centroid std'] = df['spectral_centroid'].apply(lambda x: x.std())
        df['spec_bw std'] = df['spec_bw'].apply(lambda x: x.std())
        df['spec_contrast std'] = df['spec_contrast'].apply(lambda x: x.std())
        df['flatness std'] = df['flatness'].apply(lambda x: x.std())
        df['rolloff std'] = df['rolloff'].apply(lambda x: x.std())
        df['poly_0 std'] = df['poly_0'].apply(lambda x: x.std())
        df['poly_1 std'] = df['poly_1'].apply(lambda x: x.std())
        df['poly_2 std'] = df['poly_2'].apply(lambda x: x.std())
        df['tonnetz std'] = df['tonnetz'].apply(lambda x: x.std())
        df['zero_crossing_rate std'] = df['zero_crossing_rate'].apply(lambda x: x.std())
        
        df1 = df[[ 'Type of SNR', 'Type of Machine', 'Model Number', 'Status',
           'File Name', 'y mean', 'y max', 'y min',
           'chroma_stft min', 'chroma_stft mean', 'chroma_stft max', 'chroma_cqt min', 'chroma_cqt mean', 'chroma_cqt max',
           'chroma_cens min', 'chroma_cens mean', 'chroma_cens max',
            'melspectogram min', 'melspectogram mean',
           'melspectogram max',  'mfcc min', 'mfcc mean', 'mfcc max',
           'rms min', 'rms mean', 'rms max', 'spectral_centroid min', 'spectral_centroid mean',
           'spectral_centroid max',  'spec_bw min', 'spec_bw mean',
           'spec_bw max', 'spec_contrast min', 'spec_contrast mean', 'spec_contrast max', 'flatness min',
           'flatness mean', 'flatness max',  'rolloff min',
           'rolloff mean', 'rolloff max',   'poly_0 min', 'poly_0 mean',
           'poly_0 max',  'poly_1 min', 'poly_1 mean', 'poly_1 max',
             'poly_2 min', 'poly_2 mean', 'poly_2 max', 
           'tonnetz min', 'tonnetz mean', 'tonnetz max', 
              'zero_crossing_rate min', 'zero_crossing_rate mean',
           'zero_crossing_rate max', 'y std', 'chroma_stft std', 'chroma_cqt std',
           'chroma_cens std', 'melspectogram std', 'mfcc std', 'rms std',
           'spectral_centroid std', 'spec_bw std', 'spec_contrast std',
           'flatness std', 'rolloff std', 'poly_0 std', 'poly_1 std', 'poly_2 std',
           'tonnetz std', 'zero_crossing_rate std']].copy()
        
        return df1
    
    
   
    
    
    