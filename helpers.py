import librosa
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np


def getLoudness(df, data):
    dic_loudness = dict()
    for sound in data:
        signal, sample_rate = librosa.load(sound)
        S, phase = librosa.magphase(librosa.stft(signal))
        rms = librosa.feature.rms(S=S)
        dic_loudness[sound] = rms[0]
    test_dic = {'Loudness': dic_loudness.values()}
    df_loud = pd.DataFrame.from_dict(test_dic)
    df_loud = df_loud.reset_index(drop=True)
    df = pd.concat([df, df_loud], axis=1)
    return df


def getMelFreq(df, data):
    dic_mfc = dict()
    for sound in data:
        mfccsList = dict()
        signal, sample_rate = librosa.load(sound)
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12)
        for n_mfcc in range(len(mfccs)):
            mfccsList['MFCC_%d' % (n_mfcc + 1)] = mfccs.T[n_mfcc]
        dic_mfc[sound] = mfccsList
    df_mfc = pd.DataFrame.from_dict(dic_mfc, orient='index')
    df_mfc = df_mfc.reset_index(drop=True)
    df = pd.concat([df, df_mfc], axis=1)
    return df


def getZCR(df, data):
    dic_zcr = dict()
    for sound in data:
        signal, sample_rate = librosa.load(sound)
        zcr = librosa.feature.zero_crossing_rate(y=signal)
        dic_zcr[sound] = zcr[0]
    test_dic = {'ZCR': dic_zcr.values()}
    df_zcr = pd.DataFrame.from_dict(test_dic)
    df_zcr = df_zcr.reset_index(drop=True)
    df = pd.concat([df, df_zcr], axis=1)
    return df


def getChroma(df, data):
    dic_chroma = dict()
    for sound in data:
        chroma = dict()
        signal, sample_rate = librosa.load(sound)
        chromagram = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
        for n_chroma in range(len(chromagram)):
            chroma['Chroma_%d' % (n_chroma + 1)] = chromagram.T[n_chroma]
        dic_chroma[sound] = chroma
    df_chroma = pd.DataFrame.from_dict(dic_chroma, orient='index')
    df_chroma = df_chroma.reset_index(drop=True)
    df = pd.concat([df, df_chroma], axis=1)
    return df


def getMelSpect(df, data):
    dic_melSpect = dict()
    for sound in data:
        mel_spectrogramList = dict()
        signal, sample_rate = librosa.load(sound)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=12)
        for n_mel in range(len(mel_spectrogram)):
            mel_spectrogramList['Mel_Spectrogram_%d' % (n_mel + 1)] = mel_spectrogram.T[n_mel]
        dic_melSpect[sound] = mel_spectrogramList
    df_melSpect = pd.DataFrame.from_dict(dic_melSpect, orient='index')
    df_melSpect = df_melSpect.reset_index(drop=True)
    df = pd.concat([df, df_melSpect], axis=1)
    return df


# Define a function to scale a NumPy array
def scale_array(arr):
    if isinstance(arr, np.ndarray):
        scaler = StandardScaler()
        return scaler.fit_transform(arr.reshape(-1, 1)).flatten()
    else:
        return arr


def avg_array(arr):
    if isinstance(arr, np.ndarray):
        return np.mean(arr)
    else:
        return arr
