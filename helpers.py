import librosa
import pandas as pd


def getLoudness(df, data):
    loudness = []
    for sound in data:
        signal, sample_rate = librosa.load(sound)
        S, phase = librosa.magphase(librosa.stft(signal))
        rms = librosa.feature.rms(S=S)
        loudness.append(rms[0])
    df["Loudness"] = loudness
    return df


def getMelFreq(df, data):
    mfccsList = dict()
    for sound in data:
        signal, sample_rate = librosa.load(sound)
        mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12)
        for n_mfcc in range(len(mfccs)):
            mfccsList['MFCC_%d' % (n_mfcc + 1)] = mfccs.T[n_mfcc]
    df_mfccs = pd.DataFrame(mfccsList)
    df = pd.concat([df, df_mfccs], axis=1)
    return df


def tester():
    return "help"
