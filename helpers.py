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


def getZCR(df, data):
    zcrList = []
    for sound in data:
        signal, sample_rate = librosa.load(sound)
        zcr = librosa.feature.zero_crossing_rate(y=signal)
        zcrList.append(zcr[0])
    df["ZCR"] = zcrList
    return df


def getChroma(df, data):
    chroma = dict()
    for sound in data:
        signal, sample_rate = librosa.load(sound)
        chromagram = librosa.feature.chroma_stft(y=signal, sr=sample_rate)
        for n_chroma in range(len(chromagram)):
            chroma['Chroma_%d' % (n_chroma + 1)] = chromagram.T[n_chroma]
    df_chroma = pd.DataFrame(chroma)
    df = pd.concat([df, df_chroma], axis=1)
    return df


def getMelSpect(df, data):
    mel_spectrogramList = dict()
    for sound in data:
        signal, sample_rate = librosa.load(sound)
        mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_mels=12)
        for n_mel in range(len(mel_spectrogram)):
            mel_spectrogramList['Mel_Spectrogram_%d' % (n_mel + 1)] = mel_spectrogram.T[n_mel]
    df_mel_spectrogram = pd.DataFrame(mel_spectrogramList)
    df = pd.concat([df, df_mel_spectrogram], axis=1)
    return df
