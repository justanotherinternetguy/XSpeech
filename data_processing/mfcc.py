import librosa
import matplotlib.pyplot as plt
import librosa.display
import os
import numpy as np
from IPython.display import Audio
import wave
import pandas as pd
import matplotlib.pyplot as plt

root_fp = "../../DATA/ml-stuttering-events-dataset/"
CLIPS_DIR = "/home/alien/Git/DATA/ml-stuttering-events-dataset/CLIP_DIR/"
df = pd.read_csv(root_fp + "SEP-28k_labels.csv")

df['Name'] = df[df.columns[0:3]].apply(
    lambda x: '_'.join(x.dropna().astype(str)),
    axis=1
)

features = {}
for d in os.listdir(CLIPS_DIR):
    for subd in os.listdir(CLIPS_DIR + d):
        for filename in os.listdir(CLIPS_DIR + d + "/" + subd):
            temp = filename[:-4]
            fp = CLIPS_DIR + d + "/" + subd + "/" + filename
            audio, sample_rate = librosa.load(fp, res_type='kaiser_fast', sr=None)
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T,axis=0)
            features[temp] = mfccs


print(len(features))
print(features['MyStutteringLife_20_4'])

df_features = pd.DataFrame.from_dict(features)
df_features = df_features.transpose()

df_features = df_features.reset_index()

df_features = df_features.sort_values(by='index')

df_features.rename(columns = {'index':'Name'}, inplace = True)


df_final = pd.DataFrame()

df_final = pd.merge(df, df_features, how='inner', on='Name')
df_final = df_final[df_final.PoorAudioQuality == 0]
df_final = df_final[df_final.DifficultToUnderstand == 0]
df_final = df_final[df_final.Music == 0]
df_final = df_final[df_final.NoSpeech == 0]

df_final.to_csv("./sep28k-mfcc.csv",index=False)