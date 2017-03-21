import numpy as np
import matplotlib
import os
import librosa
from dtw import dtw
import shelve
from scipy.io import wavfile
import string
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def stretch(inp_array,factor):
        new_array = np.empty(0)
        for i in range(len(inp_array)-1):
            start = inp_array[i]
            end = inp_array[i+1]
            mid = np.linspace(start, end, factor+1)[:-1]*factor
            #print mid
            new_array = np.append(new_array,mid)

        mid = np.linspace(inp_array[len(inp_array)-1], inp_array[-1], factor)*factor 
        new_array = np.append(new_array,mid)
        return new_array.reshape(new_array.size)
    
    


def align_sound(fileName_inp,fileName_out,sr_long=8000,sr_short=1000):
    sr_factr = sr_long / sr_short;
    x,_ = librosa.load(fileName_inp,sr=sr_long)
    x = x.reshape(-1, 1)
    y,_ = librosa.load(fileName_out,sr=sr_long)
    y = y.reshape(-1, 1)

    x_s,_ = librosa.load(fileName_inp,sr=sr_short)
    x_s = x_s.reshape(-1, 1)
    y_s,_ = librosa.load(fileName_out,sr=sr_short)
    y_s = y_s.reshape(-1, 1)


    apply_fast  = False

    if (apply_fast):
        dist, path = fastdtw(x, y, dist=euclidean)
        path = np.transpose(path)
    else:
        cost, dist, acc, path = dtw(x_s, y_s, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

    path_inp = path[0]
    path_out = path[1]    
    long_path_inp = stretch(path_inp, sr_factr)
    long_path_out = stretch(path_out, sr_factr)

    filt = np.ones(500)/(500) 
    long_path_inp = np.convolve(long_path_inp, filt, 'same').astype(int)
    long_path_out = np.convolve(long_path_out, filt, 'same').astype(int)
   
    
    x_sample = x[long_path_inp]
    y_sample = y[long_path_out]

    name = ''.join(random.choice(string.ascii_uppercase) for _ in range(10))
    print name
    
    wavfile.write("newfiles/%s_input.wav" %name,sr_long, x_sample)

    wavfile.write("newfiles/%s_output.wav" %name,sr_long, y_sample)
    return (x_sample,y_sample)

if __name__ == "__main__":
    d=shelve.open("/Users/guyknaan/my-stuff/voiceswap/code/data")["train"]
    fileName_x = "/Users/guyknaan/my-stuff/voiceswap/code/%s" %d[10][0]
    fileName_y = "/Users/guyknaan/my-stuff/voiceswap/code/%s" %d[10][1]
    os.system("play %s -r 1000 " %fileName_x)
    os.system("play %s -r 1000 " %fileName_y)
    align_sound(fileName_x,fileName_y)
