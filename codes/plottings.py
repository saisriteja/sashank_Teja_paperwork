import matplotlib.pyplot as plt
import librosa
import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np


def spectrogram(self,save_path,limits = (0,10000)):
    '''
    inputs self,save_path,frequency limits,save 
    saves a image as output
    '''
    plt.figure(figsize=(14, 5))
    X = librosa.stft(self.signalData)
    Xdb = librosa.amplitude_to_db(abs(X))
    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge    
    librosa.display.specshow(Xdb, sr=self.samplingFrequency, x_axis='time', y_axis='hz')
    l1,l2 = limits
    plt.ylim(l1,l2)
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()