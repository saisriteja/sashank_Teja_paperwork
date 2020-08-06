import numpy as np
import librosa


def noise(data, noise_factor):
    '''
    This function is about adding noise
    input signaldata,noisefactor
    output nosiy_data
    '''
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def speed(data, speed_factor):
    '''
    This function is about speeding up the audio file
    input signaldata,speedfactor
    output speed data
    '''
    return librosa.effects.time_stretch(data, speed_factor)

def pitch(data, sampling_rate, pitch_factor):
    '''
    This function is about speeding up the audio file
    input signaldata,sampling_rate,pitch_factor
    output pitch_data
    '''
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

