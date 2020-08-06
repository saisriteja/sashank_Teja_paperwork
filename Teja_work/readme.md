# This is about teja work.

We are here with a aim of making a neural network for detection of interjections(filler words) in the speech.
<br>
We have opted two methods 
<br>
1.Prosodic - detect using speech feature alone.
<br>
2.Lexical - converting texttospeech and then use a NLP to detect 


## Prosodic:
Pros:<br>

1.Using this we can eliminate the troubles of miscallisifcation of text provoded by the TTS.<br>

2.There are many features we can extract namely 
  audio features - MFCC,LPC,LPCC,LSF,DWT,PLP
  image features - Spectrogram,MelSpectrogram,Cochleagram
  <br>

Cons:
Training from scratch takes time and a lot of experiementations

Libraries:
1. Librosa
2. Pydub
3. PyMIR

## Lexical
Pros:
No need of hetic work of going through pre processing stuff.
Best NLP models out are avaiable.

Cons:
TTS cant detect filler words or prolongations or blocks or relavant stuff similar to that



