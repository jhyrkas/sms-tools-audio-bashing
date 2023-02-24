import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import sys

if len(sys.argv) < 3 :
    print('usage: python3 spect_compare.py <file1> <file2> [normalize]')
    sys.exit(1)

s1,fs1 = sf.read(sys.argv[1])
s2,fs2 = sf.read(sys.argv[2])
normalize = bool(sys.argv[3]) if len(sys.argv) == 4 else False

if s1.shape[0] < s2.shape[0] :
    s1_final = np.zeros(s2.shape)
    s1_final[:s1.shape[0]] = s1
    s2_final = s2.copy()
else :
    s2_final = np.zeros(s1.shape)
    s2_final[:s2.shape[0]] = s2
    s1_final = s1.copy()


S1 = librosa.stft(s1, n_fft = 4096)
S2 = librosa.stft(s2, n_fft = 4096)
ss = librosa.istft(S1-S2)
#SD = np.abs(S1) - np.abs(S2)
#sgl = librosa.griffinlim(SD, n_iter=100)

print('method1')
sd.play(s1_final-s2_final,fs1)
sd.wait()
print('method2')
sd.play(ss,fs1)
sd.wait()
#print('method3')
#sd.play(sgl,fs1)
#sd.wait()


#librosa.display.specshow(np.abs(S1-S2), y_axis='log', x_axis='time', sr=fs1, ax=ax)
