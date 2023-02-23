import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import sys

if len(sys.argv) < 3 :
    print('usage: python3 spect_compare.py <file1> <file2>')
    sys.exit(1)

s1,fs1 = sf.read(sys.argv[1])
s2,fs2 = sf.read(sys.argv[2])

S1 = librosa.stft(s1, n_fft = 4096)
S2 = librosa.stft(s2, n_fft = 4096)

D1 = librosa.amplitude_to_db(np.abs(S1), ref=np.max)
D2 = librosa.amplitude_to_db(np.abs(S2), ref=np.max)

fig = plt.figure()

ax=fig.add_subplot(3,1,1)
librosa.display.specshow(D1, y_axis='log', x_axis='time', sr=fs1, ax=ax)
ax.set(title='Audiofile 1')
ax=fig.add_subplot(3,1,2)
librosa.display.specshow(D2, y_axis='log', x_axis='time', sr=fs1, ax=ax)
ax.set(title='Audiofile 2')
ax=fig.add_subplot(3,1,3)
librosa.display.specshow(np.abs(S1-S2), y_axis='log', x_axis='time', sr=fs1, ax=ax)
ax.set(title='Difference')

plt.show()
