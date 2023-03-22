import numpy as np
import soundfile as sf

fs = 48000

f1 = 440
f2 = 470

db1 = -20
db2 = -30

a1 = 10**(db1/20)
a2 = 10**(db2/20)

s1 = a1*np.sin(2*np.pi*f1/fs*np.arange(3*fs))
s2 = a2*np.sin(2*np.pi*f2/fs*np.arange(3*fs))

sf.write('sin440_neg20.wav', s1,fs)
sf.write('sin470_neg30.wav', s2,fs)
