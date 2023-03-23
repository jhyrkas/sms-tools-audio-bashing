import numpy as np
import soundfile as sf

fs = 48000

f1 = 440
f2 = 470
f3 = 880
f4 = 910

db1 = -20
db2 = -30
db3 = -30
db4 = -20

a1 = 10**(db1/20)
a2 = 10**(db2/20)
a3 = 10**(db3/20)
a4 = 10**(db4/20)

s1 = a1*np.sin(2*np.pi*f1/fs*np.arange(3*fs))
s2 = a2*np.sin(2*np.pi*f2/fs*np.arange(3*fs))
s3 = a3*np.sin(2*np.pi*f3/fs*np.arange(3*fs))
s4 = a4*np.sin(2*np.pi*f4/fs*np.arange(3*fs))

sf.write('sin440_neg20.wav', s1,fs)
sf.write('sin470_neg30.wav', s2,fs)
sf.write('sin880_neg30.wav', s3,fs)
sf.write('sin910_neg20.wav', s4,fs)
