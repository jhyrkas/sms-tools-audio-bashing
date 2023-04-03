import numpy as np
import soundfile as sf

fs = 48000

# sine

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

# sawtooth

def generate_saw(f) :
    fs = 48000
    outsig = np.zeros(3*fs)
    t = np.arange(3*fs) / fs
    for h in range(1,21) :
        outsig += (((-1)**h)/h) * np.sin(2*np.pi*h*f*t)
    outsig = 0.5 - (1/np.pi)*outsig
    return outsig

def generate_square(f) :
    fs = 48000
    outsig = np.zeros(3*fs)
    t = np.arange(3*fs) / fs
    for h in range(1,22,2) :
        outsig += (1/h) * np.sin(2*np.pi*h*f*t)
    outsig /= np.max(np.abs(outsig)) # avoid clipping
    return outsig

f1 = 220
f2_e = f1*(2**(7/12))
f2_j = f1*(3/2)
f3_e = f1*(2**(4/12))
f3_j = f1*(5/4)
f4_e = f1*(2**(11/12))
f4_j = f1*(15/8)

s1 = generate_saw(f1)
s2_e = generate_saw(f2_e)
s2_j = generate_saw(f2_j)
s3_e = generate_saw(f3_e)
s3_j = generate_saw(f3_j)
s4_e = generate_saw(f4_e)
s4_j = generate_saw(f4_j)

sf.write('saw_root.wav', s1, fs)
sf.write('saw_fifth_equal.wav', s2_e, fs)
sf.write('saw_fifth_just.wav', s2_j, fs)
sf.write('saw_third_equal.wav', s3_e, fs)
sf.write('saw_third_just.wav', s3_j, fs)
sf.write('saw_seven_equal.wav', s4_e, fs)
sf.write('saw_seven_just.wav', s4_j, fs)

s1 = generate_square(f1)
s2_e = generate_square(f2_e)
s2_j = generate_square(f2_j)
s3_e = generate_square(f3_e)
s3_j = generate_square(f3_j)
s4_e = generate_square(f4_e)
s4_j = generate_square(f4_j)

sf.write('square_root.wav', s1, fs)
sf.write('square_fifth_equal.wav', s2_e, fs)
sf.write('square_fifth_just.wav', s2_j, fs)
sf.write('square_third_equal.wav', s3_e, fs)
sf.write('square_third_just.wav', s3_j, fs)
sf.write('square_seven_equal.wav', s4_e, fs)
sf.write('square_seven_just.wav', s4_j, fs)
