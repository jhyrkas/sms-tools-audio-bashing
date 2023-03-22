import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))
try :
    import basher_utils as bu
except :
    print("couldn't import basher_utils")
    sys.exit(1)

def roughness_func_exp(f1,f2) :
    return (0.24 / (0.021 * min(f1,f2) + 19)) * abs(f1-f2)

def get_roughness_x_y(f1, f2) :
    x = roughness_func_exp(f1, f2)
    y = bu.calculate_roughness_sethares(f1, 1.0, f2, 1.0)
    return x,y

def get_vline_params(f_const, f_change, delta) :
    lines_x = []
    lines_ymin = []
    lines_ymax = []
    # starting roughness
    x,y = get_roughness_x_y(f_const, f_change)
    lines_x.append(x)
    lines_ymin.append(0)
    lines_ymax.append(y)

    # maximum consonance
    cons_freq = bu.bash_freq(f_change, f_const, 0.05, 0.35, hard_bash=False, delta=delta, consonance = True)
    x,y = get_roughness_x_y(f_const, cons_freq)
    lines_x.append(x)
    lines_ymin.append(0)
    lines_ymax.append(y)

    # maximum dissonance
    diss_freq = bu.bash_freq(f_change, f_const, 0.05, 0.35, hard_bash=False, delta=delta, consonance = False)
    x,y = get_roughness_x_y(f_const, diss_freq)
    lines_x.append(x)
    lines_ymin.append(0)
    lines_ymax.append(y)

    # hard bashing
    hard_freq = bu.bash_freq(f_change, f_const, 0.05, 0.35, hard_bash=True, delta=delta, consonance = True)
    x,y = get_roughness_x_y(f_const, hard_freq)
    lines_x.append(x)
    lines_ymin.append(0)
    lines_ymax.append(y)

    labels = ['orig', 'consonant', 'dissonant', 'hard bash {d} Hz'.format(d=delta)]

    return lines_x, lines_ymin, lines_ymax, labels

# getting audio files (might not be necessary)
fs = 48000

s1,fs1 = librosa.core.load('audio_files/sin440_neg20.wav', sr=fs)
s2,fs2 = librosa.core.load('audio_files/sin470_neg30.wav', sr=fs)

# ----------------------
# BASHING FIGURES
# ----------------------
plt.figure(figsize=(14,6))

# plot 1
ax1 = plt.subplot(1,2,1)
# dissonance function
x = np.linspace(0,1,250)
y = np.exp(-3.5*x) - np.exp(-5.75*x)
ax1.plot(x,y)

# starting roughness
f1 = 440
f2 = 470 # changing bc it's quieter

colors=['red', 'orange', 'green', 'purple']
linestyles=['solid', 'dashed', 'dashdot', 'dotted']

lines_x, lines_ymin, lines_ymax, labels = get_vline_params(f1, f2, 3)

for i in range(4) :
    ax1.vlines(lines_x[i], lines_ymin[i], lines_ymax[i],
               colors=colors[i], linestyles=linestyles[i], label=labels[i])
#ax1.vlines(lines_x, lines_ymin, lines_ymax, 
#           colors=['red', 'orange', 'green', 'purple'],
#           linestyles=['solid', 'dashed', 'dashdot', 'dotted'],
#           label = ['orig', 'consonant', 'dissonant', 'hard bashed'])
ax1.set_title('base freq: {f1}, changing freq: {f2}'.format(f1=f1, f2=f2))
ax1.set_xlabel('Modeled % CB diff')
ax1.set_ylabel('Roughness (unitless)')
ax1.legend()

# plot 2
ax2 = plt.subplot(1,2,2)
# dissonance function
x = np.linspace(0,1,250)
y = np.exp(-3.5*x) - np.exp(-5.75*x)
ax2.plot(x,y)

# starting roughness
f1 = 880
f2 = 910 # changing bc it's quieter

lines_x, lines_ymin, lines_ymax, labels = get_vline_params(f1, f2, 3)
for i in range(4) :
    ax2.vlines(lines_x[i], lines_ymin[i], lines_ymax[i],
               colors=colors[i], linestyles=linestyles[i], label=labels[i])

ax2.set_title('base freq: {f1}, changing freq: {f2}'.format(f1=f1, f2=f2))
ax2.legend()

plt.savefig('freq_bashing.eps')
plt.clf()

# ----------------------
# WHACKING FIGURES
# ----------------------

f1 = 440
f2 = 470
db1 = -20
db2 = -30

# masking curve
ax1 = plt.subplot(1,2,1)
ax1.vlines([f1,f2], [-70,-70], [db1,db2])
x = np.linspace(300, 550, 1000)
y = [db1-bu.get_masking_level_dB(f1, f) for f in x]
ax1.plot(x,y)
ax1.set_xlim([280, 570])
ax1.set_ylim([-100, 0])

plt.show()




