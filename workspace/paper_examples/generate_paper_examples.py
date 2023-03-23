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

if not os.path.exists('audio_files/sin440_neg20.wav') :
    print('Please run and then try again: cd audio_files; python3 generate_sines.py; cd ..')
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

    labels = ['orig ({f:.1f} Hz)'.format(f=f_change),
              'consonant ({f:.1f} Hz)'.format(f=cons_freq),
              'dissonant ({f:.1f} Hz)'.format(f=diss_freq),
              'hard bash ({f:.1f} Hz)'.format(f=hard_freq)
             ]

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
x = np.linspace(0,0.6,250)
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
ax1.set_title('Frequency bashing - base: {f1} Hz'.format(f1=f1))
ax1.set_xlabel('Modeled % CB difference')
ax1.set_ylabel('Roughness (unitless)')
ax1.legend()

# plot 2
ax2 = plt.subplot(1,2,2)
# dissonance function
x = np.linspace(0,0.6,250)
y = np.exp(-3.5*x) - np.exp(-5.75*x)
ax2.plot(x,y)

# starting roughness
f1 = 880
f2 = 910 # changing bc it's quieter

lines_x, lines_ymin, lines_ymax, labels = get_vline_params(f1, f2, 3)
for i in range(4) :
    ax2.vlines(lines_x[i], lines_ymin[i], lines_ymax[i],
               colors=colors[i], linestyles=linestyles[i], label=labels[i])

ax2.set_title('Frequency bashing - base: {f1} Hz'.format(f1=f1))
ax2.set_xlabel('Modeled % CB difference')
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
a1 = 10**(db1/20)
a2 = 10**(db2/20)

# masking curve
ax1 = plt.subplot(1,2,1)
ax1.vlines([f1,f2], [-70,-70], [db1,db2], colors='red', label='Original amplitudes')
x = np.linspace(400, 500, 250)
y = [db1-bu.get_masking_level_dB(f1, f) for f in x]
ax1.plot(x,y, label='Original masking curve')
ax1.set_xlim([400, 500])
ax1.set_ylim([-40, -19])
ax1.set_xlabel('Frequency')
ax1.set_ylabel('dB')

new_a1, new_a2 = bu.whack_amp(f1, a1, f2, a2, perc_move=1.0, consonance = True)
new_db1 = 20*np.log10(new_a1)
new_db2 = 20*np.log10(new_a2)
x_p = np.linspace(f1, f2, 250)
y_p = [new_db1-bu.get_masking_level_dB(f1, f) for f in x_p]
ax1.plot(x_p,y_p, 'g:', label='New masking curve')
ax1.scatter([f1,f2], [new_db1, new_db2], c='black', marker='x', label='Whacked amplitudes')
ax1.set_title('Amplitude whacking - lower freq. louder'.format(f1=f1))

ax1.legend()

f1 = 880
f2 = 910
db1 = -30
db2 = -20
a1 = 10**(db1/20)
a2 = 10**(db2/20)

ax2 = plt.subplot(1,2,2)
ax2.vlines([f1,f2], [-70,-70], [db1,db2], colors='red', label='Original amplitudes')
x = np.linspace(830, 990, 250)
y = [db2-bu.get_masking_level_dB(f2, f) for f in x]
ax2.plot(x,y, label='Original masking curve')
ax2.set_xlim([860, 960])
ax2.set_ylim([-40, -19])
ax2.set_xlabel('Frequency')
ax2.set_ylabel('dB')

new_a1, new_a2 = bu.whack_amp(f1, a1, f2, a2, perc_move=1.0, consonance = True)
new_db1 = 20*np.log10(new_a1)
new_db2 = 20*np.log10(new_a2)
x_p = np.linspace(f1, f2, 250)
y_p = [new_db2-bu.get_masking_level_dB(f2, f) for f in x_p]
ax2.plot(x_p,y_p, 'g:', label='New masking curve')
ax2.scatter([f1,f2], [new_db1, new_db2], c='black', marker='x', label='Whacked amplitudes')
ax2.set_title('Amplitude whacking - higher freq. louder'.format(f1=f1))
ax2.legend()

# for plotting reduction in roughness
'''
# roughness calc
a1s = np.linspace(a1,new_a1,100)
a2s = np.linspace(a1,new_a2,100)
roughnesses = bu.calculate_roughness_sethares(f1, a1s, f2, a2s)
#roughnesses = [bu.calculate_roughness_vassilakis(f1, a1s[i], f2, a2s[i]) for i in range(100)]
ax2.plot(np.linspace(0.0, 1.0, 100), roughnesses)
ax2.set_xlabel('Whack amount [0.0-1.0]')
ax2.set_ylabel('Roughness (unitless)')
ax2.set_title('Roughness calculation (Sethares 2005)')
'''

plt.savefig('amp_whacking.eps')
plt.clf()

bash_cmd_sine = 'python3 ../audio_basher_cf.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 {d_option} {h_option} audio_files/sin440_neg20.wav audio_files/sin470_neg30.wav'
mv_cmd = 'mv {in_name} audio_files/{out_name}'
whack_cmd_sine = 'python3 ../audio_whacker_cf.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 -whack_percent={p} audio_files/sin440_neg20.wav audio_files/sin470_neg30.wav'

# could make this a function if necessary
os.system(bash_cmd_sine.format(d_option='--consonance', h_option='--no-hard_bash'))
os.system(mv_cmd.format(in_name='vanilla.wav', out_name='sine440_470_vanilla.wav'))
os.system(mv_cmd.format(in_name='bashed.wav', out_name='sine440_470_consonance.wav'))
os.system(bash_cmd_sine.format(d_option='--no-consonance', h_option='--no-hard_bash'))
os.system(mv_cmd.format(in_name='bashed.wav', out_name='sine440_470_dissonance.wav'))
os.system(bash_cmd_sine.format(d_option='--consonance', h_option='--hard_bash'))
os.system(mv_cmd.format(in_name='bashed.wav', out_name='sine440_470_hard_bash.wav'))
os.system('rm filtered.wav')

os.system(whack_cmd_sine.format(p=0.0))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_0.wav'))
os.system(whack_cmd_sine.format(p=0.25))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_25.wav'))
os.system(whack_cmd_sine.format(p=0.5))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_50.wav'))
os.system(whack_cmd_sine.format(p=0.75))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_75.wav'))
os.system(whack_cmd_sine.format(p=1.0))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_100.wav'))
os.system('rm vanilla.wav')

# copy paste for second examples
bash_cmd_sine = 'python3 ../audio_basher_cf.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 {d_option} {h_option} audio_files/sin880_neg30.wav audio_files/sin910_neg20.wav'
mv_cmd = 'mv {in_name} audio_files/{out_name}'
whack_cmd_sine = 'python3 ../audio_whacker_cf.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 -whack_percent={p} audio_files/sin880_neg30.wav audio_files/sin910_neg20.wav'

os.system(bash_cmd_sine.format(d_option='--consonance', h_option='--no-hard_bash'))
os.system(mv_cmd.format(in_name='vanilla.wav', out_name='sine880_910_vanilla.wav'))
os.system(mv_cmd.format(in_name='bashed.wav', out_name='sine880_910_consonance.wav'))
os.system(bash_cmd_sine.format(d_option='--no-consonance', h_option='--no-hard_bash'))
os.system(mv_cmd.format(in_name='bashed.wav', out_name='sine880_910_dissonance.wav'))
os.system(bash_cmd_sine.format(d_option='--consonance', h_option='--hard_bash'))
os.system(mv_cmd.format(in_name='bashed.wav', out_name='sine880_910_hard_bash.wav'))
os.system('rm filtered.wav')

os.system(whack_cmd_sine.format(p=0.0))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_0.wav'))
os.system(whack_cmd_sine.format(p=0.25))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_25.wav'))
os.system(whack_cmd_sine.format(p=0.5))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_50.wav'))
os.system(whack_cmd_sine.format(p=0.75))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_75.wav'))
os.system(whack_cmd_sine.format(p=1.0))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_100.wav'))
os.system('rm vanilla.wav')
