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
    print('Please run and then try again: cd audio_files; python3 generate_wavs.py; cd ..')
    sys.exit(1)

def roughness_func_exp(f1,f2) :
    return (0.24 / (0.021 * min(f1,f2) + 19)) * abs(f1-f2)

def roughness_func_parncutt(f1,f2) :
    b_diff = abs(bu.hz_to_bark(f1) - bu.hz_to_bark(f2))
    return (np.exp(1)*b_diff*4 * np.exp(-b_diff*4))**2 if b_diff < 1.2 else 0

def roughness_func_hutchinson(f1,f2) :
    f_diff = np.abs(f1-f2)
    f_mean = (f1+f2) / 2.0
    b_diff = f_diff / (1.72*np.power(f_mean, 0.65))
    r = np.power(np.exp(1)*b_diff*4 * np.exp(-b_diff*4),2)
    if isinstance(r, np.ndarray) :
        r[b_diff >= 1.2] = 0
        return r
    else :
        return r if b_diff < 1.2 else 0

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

    labels = ['original ({f:.1f} Hz)'.format(f=f_change),
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
# EQUATION FIGURES
# ----------------------
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
# dissonance function
x = np.linspace(0,1.2,500)
y1 = np.exp(-3.5*x) - np.exp(-5.75*x)
y1p = y1 / np.max(y1)
ax1.plot(x,y1p)
ax1.set_title('Roughness of two sinusoids of same amplitude (Sethares)')
ax1.set_xlabel('Modeled % Crit. Band difference')
ax1.set_ylabel('Roughness (unitless)')

y2 = np.power(np.exp(1)*x*4 * np.exp(-x*4), 2)
y2p = y2 / np.max(y2)
ax2.plot(x,y2p)
ax2.set_title('Roughness of two sinusoids of same amplitude (Parncutt)')
ax2.set_xlabel('Difference in Barks')
ax2.set_ylabel('Roughness (unitless)')

f.tight_layout()
plt.savefig('roughness_equations.pdf')
plt.savefig('roughness_equations.png')
plt.clf()

f, ax1 = plt.subplots(1, 1, figsize=(6,6))
# dissonance function
ax1.plot(x,y1p-y2)
ax2.set_title('Difference in roughness equations')
ax2.set_xlabel('Critical band difference')
ax2.set_ylabel('Difference in roughness')

f.tight_layout()
plt.savefig('roughness_difference.pdf')
plt.savefig('roughness_difference.png')
plt.clf()


x = np.linspace(440, bu.bark_to_hz(bu.hz_to_bark(440) + 1.2), 500)
max_seth = np.max(bu.calculate_roughness_sethares(440, 1.0, x, 1.0))
max_vass = np.max(bu.calculate_roughness_vassilakis(440, 1.0, x, 1.0))

y1 = bu.calculate_roughness_vassilakis(440, 0.9, x, 0.9)
y2 = bu.calculate_roughness_sethares(440, 0.9, x, 0.9)
y3 = bu.calculate_roughness_vassilakis(440, 1.0, x, 0.5)
y4 = bu.calculate_roughness_sethares(440, 1.0, x, 0.5)

y1p = y1 / max_vass
y2p = y2 / max_seth
y3p = y3 / max_vass
y4p = y4 / max_seth

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey='row', sharex='col', figsize=(14,12))
ax1.plot(x,y1p)
ax1.set_title('Vassilakis: Roughness of two tones with amplitudes 0.9, 0.9')
ax1.set_ylim([0, 1.0])
ax2.plot(x,y2p)
ax2.set_title('Sethares: Roughness of two tones with amplitudes 0.9, 0.9')
ax2.set_ylim([0, 1.0])
ax3.plot(x,y3p)
ax3.set_title('Vassilakis: Roughness of two tones with amplitudes 1.0, 0.5')
ax3.set_ylim([0, 1.0])
ax4.plot(x,y4p)
ax4.set_title('Sethares: Roughness of two tones with amplitudes 1.0, 0.5')
ax4.set_ylim([0, 1.0])
ax3.set_xlabel('Frequency of second partial (first partial 440 Hz)')
ax4.set_xlabel('Frequency of second partial (first partial 440 Hz)')

ax1.set_ylabel('Roughness (unitless)')
ax3.set_ylabel('Roughness (unitless)')
f.tight_layout()
plt.savefig('roughness_vass.pdf')
plt.savefig('roughness_vass.png')
plt.clf()

# masking
f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(14,6))
ax1.vlines(440, -70, -10, colors='red', label='Masking sinusoid')
x = np.linspace(240, 740, 500)
y = [-10-bu.get_masking_level_dB(440, f) for f in x]
ax1.plot(x,y)
ax1.set_xlim([240, 740])
ax1.set_ylim([-50, -9])
ax1.set_title('Masking curve (440 Hz, -10 dB)')
ax1.set_xlabel('Frequency')
ax1.set_ylabel('dB')
ax1.legend(fontsize='x-large')

ax2.vlines(880, -70, -10, colors='red', label='Masking sinusoid')
x = np.linspace(680, 1080, 500)
y = [-10-bu.get_masking_level_dB(880, f) for f in x]
ax2.plot(x,y)
ax2.set_xlim([680, 1080])
ax2.set_ylim([-50, -9])
ax2.set_title('Masking curve (880 Hz, -10 dB)')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('dB')
ax2.legend(fontsize='x-large')
f.tight_layout()
plt.savefig('masking_curves.pdf')
plt.savefig('masking_curves.png')
plt.clf()

n1 = 440 * np.arange(1,21,2)
n2 = 440 * (3/2) * np.arange(1,21,2)
n3 = 440 * (5/3) * np.arange(1,21,2)
r1 = 0
for p1 in n1 :
    for p2 in n2 :
        r1 += roughness_func_hutchinson(p1,p2)

for i in range(10) :
    for j in range(i+1,10) :
        r1 += roughness_func_hutchinson(n1[i], n1[j])
        r1 += roughness_func_hutchinson(n2[i], n2[j])
r2 = 0
for p1 in n1 :
    for p2 in n3 :
        r2 += roughness_func_hutchinson(p1,p2)

for i in range(10) :
    for j in range(i+1,10) :
        r2 += roughness_func_hutchinson(n1[i], n1[j])
        r2 += roughness_func_hutchinson(n3[i], n3[j])

print('Perfect fifth roughness: {r}'.format(r=r1))
print('Perfect sixth roughness: {r}'.format(r=r2))

'''
# ----------------------
# BASHING FIGURES
# ----------------------

# plot 1

f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(14,6))

#ax1 = plt.subplot(1,2,1)
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

ax1.annotate('Allowable range in Barks', xy=(0.336, 0.70), xytext=(0.336, 0.80), xycoords='axes fraction', 
            fontsize=12.0, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=8.5, lengthB=1.5', lw=2.0))
#ax1.vlines(lines_x, lines_ymin, lines_ymax, 
#           colors=['red', 'orange', 'green', 'purple'],
#           linestyles=['solid', 'dashed', 'dashdot', 'dotted'],
#           label = ['orig', 'consonant', 'dissonant', 'hard bashed'])
ax1.set_title('Frequency bashing - base: {f1} Hz'.format(f1=f1))
ax1.set_xlabel('Modeled % CB difference')
ax1.set_ylabel('Roughness (unitless)')
ax1.legend(fontsize='x-large')

# plot 2
#ax2 = plt.subplot(1,2,2)
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
ax2.legend(fontsize='x-large')
ax2.annotate('Allowable range in Barks', xy=(0.336, 0.70), xytext=(0.336, 0.80), xycoords='axes fraction', 
            fontsize=12.0, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=8.5, lengthB=1.5', lw=2.0))
f.tight_layout()
plt.savefig('freq_bashing.pdf')
plt.savefig('freq_bashing.png')
plt.clf()

# ----------------------
# WHACKING FIGURES
# ----------------------
f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(14,6))
f1 = 440
f2 = 470
db1 = -20
db2 = -30
a1 = 10**(db1/20)
a2 = 10**(db2/20)

# masking curve
ax1.vlines([f1,f2], [-70,-70], [db1,db2], colors='red', label='Original amplitudes')
x = np.linspace(400, 550, 250)
y = [db1-bu.get_masking_level_dB(f1, f) for f in x]
ax1.plot(x,y, label='Original masking curve')
ax1.set_xlim([400, 550])
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

ax1.legend(fontsize='x-large')

f1 = 880
f2 = 910
db1 = -30
db2 = -20
a1 = 10**(db1/20)
a2 = 10**(db2/20)

ax2.vlines([f1,f2], [-70,-70], [db1,db2], colors='red', label='Original amplitudes')
x = np.linspace(850, 1050, 250)
y = [db2-bu.get_masking_level_dB(f2, f) for f in x]
ax2.plot(x,y, label='Original masking curve')
ax2.set_xlim([850, 1050])
ax2.set_ylim([-40, -19])
ax2.set_xlabel('Frequency')

new_a1, new_a2 = bu.whack_amp(f1, a1, f2, a2, perc_move=1.0, consonance = True)
new_db1 = 20*np.log10(new_a1)
new_db2 = 20*np.log10(new_a2)
x_p = np.linspace(f1, f2, 250)
y_p = [new_db2-bu.get_masking_level_dB(f2, f) for f in x_p]
ax2.plot(x_p,y_p, 'g:', label='New masking curve')
ax2.scatter([f1,f2], [new_db1, new_db2], c='black', marker='x', label='Whacked amplitudes')
ax2.set_title('Amplitude whacking - higher freq. louder'.format(f1=f1))
ax2.legend(fontsize='x-large')
f.tight_layout()

# for plotting reduction in roughness
plt.savefig('amp_whacking.pdf')
plt.savefig('amp_whacking.png')
plt.clf()

bash_cmd_sine = 'python3 ../audio_basher.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 {d_option} {h_option} audio_files/sin440_neg20.wav audio_files/sin470_neg30.wav'
mv_cmd = 'mv {in_name} audio_files/{out_name}'
whack_cmd_sine = 'python3 ../audio_whacker.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 -whack_percent={p} audio_files/sin440_neg20.wav audio_files/sin470_neg30.wav'

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
os.system(whack_cmd_sine.format(p=0.33))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_33.wav'))
os.system(whack_cmd_sine.format(p=0.66))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_66.wav'))
os.system(whack_cmd_sine.format(p=1.0))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine440_470_whack_100.wav'))
os.system('rm vanilla.wav')

# copy paste for second examples
bash_cmd_sine = 'python3 ../audio_basher.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 {d_option} {h_option} audio_files/sin880_neg30.wav audio_files/sin910_neg20.wav'
mv_cmd = 'mv {in_name} audio_files/{out_name}'
whack_cmd_sine = 'python3 ../audio_whacker.py -nsines=1 -bw_percent_low=0.05 -bw_percent_high=0.35 -whack_percent={p} audio_files/sin880_neg30.wav audio_files/sin910_neg20.wav'

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
os.system(whack_cmd_sine.format(p=0.33))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_33.wav'))
os.system(whack_cmd_sine.format(p=0.66))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_66.wav'))
os.system(whack_cmd_sine.format(p=1.0))
os.system(mv_cmd.format(in_name='whacked.wav', out_name='sine880_910_whack_100.wav'))
os.system('rm vanilla.wav')

# tuning examples
os.system('python3 ../audio_basher.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --consonance audio_files/saw_root.wav audio_files/saw_third_equal.wav audio_files/saw_fifth_equal.wav')
os.system('mv vanilla.wav audio_files/saw_equal_simple_chord.wav')
os.system('rm filtered.wav')
os.system('mv bashed.wav audio_files/saw_equal_simple_bashed.wav')
os.system('python3 ../audio_basher.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --consonance audio_files/saw_root.wav audio_files/saw_third_just.wav audio_files/saw_fifth_just.wav')
os.system('mv vanilla.wav audio_files/saw_just_simple_chord.wav')
os.system('rm filtered.wav')
os.system('rm bashed.wav')
s1,fs1 = sf.read('audio_files/saw_equal_simple_chord.wav')
s2,fs2 = sf.read('audio_files/saw_equal_simple_bashed.wav')
sf.write('audio_files/saw_equal_simple_diff.wav', s1-s2, fs1)
os.system('python3 ../audio_basher.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --consonance audio_files/saw_root.wav audio_files/saw_third_equal.wav audio_files/saw_fifth_equal.wav audio_files/saw_seven_equal.wav')
os.system('mv vanilla.wav audio_files/saw_equal_complex_chord.wav')
os.system('rm filtered.wav')
os.system('mv bashed.wav audio_files/saw_equal_complex_bashed.wav')
os.system('python3 ../audio_basher.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --consonance audio_files/saw_root.wav audio_files/saw_third_just.wav audio_files/saw_fifth_just.wav audio_files/saw_seven_just.wav')
os.system('mv vanilla.wav audio_files/saw_just_complex_chord.wav')
os.system('rm filtered.wav')
os.system('rm bashed.wav')
os.system('python3 ../audio_basher.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --no-consonance audio_files/saw_root.wav audio_files/saw_third_equal.wav audio_files/saw_fifth_equal.wav audio_files/saw_seven_equal.wav')
os.system('rm vanilla.wav')
os.system('rm filtered.wav')
os.system('mv bashed.wav audio_files/saw_complex_dissonant.wav')
s1,fs1 = sf.read('audio_files/saw_equal_complex_chord.wav')
s2,fs2 = sf.read('audio_files/saw_equal_complex_bashed.wav')
sf.write('audio_files/saw_equal_complex_diff.wav', s1-s2, fs1)

# some dynamic examples
# detuned
os.system('python3 ../audio_basher.py -nsines=7 -delta=3 -bw_percent_low=0.01 -bw_percent_high=0.35 --normalize --consonance ../audio/detuned/*')
tmp_s1, fs1 = sf.read('vanilla.wav')
tmp_s2, fs2 = sf.read('bashed.wav')
sf.write('audio_files/detuned_difference.wav', tmp_s1-tmp_s2, fs1)
os.system('mv vanilla.wav audio_files/detuned_vanilla.wav')
os.system('mv bashed.wav audio_files/detuned_bashed.wav')
os.system('rm filtered.wav')
# horns
os.system('python3 ../audio_whacker.py -nsines=7 -bw_percent_low=0.01 -bw_percent_high=0.35 --normalize --consonance ../audio/rodrigobonelli_balladforlaura_medleydb/*')
tmp_s1, fs1 = sf.read('vanilla.wav')
tmp_s2, fs2 = sf.read('whacked.wav')
sf.write('audio_files/horns_difference.wav', tmp_s1-tmp_s2, fs1)
os.system('mv vanilla.wav audio_files/horns_vanilla.wav')
os.system('mv whacked.wav audio_files/horns_whacked.wav')
# minor chord
os.system('python3 ../audio_basher.py -nsines=10 -delta=3 -bw_percent_low=0.01 -bw_percent_high=0.35 --normalize --consonance ../audio/chord2/*')
tmp_s1, fs1 = sf.read('vanilla.wav')
tmp_s2, fs2 = sf.read('bashed.wav')
sf.write('audio_files/minor_chord_difference.wav', tmp_s1-tmp_s2, fs1)
os.system('mv vanilla.wav audio_files/minor_chord_vanilla.wav')
os.system('mv filtered.wav audio_files/minor_chord_filtered.wav')
os.system('rm bashed.wav')
# choir dissonance
os.system('python3 ../audio_basher.py -nsines=7 -delta=3 -bw_percent_low=0.01 -bw_percent_high=0.4 --normalize --no-consonance -roughness_thresh=0.0005 ../audio/vocal_large/*')
tmp_s1, fs1 = sf.read('vanilla.wav')
tmp_s2, fs2 = sf.read('bashed.wav')
sf.write('audio_files/choir_difference.wav', tmp_s1-tmp_s2, fs1)
os.system('mv vanilla.wav audio_files/choir_vanilla.wav')
os.system('mv bashed.wav audio_files/choir_bashed.wav')
os.system('rm filtered.wav')
# tremolo effect
os.system('python3 ../audio_basher.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --consonance -delta=3 --hard_bash -roughness_thresh=0.0001 audio_files/saw_root.wav audio_files/saw_third_equal.wav audio_files/saw_fifth_equal.wav')
tmp_s1, fs1 = sf.read('vanilla.wav')
mod = 0.1*np.sin(2*np.pi*3.2*np.arange(tmp_s1.shape[0])/fs1) + 0.9
sf.write('audio_files/major_chord_tremolo.wav', tmp_s1*mod, fs1)
os.system('mv vanilla.wav audio_files/major_chord_vanilla.wav')
os.system('mv bashed.wav audio_files/major_chord_hard_bash.wav')
os.system('rm filtered.wav')

# spectrogram
s1,fs1 = sf.read('audio_files/saw_equal_simple_chord.wav')
s2,fs2 = sf.read('audio_files/saw_equal_simple_bashed.wav')

S1 = librosa.stft(s1, n_fft = 8192)
S2 = librosa.stft(s2, n_fft = 8192)

D1 = librosa.amplitude_to_db(np.abs(S1), ref=np.max)
D2 = librosa.amplitude_to_db(np.abs(S2), ref=np.max)
D3 = np.abs(S1-S2)
freqs = librosa.fft_frequencies(sr=fs1, n_fft = 8192)
P3 = librosa.perceptual_weighting(D3, freqs)

f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(14,6))
librosa.display.specshow(D1, y_axis='log', x_axis='time', sr=fs1, ax=ax1, hop_length=8192//4)
librosa.display.specshow(D3, y_axis='log', x_axis='time', sr=fs1, ax=ax2, hop_length=8192//4)

ax1.set_title('Vanilla Signal')
ax2.set_title('Signal Difference After Frequency Bashing')

f.tight_layout()
plt.savefig('spect_difference.pdf')
plt.savefig('spect_difference.png')
plt.clf()

# spectrogram 2
s1,fs1 = sf.read('audio_files/horns_vanilla.wav')
s2,fs2 = sf.read('audio_files/horns_whacked.wav')

S1 = librosa.stft(s1, n_fft = 4096)
S2 = librosa.stft(s2, n_fft = 4096)

D1 = librosa.amplitude_to_db(np.abs(S1), ref=np.max)
D2 = librosa.amplitude_to_db(np.abs(S2), ref=np.max)
D3 = np.abs(S1-S2)
freqs = librosa.fft_frequencies(sr=fs1, n_fft = 4096)
P3 = librosa.perceptual_weighting(D3, freqs)

f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(14,6))
librosa.display.specshow(D1, y_axis='log', x_axis='time', sr=fs1, ax=ax1, hop_length=4096//4)
librosa.display.specshow(D3, y_axis='log', x_axis='time', sr=fs1, ax=ax2, hop_length=4096//4)

ax1.set_title('Vanilla Signal')
ax2.set_title('Signal Difference After Amplitude Whacking')

f.tight_layout()
plt.savefig('spect_difference2.pdf')
plt.savefig('spect_difference2.png')
plt.clf()



'''
