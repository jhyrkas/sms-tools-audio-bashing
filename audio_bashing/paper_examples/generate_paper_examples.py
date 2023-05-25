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
    return (0.24 / (0.021 * np.minimum(f1,f2) + 19)) * np.abs(f1-f2)

def get_roughness_x_y(f1, f2) :
    x = roughness_func_exp(f1, f2)
    y = bu.calculate_roughness_sethares(f1, 1.0, f2, 1.0)
    return x,y

def get_erb(f) :
    f_khz = f/1000
    return 6.23*np.power(f_khz,2) + 93.39*f_khz + 28.52

def get_bark_distance_wang(f1, f2) :
    b_f1 = 6*np.arcsinh(f1/600)
    b_f2 = 6*np.arcsinh(f2/600)
    return np.abs(b_f1-b_f2)

def get_bark_distance_zwicker(f1, f2) :
    b_f1 = 13*np.arctan(0.00076*f1) + 3.5*np.arctan(np.power(f1/7500, 2))
    b_f2 = 13*np.arctan(0.00076*f2) + 3.5*np.arctan(np.power(f2/7500, 2))
    return np.abs(b_f1-b_f2)

def cbw_hutchinson(f) :
    return 1.72 * np.power(f, 0.65)

def get_cbw_diff_hutchinson(f1, f2) :
    return (np.abs(f1-f2)) / cbw_hutchinson((f1+f2)*0.5)

def get_cbw_diff_erb(f1, f2) :
    return (np.abs(f1-f2)) / get_erb((f1+f2)*0.5)

def get_vline_params(f_const, f_change, delta) :
    lines_x = []
    lines_ymin = []
    lines_ymax = []
    # starting roughness
    x,y = get_roughness_x_y(f_const, f_change)
    #lines_x.append(x)
    lines_x.append(f_change)
    lines_ymin.append(0)
    lines_ymax.append(y)

    # maximum consonance
    cons_freq = bu.bash_freq(f_change, f_const, 0.05, 0.35, hard_bash=False, delta=delta, consonance = True)
    x,y = get_roughness_x_y(f_const, cons_freq)
    #lines_x.append(x)
    lines_x.append(cons_freq)
    lines_ymin.append(0)
    lines_ymax.append(y)

    # maximum dissonance
    diss_freq = bu.bash_freq(f_change, f_const, 0.05, 0.35, hard_bash=False, delta=delta, consonance = False)
    x,y = get_roughness_x_y(f_const, diss_freq)
    #lines_x.append(x)
    lines_x.append(diss_freq)
    lines_ymin.append(0)
    lines_ymax.append(y)

    # hard bashing
    hard_freq = bu.bash_freq(f_change, f_const, 0.05, 0.35, hard_bash=True, delta=delta, consonance = True)
    x,y = get_roughness_x_y(f_const, hard_freq)
    #lines_x.append(x)
    lines_x.append(hard_freq)
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
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,3))

# dissonance function
base_fs = [125, 250, 500, 1000, 2000]
line_styles = ['-', '--', '-.', ':', (0, (5, 10))]
for i in range(4,-1,-1) :
    x_log = np.logspace(0, 1, 500, base=2.0)
    y = bu.calculate_roughness_sethares(base_fs[i], 1.0, x_log*base_fs[i],1.0)
    ax1.plot(x_log,y, label = 'Base freq: {b}'.format(b=base_fs[i]), linestyle=line_styles[i])
ax1.legend()
ax1.set_title('Roughness curves of different base frequencies')
ax1.set_xlabel('Frequency Ratio')
ax1.set_ylabel('Roughness')

ax2.vlines(440, 0, 70, colors='red', label='Masking sinusoid')
x = np.linspace(350, 640, 500)
y = [70-bu.get_masking_level_dB(440, f) for f in x]
ax2.plot(x,y)
ax2.set_xlim([350, 640])
ax2.set_ylim([30, 71])
ax2.set_title('Masking curve (440 Hz, 70 dB)')
ax2.set_xlabel('Frequency')
ax2.set_ylabel('dB')
ax2.legend(fontsize='x-large')
f.tight_layout()
plt.savefig('equations.pdf')
plt.savefig('equations.png')
plt.clf()

# new sethares plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6), sharey='row')
base_fs = [125, 250, 500, 1000, 2000]
for i in range(4,-1,-1) :
    x_log = np.logspace(0, 1, 500, base=2.0)
    y = bu.calculate_roughness_sethares(base_fs[i], 1.0, x_log*base_fs[i],1.0)
    ax1.plot(x_log,y, label = 'Base freq: {b}'.format(b=base_fs[i]))
ax1.legend()
ax1.set_title('Sethares diss. as function of interval')
ax1.set_xlabel('Interval Ratio')
ax1.set_ylabel('Roughness')
# modeled CB distance
for i in range(4,-1,-1) :
    x_log = np.logspace(0, np.log2(3/2), 500, base=2.0)
    new_x = roughness_func_exp(base_fs[i], x_log*base_fs[i])
    y = np.exp(-3.5*new_x) - np.exp(-5.75*new_x)
    ax2.plot(new_x,y, label = 'Base freq: {b}'.format(b=base_fs[i]))
ax2.legend()
ax2.set_title('Sethares diss. as function of modeled CB distance')
ax2.set_xlabel('Sethares exponent (f1,f2)')
ax2.set_ylabel('Roughness')
f.tight_layout()
plt.savefig('sethares_axes.png')
plt.savefig('sethares_axes.pdf')
plt.clf()

# CB equations
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11,8), sharey='row', sharex='col')

fs = [500, 1500]
axs = [ax1,ax2,ax3,ax4]
for i in range(2) :
    x = np.linspace(fs[i], fs[i]+200, 500)
    y_seth = roughness_func_exp(fs[i], x)
    y_erb = get_cbw_diff_erb(fs[i], x)
    y_bark = bu.get_bark_diff(fs[i],x)
    y_hutch = get_cbw_diff_hutchinson(fs[i], x)
    ax1 = axs[i]
    ax1.plot(x,y_seth,'b-',label = 'Sethares exponent')
    ax1.plot(x,y_erb, 'k-', label = 'Hutchinson exponent (ERB)')
    ax1.plot(x,y_bark, 'g-', label = 'Bark difference (Traunmuller)')
    ax1.plot(x,y_hutch, 'r-', label = 'Hutchinson exponent (CBW)')
    ax1.legend()
    ax1.set_title('Frequency Difference as Critical Bandwidth')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Critical Bandwidth Estimate')

    ax2 = axs[i+2]
    seth_norm = 1/(np.exp(-3.5*0.24) - np.exp(-5.75*0.24))
    y_seth=seth_norm*(np.exp(-3.5*y_seth)-np.exp(-5.75*y_seth))
    y_erb=seth_norm*(np.exp(-3.5*y_erb)-np.exp(-5.75*y_erb))
    y_bark=seth_norm*(np.exp(-3.5*y_bark)-np.exp(-5.75*y_bark))
    y_hutch=seth_norm*(np.exp(-3.5*y_hutch)-np.exp(-5.75*y_hutch))
    ax2.plot(x, y_seth, 'b-', label='Sethares exponent')
    ax2.plot(x, y_erb, 'k-', label='Hutchinson exponent (ERB)')
    ax2.plot(x, y_bark, 'g-', label='Bark difference (Traunmuller)')
    ax2.plot(x, y_hutch, 'r-', label='Hutchinson exponent (CBW)')
    ax2.vlines(x[np.argmax(y_seth)], 0, 1, 'blue', 'dotted')
    ax2.vlines(x[np.argmax(y_erb)], 0, 1, 'black', 'dotted')
    ax2.vlines(x[np.argmax(y_bark)], 0, 1, 'green', 'dotted')
    ax2.vlines(x[np.argmax(y_hutch)], 0, 1, 'red', 'dotted')
    ax2.legend()
    ax2.set_title('Roughness Estimate by CB Model')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Roughness model (Sethares)')

f.tight_layout()
plt.savefig('cb_models.pdf')
plt.clf()

# ----------------------
# BASHING FIGURES
# ----------------------

# plot 1

f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(14,4))

#ax1 = plt.subplot(1,2,1)
# dissonance function
#x = np.linspace(0,0.6,250)
#y = np.exp(-3.5*x) - np.exp(-5.75*x)
#ax1.plot(x,y)

# starting roughness
f1 = 440
f2 = 470 # changing bc it's quieter

x = np.linspace(f1, f1+100, 500)
y = bu.calculate_roughness_sethares(f1, 1.0, x, 1.0)
ax1.plot(x,y,label='Roughness curve')

colors=['red', 'orange', 'green', 'purple']
linestyles=[(5, (10, 3)), 'dashed', 'dashdot', 'dotted']

lines_x, lines_ymin, lines_ymax, labels = get_vline_params(f1, f2, 3)

for i in range(4) :
    ax1.vlines(lines_x[i], lines_ymin[i], lines_ymax[i],
               colors=colors[i], linestyles=linestyles[i], label=labels[i])

ax1.annotate('Allowable range in Barks', xy=(0.279, 0.70), xytext=(0.279, 0.80), xycoords='axes fraction', 
            fontsize=12.0, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=7.0, lengthB=1.5', lw=2.0))
#ax1.vlines(lines_x, lines_ymin, lines_ymax, 
#           colors=['red', 'orange', 'green', 'purple'],
#           linestyles=['solid', 'dashed', 'dashdot', 'dotted'],
#           label = ['orig', 'consonant', 'dissonant', 'hard bashed'])
ax1.set_title('Frequency bashing - base: {f1} Hz'.format(f1=f1))
ax1.set_xlabel('Frequency')
ax1.set_ylabel('Roughness')
ax1.legend(fontsize='x-large', loc=4)

# plot 2
#ax2 = plt.subplot(1,2,2)
# dissonance function
#x = np.linspace(0,0.6,250)
#y = np.exp(-3.5*x) - np.exp(-5.75*x)
#ax2.plot(x,y)

# starting roughness
f1 = 880
f2 = 910 # changing bc it's quieter

x = np.linspace(f1, f1+100, 500)
y = bu.calculate_roughness_sethares(f1, 1.0, x, 1.0)
ax2.plot(x,y,label='Roughness curve')

lines_x, lines_ymin, lines_ymax, labels = get_vline_params(f1, f2, 3)
for i in range(4) :
    ax2.vlines(lines_x[i], lines_ymin[i], lines_ymax[i],
               colors=colors[i], linestyles=linestyles[i], label=labels[i])

ax2.set_title('Frequency bashing - base: {f1} Hz'.format(f1=f1))
ax2.set_xlabel('Frequency')
ax2.legend(fontsize='x-large', loc=4)
ax2.annotate('Allowable range in Barks', xy=(0.351, 0.70), xytext=(0.351, 0.80), xycoords='axes fraction', 
            fontsize=12.0, ha='center', va='bottom',
            bbox=dict(boxstyle='square', fc='white'),
            arrowprops=dict(arrowstyle='-[, widthB=9.2, lengthB=1.5', lw=2.0))
f.tight_layout()
plt.savefig('freq_bashing.pdf')
plt.savefig('freq_bashing.png')
plt.clf()

# ----------------------
# WHACKING FIGURES
# ----------------------
f, (ax1, ax2) = plt.subplots(1, 2, sharey='row', figsize=(12,4))
f1 = 440
f2 = 470
db1 = -20
db2 = -30
a1 = 10**(db1/20)
a2 = 10**(db2/20)

# for plotting in positive dB
plot_offset = 60

# masking curve
ax1.vlines([f1,f2], [-70+plot_offset,-70+plot_offset], [db1+plot_offset,db2+plot_offset], colors='red', label='Original amplitudes')
x = np.linspace(400, 550, 250)
y = [db1+plot_offset-bu.get_masking_level_dB(f1, f) for f in x]
ax1.plot(x,y, label='Original masking curve')
ax1.set_xlim([400, 550])
ax1.set_ylim([-40+plot_offset, -19+plot_offset])
ax1.set_xlabel('Frequency')
ax1.set_ylabel('dB')

new_a1, new_a2 = bu.whack_amp(f1, a1, f2, a2, perc_move=1.0, consonance = True)
new_db1 = 20*np.log10(new_a1)
new_db2 = 20*np.log10(new_a2)
x_p = np.linspace(f1, f2, 250)
y_p = [new_db1+plot_offset-bu.get_masking_level_dB(f1, f) for f in x_p]
ax1.scatter([f1,f2], [new_db1+plot_offset, new_db2+plot_offset], c='black', marker='x', label='Whacked amplitudes')
ax1.plot(x_p,y_p, 'g:', label='New masking curve')
ax1.set_title('Amplitude whacking - lower freq. louder'.format(f1=f1))

ax1.legend(fontsize='x-large')

f1 = 880
f2 = 910
db1 = -30
db2 = -20
a1 = 10**(db1/20)
a2 = 10**(db2/20)

ax2.vlines([f1,f2], [-70+plot_offset,-70+plot_offset], [db1+plot_offset,db2+plot_offset], colors='red', label='Original amplitudes')
x = np.linspace(850, 1050, 250)
y = [db2+plot_offset-bu.get_masking_level_dB(f2, f) for f in x]
ax2.plot(x,y, label='Original masking curve')
ax2.set_xlim([850, 1050])
ax2.set_ylim([-40+plot_offset, -19+plot_offset])
ax2.set_xlabel('Frequency')

new_a1, new_a2 = bu.whack_amp(f1, a1, f2, a2, perc_move=1.0, consonance = True)
new_db1 = 20*np.log10(new_a1)
new_db2 = 20*np.log10(new_a2)
x_p = np.linspace(f1, f2, 250)
y_p = [new_db2+plot_offset-bu.get_masking_level_dB(f2, f) for f in x_p]
ax2.scatter([f1,f2], [new_db1+plot_offset, new_db2+plot_offset], c='black', marker='x', label='Whacked amplitudes')
ax2.plot(x_p,y_p, 'g:', label='New masking curve')
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
print('TUNING EXAMPLES')
print('---------------')
os.system('python3 ../audio_basher.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --consonance audio_files/saw_root.wav audio_files/saw_third_equal.wav audio_files/saw_fifth_equal.wav')
os.system('mv vanilla.wav audio_files/saw_equal_simple_chord.wav')
os.system('rm filtered.wav')
os.system('mv bashed.wav audio_files/saw_equal_simple_bashed.wav')
os.system('python3 ../audio_whacker.py -nsines=20 -bw_percent_low=0.001 -bw_percent_high=0.35 --normalize --consonance audio_files/saw_root.wav audio_files/saw_third_equal.wav audio_files/saw_fifth_equal.wav')
os.system('rm vanilla.wav')
os.system('mv whacked.wav audio_files/saw_equal_simple_whacked.wav')
print('---------------')
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

# spectrogram theoretical

def saw_params(f0) :
    amp = 1.0
    freqs = []
    amps = []
    for h in range(1,11) :
        freqs.append(h*f0)
        amps.append(1/h * amp)
    return freqs, amps

srf,sra = saw_params(220)
stf,sta = saw_params(220*(2**(4/12)))
sff,sfa = saw_params(220*(2**(7/12)))
vanilla_freqs = []
vanilla_amps = []
for i in range(10) :
    vanilla_freqs.append(srf[i])
    vanilla_amps.append(sra[i])
for i in range(10) :
    vanilla_freqs.append(stf[i])
    vanilla_amps.append(sta[i])
for i in range(10) :
    vanilla_freqs.append(sff[i])
    vanilla_amps.append(sfa[i])

bashed_freqs = vanilla_freqs.copy()
whacked_freqs = vanilla_freqs.copy()
bashed_amps = vanilla_amps.copy()
whacked_amps = vanilla_amps.copy()

#bashed_freqs[3] = 831.76 # root harmonic 4
#bashed_freqs[24] = 1319.07 # third harmonic 5
#bashed_freqs[4] = 1108.9 # root harmonic 5

bashed_freqs[3] = bu.bash_freq(vanilla_freqs[3], vanilla_freqs[12], 0.03, 0.35, False)
bashed_freqs[14] = bu.bash_freq(vanilla_freqs[14], vanilla_freqs[23], 0.03, 0.35, False)
bashed_freqs[4] = bu.bash_freq(vanilla_freqs[4], vanilla_freqs[13], 0.03, 0.35, False)

tmp_freqs1 = np.array([bashed_freqs[3], bashed_freqs[14], bashed_freqs[4]]).copy()
tmp_amps1 = np.array([bashed_amps[3], bashed_amps[14], bashed_amps[4]]).copy()

whacked_amps[3], whacked_amps[12] = bu.whack_amp(vanilla_freqs[3], vanilla_amps[3], vanilla_freqs[12], vanilla_amps[12], 1.0)
whacked_amps[14], whacked_amps[23] = bu.whack_amp(vanilla_freqs[14], vanilla_amps[14], vanilla_freqs[23], vanilla_amps[23], 1.0)
whacked_amps[4], whacked_amps[13] = bu.whack_amp(vanilla_freqs[4], vanilla_amps[4], vanilla_freqs[13], vanilla_amps[13], 1.0)

print(vanilla_amps[3])
print(vanilla_amps[12])
print(bu.whack_amp(vanilla_freqs[3], vanilla_amps[3], vanilla_freqs[12], vanilla_amps[12], 1.0))
print(whacked_amps[3])
print(whacked_amps[12])

tmp_freqs2 = np.array([whacked_freqs[3], whacked_freqs[12], whacked_freqs[14], whacked_freqs[23], whacked_freqs[4], whacked_freqs[13]]).copy()
tmp_amps2 = np.array([whacked_amps[3], whacked_amps[12], whacked_amps[14], whacked_amps[23], whacked_amps[4], whacked_amps[13]]).copy()

vanilla_freqs = np.array(vanilla_freqs)
vanilla_amps = np.array(vanilla_amps)
whacked_freqs = np.array(whacked_freqs)
whacked_amps = np.array(whacked_amps)
bashed_freqs = np.array(bashed_freqs)
bashed_amps = np.array(bashed_amps)

vanilla_amps = 20*np.log10(vanilla_amps)
whacked_amps = 20*np.log10(whacked_amps)
bashed_amps = 20*np.log10(bashed_amps)

#original_amps1 = (20*np.log10(original_amps1)) - np.min(vanilla_amps)
#original_amps2 = (20*np.log10(original_amps2)) - np.min(vanilla_amps)
#vanilla_amps -= np.min(vanilla_amps)
#bashed_amps -= np.min(vanilla_amps)
#whacked_amps -= np.min(vanilla_amps)
#original_amps1 = (20*np.log10(original_amps1)) + 40
#original_amps2 = (20*np.log10(original_amps2)) + 40
vanilla_amps += 40
bashed_amps += 40
whacked_amps += 40
tmp_amps1 = 20*np.log10(tmp_amps1) + 40
tmp_amps2 = 20*np.log10(tmp_amps2) + 40

sort_order = np.argsort(vanilla_freqs)
vanilla_freqs = vanilla_freqs[sort_order]
bashed_freqs = bashed_freqs[sort_order]
whacked_freqs = whacked_freqs[sort_order]
vanilla_amps = vanilla_amps[sort_order]
bashed_amps = bashed_amps[sort_order]
whacked_amps = whacked_amps[sort_order]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4), sharey='row')
ax1.stem(vanilla_freqs, vanilla_amps)
ax2.stem(vanilla_freqs, vanilla_amps)
ax2.stem(tmp_freqs1, tmp_amps1, linefmt='r--', markerfmt='x',label='bashed frequencies')
ax3.stem(vanilla_freqs, vanilla_amps)
ax3.stem(tmp_freqs2, tmp_amps2, linefmt='r--', markerfmt='x',label='whacked amplitudes')
ax1.set_ylabel('dB')
#ax2.set_ylabel('dB')
#ax3.set_ylabel('dB')
ax1.set_xlabel('Frequency (Hz)')
ax2.set_xlabel('Frequency (Hz)')
ax3.set_xlabel('Frequency (Hz)')
ax1.set_xlim([200,3500])
#ax1.set_ylim([0,75])
ax2.set_xlim([800,1400])
ax3.set_xlim([800,1400])
#ax2.set_ylim([3,11])
#ax3.set_ylim([3,11])
#ax1.set_xscale('log')
#ax2.set_xscale('log')
#ax3.set_xscale('log')
#ax1.set_xticks(freqs)
#ax2.set_xticks(freqs)
#ax3.set_xticks(freqs)
ax2.legend()
ax3.legend()

ax1.set_title('Spectrum of original signal')
ax2.set_title('Spectrum of bashed signal')
ax3.set_title('Spectrum of whacked signal')

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
