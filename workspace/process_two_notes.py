import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal
import soundfile as sf
import sys

from analysis_classes import *
from roughness_and_criteria_functions import *

if len(sys.argv) < 3 :
    print('usage: python3 process_two_notes.py <audiofile1> <audiofile2> [optional-nsines] [optional-freq-diff-min] [optional-freq-diff-max] [optional-new-delta]')
    sys.exit(1)

audiofile1 = sys.argv[1]
audiofile2 = sys.argv[2]
n_sines = 10 if len(sys.argv) < 4 else int(sys.argv[3])
# TODO: consider what delta should be...
new_delta = 3.0 if len(sys.argv) < 5 else float(sys.argv[4])

s1,fs1 = sf.read(audiofile1)
s2,fs2 = sf.read(audiofile2)
analysis1 = AnalyzedAudio(s1,fs1,n_sines)
analysis2 = AnalyzedAudio(s2,fs2,n_sines)

overlap_dict = analysis1.calculate_roughness_overlap(analysis2, roughness_function=calculate_roughness_vassilakis, criteria_function=criteria_critical_band_barks)
threshold_r = 1.0e-2 # roughness
threshold_f = 10 # time in frames (100 ms) TODO: change

track1_filters = []
track2_filters = []
track1_sin_params = []
track2_sin_params = []
track1_sin_deltas = []
track2_sin_deltas = []
track1_sin_filts = []
track2_sin_filts = []
for key in overlap_dict.keys() :
    t_id1, t_id2 = key
    track1 = analysis1.tracks[t_id1]
    track2 = analysis2.tracks[t_id2]

    roughnesses, start_frame, end_frame = overlap_dict[key]
    roughness = np.mean(roughnesses)
    overlap_length = end_frame - start_frame
    if roughness > threshold_r and overlap_length >= threshold_f: # TODO: better
        filter_track1 = track1.get_avg_amp() < track2.get_avg_amp()
        if filter_track1 :
            w0 = track1.get_avg_freq()
            bw = max(track1.get_max_freq() - track1.get_avg_freq(), track1.get_avg_freq() - track1.get_min_freq())
            Q = min(w0/bw,100) # TODO: do this better
            fs = 48000 # TODO: better
            b,a = scipy.signal.iirnotch(w0,Q,fs)
            track1_filters.append(np.hstack((b,a)))
            print('TRACK 1: filtering frequency {f} with Q {Q}, registered roughness {r: .5f} for {i} frames'.format(f=w0,Q=Q,r=roughness,i=overlap_length))
            print(b)
            print(a)
            
            new_f = track2.get_avg_freq() - new_delta if track2.get_avg_freq() > track1.get_avg_freq() else track2.get_avg_freq() + new_delta
            b,a = scipy.signal.iirpeak(w0,Q,fs)
            track1_sin_params.append(track1)
            track1_sin_filts.append(np.hstack((b,a)))
            track1_sin_deltas.append(new_f-w0)
            print('TRACK 1: adding frequency {f}'.format(f=new_f))
            print(b)
            print(a)
            print()
        else :
            w0 = track2.get_avg_freq()
            bw = max(track2.get_max_freq() - track2.get_avg_freq(), track2.get_avg_freq() - track2.get_min_freq())
            Q = min(w0/bw,100) # TODO: do this better
            fs = 48000 # TODO: better
            b,a = scipy.signal.iirnotch(w0,Q,fs)
            track2_filters.append(np.hstack((b,a)))
            print('TRACK 2: filtering frequency {f} with Q {Q}, registered roughness {r: .5f} for {i} frames'.format(f=w0,Q=Q,r=roughness,i=overlap_length))
            print(b)
            print(a)

            new_f = track1.get_avg_freq() - new_delta if track1.get_avg_freq() > track2.get_avg_freq() else track1.get_avg_freq() + new_delta
            b,a = scipy.signal.iirpeak(w0,Q,fs)
            track2_sin_params.append(track2)
            track2_sin_filts.append(np.hstack((b,a)))
            track2_sin_deltas.append(new_f-w0)
            print('TRACK 2: adding frequency {f}'.format(f=new_f))
            print(b)
            print(a)
            print()

filts1 = np.array(track1_filters)
filts2 = np.array(track2_filters)

s1,fs1 = sf.read(audiofile1)
s2,fs2 = sf.read(audiofile2)
s1 *= 0.5
s2 *= 0.5

out_vanilla = np.zeros(max(s1.shape[0],s2.shape[0]))
out_vanilla[:s1.shape[0]] += s1
out_vanilla[:s2.shape[0]] += s2
sf.write('vanilla.wav', out_vanilla, fs1)

s1_filt = scipy.signal.sosfiltfilt(filts1,s1) if len(filts1.shape) > 1 else s1
s2_filt = scipy.signal.sosfiltfilt(filts2,s2) if len(filts2.shape) > 1 else s2

# MIGHT BE A BAD IDEA
#s1_filt = (s1_filt / np.max(np.abs(s1_filt))) * np.max(np.abs(s1))
#s2_filt = (s2_filt / np.max(np.abs(s2_filt))) * np.max(np.abs(s2))

out_filt = np.zeros(max(s1_filt.shape[0],s2_filt.shape[0]))
out_filt[:s1_filt.shape[0]] += s1_filt
out_filt[:s2_filt.shape[0]] += s2_filt
sf.write('filtered.wav', out_filt, fs2)

out_bashed = out_filt.copy()
for i in range(len(track1_sin_params)) :
    f0s = track1_sin_params[i].get_interpolated_f0s() + track1_sin_deltas[i]
    cumphase = np.cumsum(f0s / fs1)
    amps = track1_sin_params[i].get_interpolated_amps()
    new_sin = np.sin(2*np.pi*cumphase) * amps
    limit = min(out_bashed.shape[0], new_sin.shape[0])
    out_bashed[:limit] += new_sin[:limit]
    #sf.write('track1_sin{i}.wav'.format(i=i), new_sin, fs1)
    #plt.plot(f0s)
    #plt.show()

for i in range(len(track2_sin_params)) :
    f0s = track2_sin_params[i].get_interpolated_f0s() + track2_sin_deltas[i]
    cumphase = np.cumsum(f0s / fs1)
    amps = track2_sin_params[i].get_interpolated_amps()
    new_sin = np.sin(2*np.pi*cumphase) * amps
    limit = min(out_bashed.shape[0], new_sin.shape[0])
    out_bashed[:limit] += new_sin[:limit]
    #sf.write('track2_sin{i}.wav'.format(i=i), new_sin, fs1)
    #plt.plot(f0s)
    #plt.show()

sf.write('bashed.wav', out_bashed, fs1)
