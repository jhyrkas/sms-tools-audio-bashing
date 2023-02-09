import matplotlib.pyplot as plt
import numpy as np
import os
import resampy
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

intended_fs = 48000
s1,fs1 = sf.read(audiofile1)
s2,fs2 = sf.read(audiofile2)
if len(s1.shape) > 1 :
    s1 = (s1[:,0] + s1[:,1]) * 0.5
if len(s2.shape) > 1 :
    s2 = (s2[:,0] + s2[:,1]) * 0.5
if fs1 != intended_fs :
    s1 = resampy.core.resample(s1,fs1,intended_fs)
if fs2 != intended_fs :
    s2 = resampy.core.resample(s2,fs2,intended_fs)
analysis1 = AnalyzedAudio(1,s1,intended_fs,n_sines)
analysis2 = AnalyzedAudio(2,s2,intended_fs,n_sines)

overlap_dict = analysis1.calculate_roughness_overlap(analysis2, roughness_function=calculate_roughness_vassilakis, criteria_function=criteria_critical_band_barks)
#overlap_dict = analysis1.calculate_roughness_overlap(analysis2, roughness_function=calculate_roughness_vassilakis)
#overlap_dict = analysis1.calculate_roughness_overlap(analysis2)
threshold_r = 1.0e-2 # roughness
threshold_r_std = 0.005
threshold_f = 10 # time in frames (100 ms) TODO: change

track1_filters = []
track2_filters = []
track1_sin_params = []
track2_sin_params = []
track1_sin_deltas = []
track2_sin_deltas = []
track1_sin_filts = []
track2_sin_filts = []

# trying a different structure
filter_candidates = []
for key in overlap_dict.keys() :
    t_id1, t_id2 = key
    track1 = analysis1.tracks[t_id1]
    track2 = analysis2.tracks[t_id2]
    roughnesses, start_frame, end_frame = overlap_dict[key]
    roughness = np.mean(roughnesses)
    roughness_std = np.std(roughnesses)
    overlap_length = end_frame - start_frame
    if roughness > threshold_r and overlap_length >= threshold_f: # TODO: better
    #if roughness > threshold_r and roughness_std < threshold_r_std and overlap_length >= threshold_f: # TODO: better
        filter_candidates.append((roughness, t_id1, t_id2, track1, track2))

filter_candidates.sort(reverse = True)

for roughness,id1,id2,track1,track2 in filter_candidates :
    filter_track1 = track1.get_avg_amp() < track2.get_avg_amp()
    if filter_track1 and not track1.filtered :
        w0 = track1.get_avg_freq()
        bw = max(track1.get_max_freq() - track1.get_avg_freq(), track1.get_avg_freq() - track1.get_min_freq())
        Q = min(w0/bw,100) # TODO: do this better
        fs = 48000 # TODO: better
        b,a = scipy.signal.iirnotch(w0,Q,fs)
        track1_filters.append(np.hstack((b,a)))
        print('TRACK 1: filtering frequency {f} with Q {Q}, registered roughness {r: .5f}'.format(f=w0,Q=Q,r=roughness))
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
        track1.filtered = True
    elif not filter_track1 and not track2.filtered :
        w0 = track2.get_avg_freq()
        bw = max(track2.get_max_freq() - track2.get_avg_freq(), track2.get_avg_freq() - track2.get_min_freq())
        Q = min(w0/bw,100) # TODO: do this better
        fs = 48000 # TODO: better
        b,a = scipy.signal.iirnotch(w0,Q,fs)
        track2_filters.append(np.hstack((b,a)))
        print('TRACK 2: filtering frequency {f} with Q {Q}, registered roughness {r: .5f}'.format(f=w0,Q=Q,r=roughness))
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
        track2.filtered = True

filts1 = np.array(track1_filters)
filts2 = np.array(track2_filters)

# trying to avoid clipping
s1 *= 0.5
s2 *= 0.5

out_vanilla = np.zeros(max(s1.shape[0],s2.shape[0]))
out_vanilla[:s1.shape[0]] += s1
out_vanilla[:s2.shape[0]] += s2
sf.write('vanilla.wav', out_vanilla, intended_fs)

s1_filt = scipy.signal.sosfiltfilt(filts1,s1) if len(filts1.shape) > 1 else s1
s2_filt = scipy.signal.sosfiltfilt(filts2,s2) if len(filts2.shape) > 1 else s2

# MIGHT BE A BAD IDEA
#s1_filt = (s1_filt / np.max(np.abs(s1_filt))) * np.max(np.abs(s1))
#s2_filt = (s2_filt / np.max(np.abs(s2_filt))) * np.max(np.abs(s2))

out_filt = np.zeros(max(s1_filt.shape[0],s2_filt.shape[0]))
out_filt[:s1_filt.shape[0]] += s1_filt
out_filt[:s2_filt.shape[0]] += s2_filt
sf.write('filtered.wav', out_filt, intended_fs)

out_bashed = out_filt.copy()
for i in range(len(track1_sin_params)) :
    f0s = track1_sin_params[i].get_interpolated_f0s() + track1_sin_deltas[i]
    cumphase = np.cumsum(f0s / intended_fs)
    amps = track1_sin_params[i].get_interpolated_amps()
    new_sin = np.sin(2*np.pi*cumphase) * amps
    track_start = int(fs*track1_sin_params[i].adj_start_time)
    limit = min(out_bashed.shape[0], new_sin.shape[0]+track_start)
    out_bashed[track_start:limit] += new_sin[:limit-track_start]
    #sf.write('track1_sin{i}.wav'.format(i=i), new_sin, intended_fs)
    #plt.plot(f0s)
    #plt.show()

for i in range(len(track2_sin_params)) :
    f0s = track2_sin_params[i].get_interpolated_f0s() + track2_sin_deltas[i]
    cumphase = np.cumsum(f0s / intended_fs)
    amps = track2_sin_params[i].get_interpolated_amps()
    new_sin = np.sin(2*np.pi*cumphase) * amps
    track_start = int(fs*track2_sin_params[i].adj_start_time)
    limit = min(out_bashed.shape[0], new_sin.shape[0]+track_start)
    out_bashed[track_start:limit] += new_sin[:limit-track_start]
    #sf.write('track2_sin{i}.wav'.format(i=i), new_sin, intended_fs)
    #plt.plot(f0s)
    #plt.show()

sf.write('bashed.wav', out_bashed, intended_fs)
