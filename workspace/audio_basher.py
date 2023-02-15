import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import resampy
import scipy.signal
import soundfile as sf
import sys

from analysis_classes import *
from roughness_and_criteria_functions import *

def merge_overlaps(list_so_far, overlap_dict, analysis1, analysis2, threshold_r, threshold_f) :
    for key in overlap_dict.keys() :
        t_id1, t_id2 = key
        track1 = analysis1.tracks[t_id1]
        track2 = analysis2.tracks[t_id2]
        roughnesses, start_frame, end_frame = overlap_dict[key]
        roughness = np.mean(roughnesses)
        overlap_length = end_frame - start_frame
        if roughness > threshold_r and overlap_length >= threshold_f: # TODO: better
            list_so_far.append((roughness, track1, track2))

parser = argparse.ArgumentParser(
    prog = 'audiobasher.py', 
    description = 'Whacks (attenuates) and bashes (frequency shifts) harmonics of sound files so they have lower auditory roughness when mixed together'
)

parser.add_argument('audio_files', action='store', type=str, nargs='+', help='the list of files to process')
parser.add_argument('-nsines', dest='n_sines', action='store', type=int, default=10, help='The number of sinusoids to fit to each audio file')
parser.add_argument('-delta', dest='delta', action='store', type=float, default=3., help='The frequency difference of bashed sinusoids and their neighor (please see the paper)')
parser.add_argument('-normalize', dest='normalize', action='store', type=bool, default=False, help='Normalizes audio files to have the same maximum peak sample (only use if audio levels are not already determined)')


if len(sys.argv) < 3 :
    print('usage: python3 process_three_notes.py <audiofile1> <audiofile2> <audiofile3> [optional-nsines] [optional-new-delta] [optional-normalize]')
    sys.exit(1)

audiofile1 = sys.argv[1]
audiofile2 = sys.argv[2]
audiofile3 = sys.argv[3]
n_sines = 10 if len(sys.argv) < 5 else int(sys.argv[4])
# TODO: consider what delta should be...
new_delta = 3.0 if len(sys.argv) < 6 else float(sys.argv[5])
normalize = False if len(sys.argv) < 7 else bool(sys.argv[6])

intended_fs = 48000
s1,fs1 = sf.read(audiofile1)
s2,fs2 = sf.read(audiofile2)
s3,fs3 = sf.read(audiofile3)
if len(s1.shape) > 1 :
    s1 = (s1[:,0] + s1[:,1]) * 0.5
if len(s2.shape) > 1 :
    s2 = (s2[:,0] + s2[:,1]) * 0.5
if len(s3.shape) > 1 :
    s3 = (s3[:,0] + s3[:,1]) * 0.5
if fs1 != intended_fs :
    s1 = resampy.core.resample(s1,fs1,intended_fs)
if fs2 != intended_fs :
    s2 = resampy.core.resample(s2,fs2,intended_fs)
if fs3 != intended_fs :
    s3 = resampy.core.resample(s3,fs3,intended_fs)

if normalize :
    s1 = s1 / np.max(np.abs(s1))
    s2 = s2 / np.max(np.abs(s2))
    s3 = s3 / np.max(np.abs(s3))
analysis1 = AnalyzedAudio(0,s1,intended_fs,n_sines)
analysis2 = AnalyzedAudio(1,s2,intended_fs,n_sines)
analysis3 = AnalyzedAudio(2,s3,intended_fs,n_sines)

#r_func = calculate_roughness_vassilakis
#threshold_r = 1.0e-2 # roughness

r_func = calculate_roughness_sethares
threshold_r = 1.0e-4 # roughness

c_func = criteria_critical_band_barks
#c_func = criteria_func_pass

threshold_f = 10 # time in frames (100 ms) TODO: change

overlap_dict12 = analysis1.calculate_roughness_overlap(analysis2, roughness_function=r_func, criteria_function=c_func)
overlap_dict13 = analysis1.calculate_roughness_overlap(analysis3, roughness_function=r_func, criteria_function=c_func)
overlap_dict23 = analysis2.calculate_roughness_overlap(analysis3, roughness_function=r_func, criteria_function=c_func)

filter_candidates = []
merge_overlaps(filter_candidates, overlap_dict12, analysis1, analysis2, threshold_r, threshold_f)
merge_overlaps(filter_candidates, overlap_dict13, analysis1, analysis3, threshold_r, threshold_f)
merge_overlaps(filter_candidates, overlap_dict23, analysis2, analysis3, threshold_r, threshold_f)
filter_candidates = sorted(filter_candidates, key=lambda x: x[0], reverse=True)

filters=[[],[],[]]
sin_params=[[],[],[]]
sin_deltas=[[],[],[]]
sin_filts=[[],[],[]]

for roughness,track1,track2 in filter_candidates :
    filter_track1 = track1.get_avg_amp() < track2.get_avg_amp()
    filtered_track = None
    unfiltered_track = None
    if filter_track1 and not track1.filtered :
        filtered_track = track1
        unfiltered_track = track1
        track1.filtered = True
    elif not filter_track1 and not track2.filtered :
        filtered_track = track2
        unfiltered_track = track1
        track2.filtered = True
    else :
        # skip this pair
        continue
    w0 = filtered_track.get_avg_freq()
    bw = max(filtered_track.get_max_freq() - filtered_track.get_avg_freq(), filtered_track.get_avg_freq() - filtered_track.get_min_freq())
    Q = min(w0/bw,100) # TODO: do this better
    fs = intended_fs
    b,a = scipy.signal.iirnotch(w0,Q,fs)
    audio_id = filtered_track.audiofile.file_id
    filters[audio_id].append(np.hstack((b,a)))
            
    new_f = unfiltered_track.get_avg_freq() - new_delta if unfiltered_track.get_avg_freq() > filtered_track.get_avg_freq() else filtered_track.get_avg_freq() + new_delta
    b,a = scipy.signal.iirpeak(w0,Q,fs)
    sin_params[audio_id].append(filtered_track)
    sin_filts[audio_id].append(np.hstack((b,a)))
    sin_deltas[audio_id].append(new_f-w0)

    print('AUDIOFILE {audio_id}: filtering frequency {f} with Q {Q}, registered roughness {r: .5f}'.format(audio_id=audio_id,f=w0,Q=Q,r=roughness))
    print(b)
    print(a)
    print('AUDIOFILE {audio_id}: adding frequency {f}'.format(audio_id=audio_id,f=new_f))
    print()

filts1 = np.array(filters[0])
filts2 = np.array(filters[1])
filts3 = np.array(filters[2])

# trying to avoid clipping
s1 *= 0.33
s2 *= 0.33
s3 *= 0.33

out_vanilla = np.zeros(max(s1.shape[0],max(s2.shape[0],s3.shape[0])))
out_vanilla[:s1.shape[0]] += s1
out_vanilla[:s2.shape[0]] += s2
out_vanilla[:s3.shape[0]] += s3
sf.write('vanilla.wav', out_vanilla, intended_fs)

s1_filt = scipy.signal.sosfiltfilt(filts1,s1) if len(filts1.shape) > 1 else s1
s2_filt = scipy.signal.sosfiltfilt(filts2,s2) if len(filts2.shape) > 1 else s2
s3_filt = scipy.signal.sosfiltfilt(filts3,s3) if len(filts3.shape) > 1 else s3

out_filt = np.zeros(max(s1_filt.shape[0],max(s2_filt.shape[0],s3_filt.shape[0])))
out_filt[:s1_filt.shape[0]] += s1_filt
out_filt[:s2_filt.shape[0]] += s2_filt
out_filt[:s3_filt.shape[0]] += s3_filt
sf.write('filtered.wav', out_filt, intended_fs)

out_bashed = out_filt.copy()
for j in range(3) :
    for i in range(len(sin_params[j])) :
        f0s = sin_params[j][i].get_interpolated_f0s() + sin_deltas[j][i]
        cumphase = np.cumsum(f0s / intended_fs)
        amps = sin_params[j][i].get_interpolated_amps()
        new_sin = np.sin(2*np.pi*cumphase) * amps
        track_start = int(intended_fs*sin_params[j][i].adj_start_time)
        limit = min(out_bashed.shape[0], new_sin.shape[0]+track_start)
        out_bashed[track_start:limit] += new_sin[:limit-track_start]

sf.write('bashed.wav', out_bashed, intended_fs)
