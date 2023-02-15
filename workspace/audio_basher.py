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

# TODO: should this go into a standalone file to keep this script clean?
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

# --- PARSING ARGUMENTS
parser = argparse.ArgumentParser(
    prog = 'audiobasher.py', 
    description = 'Whacks (attenuates) and bashes (frequency shifts) harmonics of sound files so they have lower auditory roughness when mixed together'
)

parser.add_argument('audio_files', action='store', type=str, nargs='+', help='the list of files to process')
parser.add_argument('-nsines', dest='n_sines', action='store', type=int, default=10, help='The number of sinusoids to fit to each audio file')
parser.add_argument('-delta', dest='delta', action='store', type=float, default=3., help='The frequency difference of bashed sinusoids and their neighor (please see the paper)')
parser.add_argument('-normalize', dest='normalize', action='store', type=bool, default=False, help='Normalizes audio files to have the same maximum peak sample (only use if audio levels are not already determined)')

args = parser.parse_args()

audio_files = args.audio_files
nfiles = len(audio_files)
n_sines = args.nsines
new_delta = args.delta
normalize = args.normalize

# --- ANALYZING FILES

sigs = [None] * nfiles
analyses = [None] * nfiles

intended_fs = 48000 # TODO: arg parse?
for i in range(nfiles) :
    s, fs = sf.read(audio_files[i])
    if len(s.shape) > 1 :
        s = (s[:,0] + s[:,1]) * 0.5
    if fs != intended_fs :
        s = resampy.core.resample(s,fs,intended_fs)
    if normalize :
        s = s / np.max(np.abs(s))
    sigs[i] = s
    analyses[i] = AnalyzedAudio(i,s,intended_fs,n_sines)

# --- FINDING AREAS OF ROUGHNESS

# TODO: arg parse
#r_func = calculate_roughness_vassilakis
#threshold_r = 1.0e-2 # roughness

r_func = calculate_roughness_sethares
threshold_r = 1.0e-4 # roughness

c_func = criteria_critical_band_barks
#c_func = criteria_func_pass

threshold_f = 10 # time in frames (100 ms) TODO: arg parse?

filter_candidates = []
for i in range(nfiles-1) :
    for j in range(i+1, nfiles) :
        overlap_dict = analyses[i].calculate_roughness_overlap(analyses[j], roughness_function=r_func, criteria_function=c_func)
        merge_overlaps(filter_candidates, overlap_dict, analyses[i], analyses[j], threshold_r, threshold_f)

filter_candidates = sorted(filter_candidates, key=lambda x: x[0], reverse=True)

filters=[[] for i in range(nfiles)]
sin_params=[[] for i in range(nfiles)]
sin_deltas=[[] for i in range(nfiles)]
sin_filts=[[] for i in range(nfiles)]

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
    print('AUDIOFILE {audio_id}: adding frequency {f}'.format(audio_id=audio_id,f=new_f))
    print()

out_vanilla = np.zeros(np.max([sigs[i].shape[0] for i in range(nfiles)]))
for i in range(nfiles) :
    out_vanilla[:sigs[i].shape[0]] += sigs[i] * (1/nfiles) # avoiding clipping
sf.write('vanilla.wav', out_vanilla, intended_fs)

filt_sigs = [None] * nfiles
for i in range(nfiles) :
    filts = np.array(filters[i])
    s_filt = scipy.signal.sosfiltfilt(filts,sigs[i]) if len(filts.shape) > 1 else sigs[i]
    filt_sigs[i] = s_filt

out_filt = np.zeros(np.max([filt_sigs[i].shape[0] for i in range(nfiles)]))
for i in range(nfiles) :
    out_filt[:filt_sigs[i].shape[0]] += filt_sigs[i] * (1/nfiles) # avoiding clipping
sf.write('filtered.wav', out_filt, intended_fs)

out_bashed = out_filt.copy()
for i in range(nfiles) :
    for j in range(len(sin_filts[i])) :
        filt_sig = scipy.signal.sosfiltfilt(sin_filts[i][j], sigs[i])
        delta = sin_deltas[i][j]
        delta_abs = np.abs(delta)
        mod_sig = np.cos(2*np.pi*np.arange(s_filt.shape[0])*delta_abs/intended_fs)
        s_hil = scipy.signal.hilbert(s_filt).imag
        m_hil = scipy.signal.hilbert(mod_sig).imag
        sign = 1 if delta < 0 else -1
        shifted_sig = filt_sig*mod_sig + (sign*s_hil*m_hil)
        out_bashed[:shifted_sig.shape[0]] += shifted_sig

sf.write('bashed.wav', out_bashed, intended_fs)