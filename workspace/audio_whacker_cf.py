import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import resampy
import scipy.signal
import soundfile as sf
import sys

from analysis_classes import *
from basher_utils import *

# --- PARSING ARGUMENTS
parser = argparse.ArgumentParser(
    prog = 'audio_whacker.py', 
    description = 'Transfers energy between harmonics of sound files so they have lower auditory roughness when mixed together'
)

parser.add_argument('audio_files', action='store', type=str, nargs='+', help='the list of files to process')
parser.add_argument('-nsines', dest='n_sines', action='store', type=int, default=10, help='The number of sinusoids to fit to each audio file')
parser.add_argument('-bw_percent_low', dest='bw_percent_low', action='store', type=float, default=0.1, help='Defines the low threshold of frequencies that can be adjusted. Specify as fraction of critical bandwidth (default 0.1, minimum 0.0, maximum < bw_percent_high')
parser.add_argument('-bw_percent_high', dest='bw_percent_high', action='store', type=float, default=0.33, help='Defines the high threshold of frequencies that can be adjusted. Specify as fraction of critical bandwidth (default 0.33, minimum > bw_percent_low, maximum 1.0')
parser.add_argument('--consonance', dest='consonance', action=argparse.BooleanOptionalAction, default=True, help='When soft bashing behavior is used (see "hard_bash" parameter), frequencie pairs that fall within the bandwidth with have the quieter partial moved within the bandwidth defined. If consonance is on, the most consonant frequency will be chosen; if consonance is off (False), the most dissonant freuqncy will be chosen. This algorithm is greedy and not necessarily optimal if moving the quieter frequency causes it to be moved within the range of another partial not previously analyzed.')
parser.add_argument('-whack_percent', dest='whack_percent', action='store', type=float, default=3., help='Percent (as 0.0-1.0) whacking to frequencies that qualify. When 0.0, the output is unchanged. When 1.0, energy is transfered such that some sinusoids are completely masked (in theory).')
parser.add_argument('--normalize', dest='normalize', action=argparse.BooleanOptionalAction, default=False, help='Normalizes audio files to have the same maximum peak sample (only use if audio levels are not already determined)')
parser.add_argument('-roughness_thresh', dest='roughness_thresh', action='store', type=float, default=0.0001, help='Defines the threshold of calculated roughness that a partial pair must exceed to be adjusted. Range is dependent on roughness function. Set to 0.0 to turn off roughness thresholding')


args = parser.parse_args()
print('ARGS PARSED')
print(args)
print()

audio_files = args.audio_files
nfiles = len(audio_files)
n_sines = args.n_sines
new_whack_percent = args.whack_percent
normalize = args.normalize
bw_percent_low = args.bw_percent_low
bw_percent_high = args.bw_percent_high
consonance = args.consonance
threshold_r = args.roughness_thresh

# --- ANALYZING FILES

print('analysis')

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
    sigs[i] = sigs[i] * (1/nfiles) # trying to avoid clipping...do it after analysis though

# getting the avg amps of all tracks because they might change multiple times during whacking and we will need
# easy access to changes we have already made
amp_dicts = [{}] * nfiles
for i in range(nfiles) :
    for k in analyses[i].keys() :
        amp_dicts[i][k] = analyses[i].tracks[k].get_avg_freq()

# --- FINDING AREAS OF ROUGHNESS

# TODO: arg parse
#r_func = calculate_roughness_vassilakis
#threshold_r = 1.0e-2 # roughness

r_func = calculate_roughness_sethares

#r_func = calculate_roughness_pass
#threshold_r = 1.0e-4

c_func = criteria_critical_band_barks
#c_func = criteria_func_pass
c_func_dict = {'bw_percent_low': bw_percent_low, 'bw_percent_high': bw_percent_high}

threshold_f = 10 # time in frames (100 ms) TODO: arg parse?

print('overlap')

filter_candidates = []
for i in range(nfiles-1) :
    for j in range(i+1, nfiles) :
        #overlap_dict = analyses[i].calculate_roughness_overlap_frames(analyses[j], criteria_function=c_func, roughness_function=r_func)
        overlap_dict = analyses[i].calculate_roughness_overlap_tracks(analyses[j], criteria_function=c_func, roughness_function=r_func, c_func_kargs=c_func_dict)
        merge_overlaps(filter_candidates, overlap_dict, analyses[i], analyses[j], threshold_r, threshold_f)

filter_candidates = sorted(filter_candidates, key=lambda x: x[0], reverse=True)

tracks=[[] for i in range(nfiles)]
notch_filts=[[] for i in range(nfiles)]
peak_amps=[[] for i in range(nfiles)]
peak_filts=[[] for i in range(nfiles)]
times=[[] for i in range(nfiles)]

print('calculating filters')

for roughness,track1,track2 in filter_candidates :
    filter_track1 = track1.get_avg_amp() < track2.get_avg_amp()
    if filter_track1 and not track1.filtered :
        track1.filtered = True
    elif not filter_track1 and not track2.filtered :
        track2.filtered = True
    else :
        # skip this pair
        continue
    print('CLASHING FREQUENCIES: {f1_avg:.2f},{a1_avg: .4f}\t{f2_avg:.2f},{a2_avg:.4f}'.format(
        f1_avg=track1.get_avg_freq(), a1_avg = track1.get_avg_amp(), f2_avg=track2.get_avg_freq(), a2_avg=track2.get_avg_amp()
        ))
    new_a1, new_a2 = whack_amp(
                track1.get_avg_freq(),
                amp_dicts[track1.audiofile.file_id][track1.track_id],
                track2.get_avg_freq(),
                amp_dicts[track2.audiofile.file_id][track2.track_id],
                perc_move
            )
    w0_t1 = track1.get_avg_freq()
    bw_t1 = max(track1.get_max_freq() - track1.get_avg_freq(), track1.get_avg_freq() - track1.get_min_freq())
    Q_t1 = min(w0_t1/bw_t1,100)
    audio_id1 = track1.audiofile.file_id
    # append the filter coefs for track 1
    b,a = scipy.signal.iirnotch(w0_t1,Q_t1,intended_fs)
    notch_filts[audio_id1].append((b,a))
    b,a = scipy.signal.iirpeak(w0_t1,Q_t1,intended_fs)
    peak_filts[audio_id1].append((b,a))
    peak_amps[audio_id1].append(new_a1)
    amp_dicts[audio_id1][track1.track_id] = new_a1
 
    w0_t2 = track2.get_avg_freq()
    bw_t2 = max(track2.get_max_freq() - track2.get_avg_freq(), track2.get_avg_freq() - track2.get_min_freq())
    Q_t2 = min(w0_t2/bw_t2,100)
    audio_id2 = track2.audiofile.file_id
    # append the filter coefs for track 1
    b,a = scipy.signal.iirnotch(w0_t2,Q_t2,intended_fs)
    notch_filts[audio_id2].append((b,a))
    b,a = scipy.signal.iirpeak(w0_t2,Q_t2,intended_fs)
    peak_filts[audio_id2].append((b,a))
    peak_amps[audio_id2].append(new_a2)
    amp_dicts[audio_id2][track2.track_id] = new_a2

    t1 = track1.get_adjusted_track_times()
    t2 = track2.get_adjusted_track_times()
    times[audio_id1].append((max(t1[0],t2[0]),min(t1[1],t2[1])))
    times[audio_id2].append((max(t1[0],t2[0]),min(t1[1],t2[1])))

out_vanilla = np.zeros(np.max([sigs[i].shape[0] for i in range(nfiles)]))
for i in range(nfiles) :
    out_vanilla[:sigs[i].shape[0]] += sigs[i]
sf.write('vanilla.wav', out_vanilla, intended_fs)

filtered = np.zeros(out_vanilla.shape)
out_bashed = np.zeros(out_vanilla.shape)

print('filtering')

tmp_index = 0
out_bashed = out_filt.copy()
window_s = 0.25 # TODO: think about this
for i in range(nfiles) :
    sig = sigs[i]
    for j in range(len(notch_filts[i])) :
        # create the filtered and bashed signals

        # continue filtering out frequencies across loop
        s_filt = scipy.signal.filtfilt(notch_filts[i][j][0], notch_filts[i][j][1], sig)
        # but get peaking signal from the original audio
        s_peak = scipy.signal.filtfilt(peak_filts[i][j][0], peak_filts[i][j][1], sigs[i])
        new_amp = peak_amps[i][j]
        old_avg = np.mean(np.abs(s_peak))
        s_peak *= (new_amp / old_avg) # whacked amplitude

        # make the cross fade masks
        start_t_s, end_t_s = times[i][j]
        start_samp = int(start_t_s*intended_fs)
        end_samp = int(end_t_s*intended_fs)
        edit_start = max(int((start_t_s-window_s)*intended_fs),0)
        edit_end = min(int((end_t_s+window_s)*intended_fs),s_filt.shape[0]-1)
        
        original_mask = np.zeros(sigs[i].shape[0]) + 1.
        processed_mask = np.zeros(sigs[i].shape[0])

        # constant linear gain
        original_mask[edit_start:start_samp] = np.linspace(1,0,start_samp-edit_start)
        processed_mask[edit_start:start_samp] = np.linspace(0,1,start_samp-edit_start)
        original_mask[start_samp:end_samp] = 0.
        processed_mask[start_samp:end_samp] = 1.
        if edit_end > end_samp : # occassionally this won't happen if the track goes all the way to the end of the file
            original_mask[end_samp:edit_end] = np.linspace(0,1,edit_end-end_samp)
            processed_mask[end_samp:edit_end] = np.linspace(1,0,edit_end-end_samp)

        # housekeeping for maintaining the output signals
        sig = (original_mask * sig) + (processed_mask * s_filt)
        out_whacked[:s_filt.shape[0]] += (processed_mask * shifted_sig) # add in the cross-faded bashed sinusoid
        #sf.write('tmp{i}_pre.wav'.format(i=tmp_index), s_peak/np.max(np.abs(s_peak)), intended_fs)
        #sf.write('tmp{i}_post.wav'.format(i=tmp_index), shifted_sig/np.max(np.abs(shifted_sig)), intended_fs)
        #sf.write('tmp{i}_filt.wav'.format(i=tmp_index), (original_mask * sigs[i]) + (processed_mask * s_filt), intended_fs)
        #sf.write('tmp{i}_bash.wav'.format(i=tmp_index), (original_mask * sigs[i]) + (processed_mask * s_filt)+ (processed_mask * shifted_sig), intended_fs)
        tmp_index += 1
    out_filt[:sigs[i].shape[0]] += sig
    out_whacked[:sigs[i].shape[0]] += sig

sf.write('filtered.wav', out_filt, intended_fs) # this isn't the same thing anymore
sf.write('whacked.wav', out_whacked, intended_fs)
