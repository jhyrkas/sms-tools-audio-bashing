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
    prog = 'audiobasher.py', 
    description = 'Whacks (attenuates) and bashes (frequency shifts) harmonics of sound files so they have lower auditory roughness when mixed together'
)

parser.add_argument('audio_files', action='store', type=str, nargs='+', help='the list of files to process')
parser.add_argument('-nsines', dest='n_sines', action='store', type=int, default=10, help='The number of sinusoids to fit to each audio file')
parser.add_argument('-bw_percent_low', dest='bw_percent_low', action='store', type=float, default=0.1, help='Defines the low threshold of frequencies that can be adjusted. Specify as fraction of critical bandwidth (default 0.1, minimum 0.0, maximum < bw_percent_high')
parser.add_argument('-bw_percent_high', dest='bw_percent_high', action='store', type=float, default=0.33, help='Defines the high threshold of frequencies that can be adjusted. Specify as fraction of critical bandwidth (default 0.33, minimum > bw_percent_low, maximum 1.0')
parser.add_argument('--hard_bash', dest='hard_bash', action=argparse.BooleanOptionalAction, default=False, help='When hard bash is turned on (i.e. -hard_bash=True), frequency pairs that fall within the bandwidth range will have the quieter partial bashed directly to delta Hz above or below the louder partial, regardless of bandwidth implication. Otherwise, soft bash behavior is used (see "consonance" parameter).')
parser.add_argument('--consonance', dest='consonance', action=argparse.BooleanOptionalAction, default=True, help='When soft bashing behavior is used (see "hard_bash" parameter), frequencie pairs that fall within the bandwidth with have the quieter partial moved within the bandwidth defined. If consonance is on, the most consonant frequency will be chosen; if consonance is off (False), the most dissonant freuqncy will be chosen. This algorithm is greedy and not necessarily optimal if moving the quieter frequency causes it to be moved within the range of another partial not previously analyzed.')
parser.add_argument('-delta', dest='delta', action='store', type=float, default=3., help='The frequency difference of bashed sinusoids and their neighbor when hard bashing is enabled.')
parser.add_argument('--normalize', dest='normalize', action=argparse.BooleanOptionalAction, default=False, help='Normalizes audio files to have the same maximum peak sample (only use if audio levels are not already determined)')
parser.add_argument('-roughness_thresh', dest='roughness_thresh', action='store', type=float, default=0.0001, help='Defines the threshold of calculated roughness that a partial pair must exceed to be adjusted. Range is dependent on roughness function. Set to 0.0 to turn off roughness thresholding')


args = parser.parse_args()
print('ARGS PARSED')
print(args)
print()

audio_files = args.audio_files
nfiles = len(audio_files)
n_sines = args.n_sines
new_delta = args.delta
normalize = args.normalize
bw_percent_low = args.bw_percent_low
bw_percent_high = args.bw_percent_high
hard_bash = args.hard_bash
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
deltas=[[] for i in range(nfiles)]
notch_filts=[[] for i in range(nfiles)]
peak_filts=[[] for i in range(nfiles)]
times=[[] for i in range(nfiles)]

print('calculating filters')

for roughness,track1,track2 in filter_candidates :
    filter_track1 = track1.get_avg_amp() < track2.get_avg_amp()
    filtered_track = None
    unfiltered_track = None
    if filter_track1 and not track1.filtered :
        filtered_track = track1
        unfiltered_track = track2
        track1.filtered = True
    elif not filter_track1 and not track2.filtered :
        filtered_track = track2
        unfiltered_track = track1
        track2.filtered = True
    else :
        # skip this pair
        continue
    print('CLASHING FREQUENCIES: {f1_avg:.2f}\t{f2_avg:.2f}'.format(
        f1_avg=track1.get_avg_freq(), f2_avg=track2.get_avg_freq()
        ))
    #print('CLASHING FREQUENCIES: {f1_min:.2f},{f1_avg:.2f},{f1_max:.2f}\t{f2_min:.2f},{f2_avg:.2f},{f2_max:.2f}'.format(
    #    f1_min=track1.get_min_freq(), f1_avg=track1.get_avg_freq(), f1_max=track1.get_max_freq(),
    #    f2_min=track2.get_min_freq(), f2_avg=track2.get_avg_freq(), f2_max=track2.get_max_freq()
    #    ))
    w0 = filtered_track.get_avg_freq()
    bw = max(filtered_track.get_max_freq() - filtered_track.get_avg_freq(), filtered_track.get_avg_freq() - filtered_track.get_min_freq())
    Q = min(w0/bw,100) # TODO: do this better
    fs = intended_fs
    b,a = scipy.signal.iirnotch(w0,Q,fs)
    audio_id = filtered_track.audiofile.file_id
    notch_filts[audio_id].append((b,a))
    
    new_f = bash_freq(filtered_track.get_avg_freq(), unfiltered_track.get_avg_freq(), bw_percent_low, bw_percent_high, hard_bash, new_delta, consonance)

    b,a = scipy.signal.iirpeak(w0,Q,fs)
    tracks[audio_id].append(filtered_track)
    peak_filts[audio_id].append((b,a))
    deltas[audio_id].append(new_f-w0)
    
    t1 = track1.get_adjusted_track_times()
    t2 = track2.get_adjusted_track_times()
    # we probably don't want to crossfade from a shifted sine back to a non-shifted sine ?
    #times[audio_id].append((min(t1[0],t2[0]),max(t1[1],t2[1])))
    times[audio_id].append((max(t1[0],t2[0]),min(t1[1],t2[1])))

    print('AUDIOFILE {audio_id}: filtering frequency {f} with Q {Q}, registered roughness {r: .5f}'.format(audio_id=audio_id,f=w0,Q=Q,r=roughness))
    print('AUDIOFILE {audio_id}: adding frequency {f}'.format(audio_id=audio_id,f=new_f))
    print()

out_vanilla = np.zeros(np.max([sigs[i].shape[0] for i in range(nfiles)]))
for i in range(nfiles) :
    out_vanilla[:sigs[i].shape[0]] += sigs[i]
sf.write('vanilla.wav', out_vanilla, intended_fs)

out_filt = np.zeros(out_vanilla.shape)
out_bashed = np.zeros(out_vanilla.shape)

print('filtering')

tmp_index = 0
window_s = 0.25 # TODO: think about this
for i in range(nfiles) :
    sig = sigs[i]
    for j in range(len(notch_filts[i])) :
        # create the filtered and bashed signals

        # continue filtering out frequencies across loop
        s_filt = scipy.signal.filtfilt(notch_filts[i][j][0], notch_filts[i][j][1], sig)
        # but get peaking signal from the original audio
        s_peak = scipy.signal.filtfilt(peak_filts[i][j][0], peak_filts[i][j][1], sigs[i])
        delta = deltas[i][j]
        delta_abs = np.abs(delta)
        mod_sig = np.cos(2*np.pi*np.arange(s_peak.shape[0])*delta_abs/intended_fs)
        s_hil = scipy.signal.hilbert(s_peak).imag
        m_hil = scipy.signal.hilbert(mod_sig).imag
        sign = 1 if delta < 0 else -1
        shifted_sig = s_peak*mod_sig + (sign*s_hil*m_hil)
        shifted_sig = (shifted_sig / np.max(np.abs(shifted_sig))) * np.max(np.abs(s_peak))

        # make the cross fade masks
        start_t_s, end_t_s = times[i][j]
        start_samp = int(start_t_s*intended_fs)
        end_samp = int(end_t_s*intended_fs)
        edit_start = max(int((start_t_s-window_s)*intended_fs),0)
        edit_end = min(int((end_t_s+window_s)*intended_fs),s_filt.shape[0]-1)
        
        original_mask = np.zeros(sigs[i].shape[0]) + 1.
        processed_mask = np.zeros(sigs[i].shape[0])
        # constant power
        #original_mask[edit_start:start_samp] = np.cos(np.linspace(0,np.pi/2,start_samp-edit_start))
        #processed_mask[edit_start:start_samp] = np.sin(np.linspace(0,np.pi/2,start_samp-edit_start))
        #original_mask[start_samp:end_samp] = 0.
        #processed_mask[start_samp:end_samp] = 1.
        #original_mask[end_samp:edit_end] = np.sin(np.linspace(0,np.pi/2,edit_end-end_samp))
        #processed_mask[end_samp:edit_end] = np.cos(np.linspace(0,np.pi/2,edit_end-end_samp))

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
        out_bashed[:s_filt.shape[0]] += (processed_mask * shifted_sig) # add in the cross-faded bashed sinusoid
        #sf.write('tmp{i}_pre.wav'.format(i=tmp_index), s_peak/np.max(np.abs(s_peak)), intended_fs)
        #sf.write('tmp{i}_post.wav'.format(i=tmp_index), shifted_sig/np.max(np.abs(shifted_sig)), intended_fs)
        #sf.write('tmp{i}_filt.wav'.format(i=tmp_index), (original_mask * sigs[i]) + (processed_mask * s_filt), intended_fs)
        #sf.write('tmp{i}_bash.wav'.format(i=tmp_index), (original_mask * sigs[i]) + (processed_mask * s_filt)+ (processed_mask * shifted_sig), intended_fs)
        tmp_index += 1
    out_filt[:sigs[i].shape[0]] += sig
    out_bashed[:sigs[i].shape[0]] += sig

sf.write('filtered.wav', out_filt, intended_fs)
sf.write('bashed.wav', out_bashed, intended_fs)
