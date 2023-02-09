import numpy as np
import os
import scipy.signal
import sys

from roughness_and_criteria_functions import *

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
try :
    import sineModel as sm
except :
    print("couldn't import sms-tools: sineModel")
    sys.exit(1)

class TrackSnippet :
    def __init__(self, frame, track_id, freq, amp) :
        self.frame = frame
        self.track_id = track_id
        self.freq = freq
        self.amp = amp

    def register_track(track) :
        pass

class Track :
    def __init__(self, audiofile, fs, track_id, hop_len_s) :
        self.audiofile = audiofile
        self.fs = fs
        self.track_id = track_id
        self.freqs = []
        self.amps = []
        self.frame_count = 0
        self.start_time = 0
        self.end_time = 0
        self.hop_len_s = hop_len_s
        self.filtered = False

    def add_frame(self, snippet) :
        # maybe we can just keep them all?
        self.freqs.append(snippet.freq)
        self.amps.append(snippet.amp)
        if self.frame_count == 0 :
            self.min_freq = snippet.freq
            self.max_freq = snippet.freq
            self.freq_sum = snippet.freq
            self.min_amp = snippet.amp
            self.max_amp = snippet.amp
            self.amp_sum = snippet.amp
            self.frame_count = 1
            self.start_time = snippet.frame.start_time
            self.end_time = snippet.frame.end_time
            self.adj_start_time = self.start_time + 2*self.hop_len_s
            self.adj_end_time = self.end_time + 2*self.hop_len_s
        else :
            self.min_freq = min(self.min_freq, snippet.freq)
            self.max_freq = max(self.max_freq, snippet.freq)
            self.freq_sum += snippet.freq
            self.min_amp = min(self.min_amp, snippet.amp)
            self.max_amp = max(self.max_amp, snippet.amp)
            self.amp_sum += snippet.amp
            self.frame_count += 1
            self.end_time = snippet.frame.end_time

    def get_avg_freq(self) :
        return np.mean(np.array(self.freqs))
    
    def get_std_freq(self) :
        return np.std(np.array(self.freqs))

    def get_min_freq(self) :
        return np.min(self.freqs)

    def get_max_freq(self) :
        return np.max(self.freqs)

    def get_avg_amp(self) :
        return np.mean(np.array(self.amps))

    def get_min_amp(self) :
        return np.min(self.amps)

    def get_max_amp(self) :
        return np.max(self.amps)

    def get_track_times(self) :
        return self.start_time, self.end_time

    # TODO: more variables in the future? fs? extend length? 
    def get_interpolated_f0s(self) :
        track_len = self.end_time - self.start_time
        frame_s = self.hop_len_s
        f0s = np.zeros(int((track_len + frame_s) * self.fs))
        n = int(round((track_len)/frame_s))
        start_samp = 0 # ?
        frame_samp = int(frame_s * self.fs)
        for i in range(n-1) :
            start = start_samp + i*frame_samp
            f0s[start:start+frame_samp+1] = np.linspace(self.freqs[i],self.freqs[i+1],frame_samp+1)
        f0s[start_samp+(n-1)*frame_samp:] = self.freqs[-1]
        return f0s
 
    # TODO: more variables in the future? fs? extend length? fade in?
    def get_interpolated_amps(self) :
        track_len = self.end_time - self.start_time
        frame_s = self.hop_len_s
        amps = np.zeros(int((track_len + frame_s) * self.fs))
        n = int(round((track_len)/frame_s))
        start_samp = 0 # ?
        frame_samp = int(frame_s * self.fs)
        for i in range(n-1) :
            start = start_samp + i*frame_samp
            amps[start:start+frame_samp+1] = np.linspace(self.amps[i],self.amps[i+1],frame_samp+1)
        amps[start_samp+(n-1)*frame_samp:] = self.amps[-1]
        return amps

class Frame :
    def __init__(self, audiofile, start_time, num_partials, params, frame_len_s) :
        self.audiofile = audiofile
        self.start_time = start_time
        self.end_time = start_time + frame_len_s
        self.npar = num_partials
        self.track_snippets = []
        for i,f,a in params :
            self.track_snippets.append(TrackSnippet(self, i, f, a))

class AnalyzedAudio :
    def __init__(self, file_id, s, fs, n_sines) :
        self.file_id = file_id
        self.fs = fs
        self.n_sines = n_sines
        self.frames = []

        # analysis parameters...to be variable?
        N = 2048
        H = N//4
        w = scipy.signal.hamming(N)
        t = -50.0 # dB

        self.frame_len_s = N / fs
        self.hop_len_s = H / fs
        start_time = -(self.hop_len_s*2)
        min_dur = 1.5 * self.hop_len_s
        
        freqs, mags, phases = sm.sineModelAnal(s, fs, w, N, H, t, n_sines, min_dur)
        self.frame_count = freqs.shape[0]
        amps = np.power(10, mags/20)
        
        tracks = np.zeros(freqs.shape, dtype=int)
        trackid = 1
        in_track = False
        for j in range(n_sines) :
            # in case the last sinusoid ended on a frame with a track
            if in_track :
                in_track = False
                trackid += 1
            for i in range(self.frame_count) :
                if freqs[i,j] > 0 :
                    in_track = True
                    tracks[i,j] = trackid
                elif in_track :
                    in_track = False
                    trackid += 1

        for i in range(self.frame_count) :
            time = start_time
            params = []
            indeces = np.where(freqs[i,:] > 0)[0]
            num_partials = len(indeces)
            params = [(tracks[i,j], freqs[i,j], amps[i,j]) for j in indeces]
            self.frames.append(Frame(self, time, num_partials, params, self.hop_len_s))
            start_time += self.hop_len_s

        # create tracks
        self.tracks = {}
        for frame in self.frames :
            for snippet in frame.track_snippets :
                if snippet.track_id not in self.tracks :
                    self.tracks[snippet.track_id] = Track(self, self.fs, snippet.track_id, self.hop_len_s)
                self.tracks[snippet.track_id].add_frame(snippet)

    # TODO: consider how this is structured / what should be stored
    def calculate_roughness_overlap(self, other_audio, criteria_function = criteria_func_pass, roughness_function = calculate_roughness_sethares) :
        frame_min = min(self.frame_count, other_audio.frame_count)
        overlap_dict = {}
        for i in range(frame_min) :
            this_frame = self.frames[i]
            that_frame = other_audio.frames[i]
            t = this_frame.start_time
            partials_this = this_frame.track_snippets
            partials_that = that_frame.track_snippets
            for j in range(this_frame.npar) :
                for k in range(that_frame.npar) :
                    this_partial = partials_this[j]
                    that_partial = partials_that[k]
                    if criteria_function(this_partial.freq, this_partial.amp, that_partial.freq, that_partial.amp) :
                        r = roughness_function(this_partial.freq, this_partial.amp, that_partial.freq, that_partial.amp)
                        partial_pair = (this_partial.track_id, that_partial.track_id)
                        if partial_pair not in overlap_dict :
                            overlap_dict[partial_pair] = ([r],i,i+1)
                        else :
                            overlap_dict[partial_pair][0].append(r)
                            overlap_dict[partial_pair] = (overlap_dict[partial_pair][0], overlap_dict[partial_pair][1],i+1)
        return overlap_dict
