import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal
import soundfile as sf
import sys

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
    def __init__(self, audiofile, track_id, hop_len_s) :
        self.audiofile = audiofile
        self.track_id = track_id
        self.freqs = []
        self.amps = []
        self.frame_count = 0
        self.start_time = 0
        self.end_time = 0
        self.hop_len_s = hop_len_s

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
        fs = 48000 # TODO: might need this to be variable in the future
        frame_s = self.hop_len_s
        f0s = np.zeros(int((self.end_time + frame_s) * fs))
        n = int(round((self.end_time - self.start_time)/frame_s))
        start_samp = int(self.start_time*fs)
        f0s[:start_samp] = self.freqs[0]
        frame_samp = int(frame_s * fs)
        for i in range(n-1) :
            start = start_samp + i*frame_samp
            f0s[start:start+frame_samp+1] = np.linspace(self.freqs[i],self.freqs[i+1],frame_samp+1)
        f0s[start_samp+(n-1)*frame_samp:] = self.freqs[-1]
        return f0s
 
    # TODO: more variables in the future? fs? extend length? fade in?
    def get_interpolated_amps(self) :
        fs = 48000 # TODO: might need this to be variable in the future
        frame_s = self.hop_len_s
        amps = np.zeros(int((self.end_time + frame_s) * fs))
        n = int(round((self.end_time - self.start_time)/frame_s))
        start_samp = int(self.start_time*fs)
        frame_samp = int(frame_s * fs)
        first_frame_len = min(start_samp, frame_samp) # in case this sinusoid starts in the first frame
        amps[start_samp-first_frame_len:start_samp+1] = np.linspace(0.0,self.amps[0],first_frame_len+1)
        for i in range(n-1) :
            start = start_samp + i*frame_samp
            amps[start:start+frame_samp+1] = np.linspace(self.amps[i],self.amps[i+1],frame_samp+1)
        amps[start_samp+(n-1)*frame_samp:start_samp+n*frame_samp] = np.linspace(self.amps[-1],0.0,frame_samp)
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
    def __init__(self, s, fs, n_sines) :
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
                track_id += 1
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
                    self.tracks[snippet.track_id] = Track(self, snippet.track_id, self.hop_len_s)
                self.tracks[snippet.track_id].add_frame(snippet)

    # TODO: consider how this is structured / what should be stored
    def calculate_roughness_overlap(self, other_audio) :
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
                    freq_diff = abs(this_partial.freq - that_partial.freq)
                    #r = calculate_roughness_setheras(this_partial.freq, this_partial.amp, that_partial.freq, that_partial.amp)
                    r = calculate_roughness_vassilakis(this_partial.freq, this_partial.amp, that_partial.freq, that_partial.amp)
                    partial_pair = (this_partial.track_id, that_partial.track_id)
                    if partial_pair not in overlap_dict :
                        overlap_dict[partial_pair] = ([r],i)
                    else :
                        overlap_dict[partial_pair][0].append(r)
        return overlap_dict

# equation from Sethares (various papers)
# could use updated equation from Vassilakis 2007
def calculate_roughness_setheras(f1,v1,f2,v2) :
    a = -3.5
    b = -5.75
    d = 0.24
    s1 = 0.021
    s2 = 19
    s = d / (s1 * min(f1,f2) + s2)
    freq_diff = abs(f1 - f2)
    return v1*v2*(np.exp(a*s*freq_diff) - np.exp(b*s*freq_diff))

# equation from Vassilakis 2007
def calculate_roughness_vassilakis(f1,v1,f2,v2) :
    X = (v1*v2)**0.1
    Y = 0.5 * ((2*min(v1,v2))/(v1+v2))**3.11
    b1 = -3.5
    b2 = -5.75
    s1 = 0.0207
    s2 = 18.96
    s = 0.24/(s1*min(f1,f2) + s2)
    fdiff = abs(f1-f2)
    Z = np.exp(b1*s*fdiff)-np.exp(b2*s*fdiff)
    return X*Y*Z

if len(sys.argv) < 3 :
    print('usage: python3 analyze_audio_sms.py <audiofile1> <audiofile2> [optional-nsines] [optional-freq-diff-min] [optional-freq-diff-max] [optional-new-delta]')
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

overlap_dict = analysis1.calculate_roughness_overlap(analysis2)
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

    roughnesses, start_frame = overlap_dict[key]
    roughness = np.mean(roughnesses)
    overlap_length = len(roughnesses)
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
