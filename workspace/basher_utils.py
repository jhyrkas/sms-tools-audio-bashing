import numpy as np

#bark frequencies used by functions defined on critical bandwidth
bark_cutoffs = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
                1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
                9500, 12000, 15500]

def get_bark_diff(f1,f2) :
    bark_f1 = -1
    for i in range(len(bark_cutoffs)-1) :
        if f1 > bark_cutoffs[i] :
            bark_f1 = i
    bark_f2 = -1
    for i in range(len(bark_cutoffs)-1) :
        if f2 > bark_cutoffs[i] :
            bark_f2 = i

    bandwidth1 = bark_cutoffs[bark_f1+1] - bark_cutoffs[bark_f1]
    bandwidth2 = bark_cutoffs[bark_f2+1] - bark_cutoffs[bark_f2]
    avg_bandwidth = (bandwidth1+bandwidth2)*0.5
    return abs(f1-f2)/avg_bandwidth

# equation from Sethares (various papers)
# could use updated equation from Vassilakis 2007
# notes:    this function models roughness as: r = e^(-3.5x) - e^(-5.75x), 
#           where x is roughly the % distances in frequencies by critical
#           bandwidth. the function peaks at around x=0.22 with value y=0.18.
#           r is multiplied by the linear gain of both frequencies, so in practice
#           it is much smaller than 0.18
def calculate_roughness_sethares(f1,v1,f2,v2) :
    a = -3.5
    b = -5.75
    d = 0.24
    s1 = 0.021
    s2 = 19
    s = d / (s1 * np.minimum(f1,f2) + s2)
    freq_diff = np.abs(f1-f2)
    return v1*v2*(np.exp(a*s*freq_diff) - np.exp(b*s*freq_diff))

# equation from Vassilakis 2007
# notes:    this equation scales sethares's equation above by relative amplitude and amplitude modulation.
#           this X portion exponentiates the amplitude component by 0.1 make it closer to closer resemble dB
#           or barks, as opposed to linear amplitude. the Y portion accounts for amplitudem modulation, a key
#           component of roughness. if both sinusoids are near the same volume, there will be very noticeable
#           amplitude modulation, but if one is much louder than the other there will be less modulation due
#           to masking.
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

def get_bandwidth_cutoffs(f1,f2, bw_percent_low, bw_percent_high) :
    bark_f1 = -1
    for i in range(len(bark_cutoffs)-1) :
        if f1 > bark_cutoffs[i] :
            bark_f1 = i
    bark_f2 = -1
    for i in range(len(bark_cutoffs)-1) :
        if f2 > bark_cutoffs[i] :
            bark_f2 = i

    bandwidth1 = bark_cutoffs[bark_f1+1] - bark_cutoffs[bark_f1]
    bandwidth2 = bark_cutoffs[bark_f2+1] - bark_cutoffs[bark_f2]
    avg_bandwidth = (bandwidth1+bandwidth2)*0.5

    return (bw_percent_low * avg_bandwidth, bw_percent_high * avg_bandwidth)

def criteria_critical_band_barks(f1,v1,f2,v2, bw_percent_low=0.1, bw_percent_high = 0.35) :
    # not dealing with ridiculously high or low frequencies
    if f1 < bark_cutoffs[0] or f1 > bark_cutoffs[-1] or f2 < bark_cutoffs[0] or f2 > bark_cutoffs[-1] :
        return False

    bw_low, bw_high = get_bandwidth_cutoffs(f1,f2, bw_percent_low, bw_percent_high)
    diff = abs(f1-f2)
    return diff < bw_high and diff > bw_low

# roughness always above threshold, only use critical bandwidth to filter
def calculate_roughness_pass(f1,v1,f2,v2) :
    return 1.0

# let all pairs through, only use roughness to filter
def criteria_func_pass(f1,v1,f2,v2, bw_percent_low=0.1, bw_percent_high = 0.35) :
    return True

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

def bash_freq(orig_freq, comparison_freq, bw_percent_low, bw_percent_high, hard_bash, delta=3, consonance = True) :
    # "hard bashing"
    if hard_bash :
        return comparison_freq - delta if orig_freq < comparison_freq else comparison_freq + delta

    # rudimentary brute force search
    bandwidths = get_bandwidth_cutoffs(orig_freq, comparison_freq, bw_percent_low, bw_percent_high)
    freq_low_bound = comparison_freq - bandwidths[1] if orig_freq < comparison_freq else comparison_freq + bandwidths[0]
    freq_high_bound = comparison_freq - bandwidths[0] if orig_freq < comparison_freq else comparison_freq + bandwidths[1]
    freq_candidates = np.linspace(freq_low_bound, freq_high_bound, 100)
    # we are not adjusting amplitude, so just pass 1.0 for both since it doesn't change the location of extrema
    roughnesses = calculate_roughness_sethares(freq_candidates, 1.0, comparison_freq, 1.0)
    index = np.argmin(roughnesses) if consonance else np.argmax(roughnesses)
    return freq_candidates[index]

# find the different in dB that is necessary for f_masker to fully mask f_masked
# estimation function from Lagrange 2001
def get_masking_level_dB(f_masker, f_masked) :
    bark_diff = get_bark_diff(f_masker, f_masked)
    slope = 15 if f_masker < f_masked else 27
    return 10 + bark_diff*slope

# transfer amplitude from the quieter sinusoid to the louder sinusoid in a way that is both power preserving and
# when perc_move = 1.0, the quieter sinusoid is theoretically completely masked by the louder
def whack_amp(f1, a1, f2, a2, perc_move) :
    loud_f, loud_a, quiet_f, quiet_a = f1,a1,f2,a2 if a1 > a2 else f2,a2,f1,a1
    orig_db_diff = 20*np.log10(loud_a) - 20*np.log10(quiet_a)
    masking_level_db = get_masking_level_dB(loud_f, quiet_f)
    delta = orig_db_diff + perc_move*(masking_level_db-orig_db_diff)
    # we need the new amplitudes to maintain power and also have a db diff of exactly delta when perc_move = 1.0
    # these equations are found by doing a lot of nasty algebra. the equations are:
    #           20*np.log10(new_a1) - 20*np.log10(new_a2) = delta
    #           a1**2 + a2**2 = new_a1**2 + new_a2**2
    # solve both for new_a1, then plug back through...
    new_quiet_a = np.sqrt((loud_a**2+quiet_a**2)/(1+10**(delta/10)))
    new_loud_a = np.sqrt(loud_**2 + quiet_**2 - new_quiet_a**2)
    return new_loud_a, new_quiet_a if a1 > a2 else new_quiet_a, new_loud_a
