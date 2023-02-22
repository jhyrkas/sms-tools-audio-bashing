import numpy as np

from roughness_and_criteria_functions import calculate_roughness_sethares
from roughness_and_criteria_functions import get_bandwidth_cutoffs
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
