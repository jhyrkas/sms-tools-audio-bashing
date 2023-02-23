import numpy as np

# bark frequencies used by functions defined on critical bandwidth
bark_cutoffs = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
                1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
                9500, 12000, 15500]

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

# roughness always above threshold, only use criteria function to filter
def calculate_roughness_pass(f1,v1,f2,v2) :
    return 1.0

# let all pairs through, only use roughness to filter
def criteria_func_pass(f1,v1,f2,v2, bw_percent_low=0.1, bw_percent_high = 0.35) :
    return True
