import numpy as np

# equation from Sethares (various papers)
# could use updated equation from Vassilakis 2007
def calculate_roughness_sethares(f1,v1,f2,v2) :
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

def criteria_func_pass(f1,v1,f2,v2) :
    return True

def criteria_critical_band_barks(f1,v1,f2,v2) :
    bark_cutoffs = [20, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480,
                1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700,
                9500, 12000, 15500]
    # not dealing with ridiculously high or low frequencies
    if f1 < bark_cutoffs[0] or f1 > bark_cutoffs[-1] or f2 < bark_cutoffs[0] or f2 > bark_cutoffs[-1] :
        return False
    
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
    bw_percent = 0.33 # TODO: parameterize? rationalize?
    return abs(f1-f2) < (bw_percent*avg_bandwidth)
