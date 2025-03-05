import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
import random
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def rayleigh_channel_gain(ex, sta):
    num_samples = 1
    gain = np.random.normal(ex, sta, num_samples)
    # Square the absolute value to get Rayleigh-distributed gains
    gain = np.abs(gain) ** 2
    return gain

# Function to implement water filling algorithm for power allocation
def water(s, total_power):
    a = total_power
    # Define the channel gain and noise level
    g_n = s
    N_0 = 1  # Assuming a fixed noise-level of 1 for all transmissions, this can be changed based on your requirement

    # Initialize the upper and lower bounds for the bisection search
    L = 0
    U = a + N_0 * np.sum(1 / (g_n + 1e-6))  # Initial guess for upper bound

    # Define the precision for the bisection search
    precision = 1e-6

    # Perform the bisection search for the power level
    while U - L > precision:
        alpha_bar = (L + U) / 2  # Set the current level to be in the middle of bounds
        p_n = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)  # Calculate the power allocation
        P = np.sum(p_n)  # Calculate the total power

        # Check whether the power budget is under or over-utilized
        if P > a:  # If the power budget is over-utilized
            U = alpha_bar  # Move the upper bound to the current power level
        else:  # If the power level is below the power budget
            L = alpha_bar  # Move the lower bound up

    # Calculate the final power allocation
    p_n_final = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)

    # Calculate the data rate for each channel
    SNR = g_n * p_n_final / N_0  # Calculate the SNR
    data_rate = np.log2(1 + SNR)  # Calculate the data rate
    sumdata_rate = np.sum(data_rate)
    # print('p_n_final', p_n_final)
    # print('data_rate', sumdata_rate)
    expert = p_n_final / total_power
    subexpert = p_n_final / total_power + np.random.normal(0, 0.1, len(p_n_final))
    return expert, sumdata_rate, subexpert

# Function to compute utility (reward) for the given state and action
def CompUtility(State, Aution):
        # Unpack action into integer and float components
    r=[]
    I= Aution[0] + Aution[1] * sum(np.random.randint(0, 100, 4))
    for i in range(4):
        r.append(Aution[i+4] * I)
    #sort r from highest to lowest
    r.sort(reverse=True)
    minr = min(r)
    if minr < 0:
        r = [i - minr for i in r]
    r = [i*100/sum(r) for i in r]
    p1,c1=getPower('pose',r,State[0])
    p2,c2=getPower('seg',r,State[1])
    p3,c3=getPower('canny',r,State[2])
    p4,c4=getPower('depth',r,State[3])
    if p1+p2+p3+p4 > 100:
        reward = -10
    if any([p1,p2,p3,p4]) < 5:
        reward = -10

    reward,quality=getQ([c1,c2,c3,c4])
    print(f"r: {r}, Power: {[p1,p2,p3,p4]},reward: {reward}, quality: {quality}")
    return reward,Aution, [p1,p2,p3,p4], [c1,c2,c3,c4],quality

def getQ(compres):
    q1= 0.002011 * compres[0]**2 - 0.08085 * compres[0] + 1.036
    q2 = -0.0007786 * compres[1]**2 - 0.0114 * compres[1] + 0.7079
    q3 = 0.003463 * compres[2]**2 - 0.1118 * compres[2] + 1.014
    q4 = -0.001717 * compres[3]**2 + 0.0002312 * compres[3] + 0.7038
    # q1 = -0.03860808 * compres[0] +0.88157507
    # q2 = -0.02775602 * compres[1] +0.76780874
    # q3 = -0.0390915 * compres[2] +0.74731419
    # q4 = -0.03582018 * compres[3] +0.83597836
    return 100*(q1+q2+q3+q4)/4,[q1,q2,q3,q4]
def getPower(type, r,channel_gain):
    
   
    noise_power = 1e-9  # W (converted from -90 dBm)
    bandwidth = 2.5e6  # Hz (5 MHz)

    # Power range from 1 mW to 100 mW (converted to Watts) in the step of 10
    power_levels_mW = np.arange(1, 100, 1)  # 1 mW to 100 mW
    power_levels_W = power_levels_mW * 1e-3  # Convert to Watts

    # Calculate SNR and data rate for each power level
    data_rates = []
    for P_i in power_levels_W:
        snr = (P_i * channel_gain) / noise_power
        data_rate = bandwidth * np.log2(1 + snr)
        data_rates.append(data_rate)

    # Create lookup table with power levels and int(330.30144e6/data_rate)
    lookup_table = dict(zip(power_levels_mW, [math.floor(330.30144e6 / rate) for rate in data_rates]))

    pi_history = 0
    argmax = 0
    def Gan(pi, task):
        compress = lookup_table[pi]
        if task == 'pose':
            cap = -0.03860808 * compress +0.88157507
        elif task == 'seg':
            cap = -0.02775602 * compress +0.76780874
        elif task == 'canny':
            cap = -0.0390915 * compress +0.74731419
        elif task == 'depth':
            cap = -0.03582018 * compress +0.83597836
        else:
            raise ValueError("Unknown task type.")
        
        return 1-cap, compress
    compresshistory = 0
    for pi in power_levels_mW:
        expectedpayment = 0
        gan, compressrate = Gan(pi, type)
        for h in range(1,4):
            # Normalize reward, adjust penalty
            tem = 100*r[h-1] * math.comb(3, h-1) * math.pow(1-gan, 4 - h) * math.pow(gan, h-1)- pi/(1-gan)
            expectedpayment += tem
            # print(f"r[h-1]{r[h-1]}, math.comb(3, h-1){math.comb(3, h-1)}, math.pow(1-gan, 4 - h){math.pow(1-gan, 4 - h)}, math.pow(gan, h-1){math.pow(gan, h-1)}")
        if compressrate != compresshistory:
            compresshistory = compressrate
            # print(f"Power level (mW): {pi}, Compression ratio: {compressrate}, Gan: {gan}, Expected payment: {expectedpayment}, - pi/gan: {pi / gan}")
            if expectedpayment > pi_history:
                pi_history = expectedpayment
                compresshistory = compressrate
                argmax = pi
    return max(argmax,5) , compresshistory