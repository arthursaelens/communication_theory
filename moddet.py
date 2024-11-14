import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import erfc

#################################################
############## Mapping en Detectie ##############
#################################################

def mapper(bit_array,M,type):
    """
        Functie die de bits in bit_array omzet naar complexe symbolen van een bepaalde constellatie
        Input:
            bit_array = 1D numpy array met bits (0 of 1) die gemapped moeten worden
            M = de grootte van de constellatie
            type = str met type van de constellatie (in dit project 'PAM', 'PSK' of 'QAM'
        Output:
            symbols = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die overeenkomen met de bits in bit_array
    """
    symbols=np.array([])
    
    if type == 'PAM':
        bit_pairs = bit_array.reshape(-1,2)
        gray_mapping = {
            (0,0): -3,
            (0,1): -1,
            (1,1): 1,
            (1,0): 3
        }
        symbols = np.array([gray_mapping[tuple(b)] for b in bit_pairs])
        symbols = symbols / np.sqrt(5) #normalisatie!!
    
    elif type == 'PSK':
        symbols = np.where(bit_array == 0, -1, 1)
    elif type == 'QAM':
        bit_pairs = bit_array.reshape(-1, 2)
        gray_mapping = {
            (0, 0): -1 - 1j,
            (0, 1): -1 + 1j,
            (1, 1): 1 + 1j,
            (1, 0): 1 - 1j
        }
        symbols = np.array([gray_mapping[tuple(b)] for b in bit_pairs])
        symbols = symbols / np.sqrt(2)

    return symbols

def demapper(symbols,M,type):
    """
        Functie die de (complexe) symbolen in symbols omzet naar de bijhorende bits
        Input:
            symbols = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die gedemapped moeten worden
            M = de grootte van de constellatie
            type = str met type van de constellatie (in dit project 'PAM', 'PSK' of 'QAM'
        Output:
            bit_array_demapped = 1D numpy array met bits (0 of 1) die overeenkomen met de symbolen in symbols
    """
    bit_array_demapped = np.array([])

    if type == 'PAM':
        pam_mapping = {
            -3 / np.sqrt(5): [0, 0],
            -1 / np.sqrt(5): [0, 1],
            1 / np.sqrt(5): [1, 1],
            3 / np.sqrt(5): [1, 0]
        }
        for sym in symbols:
            if sym < -2 / np.sqrt(5):
                bit_array_demapped = np.hstack([bit_array_demapped, pam_mapping[-3 / np.sqrt(5)]])
            elif sym < 0: 
                bit_array_demapped = np.hstack([bit_array_demapped, pam_mapping[-1 / np.sqrt(5)]])
            elif sym < 2 / np.sqrt(5):
                bit_array_demapped = np.hstack([bit_array_demapped, pam_mapping[1 / np.sqrt(5)]])
            else:
                bit_array_demapped = np.hstack([bit_array_demapped, pam_mapping[3 / np.sqrt(5)]])

    elif type == 'PSK':
        bit_array_demapped = np.where(symbols.real < 0, 0, 1)

    elif type == 'QAM':

        qam_mapping = {
            (-1 / np.sqrt(2), -1 / np.sqrt(2)): [0, 0],  
            (-1 / np.sqrt(2), 1 / np.sqrt(2)): [0, 1],   
            (1 / np.sqrt(2), 1 / np.sqrt(2)): [1, 1],
            (1 / np.sqrt(2), -1 / np.sqrt(2)): [1, 0]
        }

        for sym in symbols:
            real_sign = 1 if sym.real >= 0 else -1 
            imag_sign = 1 if sym.imag >= 0 else -1
            bit_array_demapped = np.hstack([bit_array_demapped, qam_mapping[(real_sign / np.sqrt(2), imag_sign / np.sqrt(2))]])

    return bit_array_demapped

def discreet_geheugenloos_kanaal(a,sigma,A0,theta):
    """
        Functie die het discreet geheugenloos kanaal simuleert
        Input:
            a = 1D numpy array die de sequentie van datasymbolen ak bevat
            sigma = standaard deviatie van de witte ruis
            A0 = de schalingsfactor van het kanaal
            theta = de faserotatie (in radialen)
        Output:
            z = 1D numpy array die de samples van het ontvangen signaal zk bevat (aan de ontvanger)
    """
    z=np.array([])

    # Genereer witte Gaussiaanse ruis (complex)
    r_ruis = np.random.normal(0,sigma,len(a))
    i_ruis = np.random.normal(0,sigma,len(a))
    ruis = r_ruis + i_ruis *1j
    
    hch = A0 * np.exp(1j * theta)
    
    z = hch * a + ruis
    
    return z

def maak_decisie_variabele(z,A0_hat,theta_hat):
    """
        Functie die het gedecimeerde signaal z schaalt met hch_hat en de fase compenseert met rotatie over theta_hat
        Input:
            z = 1D numpy array die het gedecimeerde signaal bevat
            A0_hat = de geschatte schalingsfactor van het kanaal
            theta_hat = de geschatte faserotatie die de demodulator introduceerde (in radialen)
        Output:
            u = 1D numpy array die de decisievariabelen bevat
    """
    u=np.array([])

    u = z * np.exp(-1j * theta_hat)/ A0_hat 
    return u

def decisie(u,M,type):
    """
        Functie die de decisievariabelen afbeeldt op de meest waarschijnlijke bijhorende symbolen
        Input:
            u = 1D numpy array met de decisievariabelen
            M = de grootte van de constellatie
            type = str met type van de constellatie (in dit project 'PAM', 'PSK' of 'QAM'
        Output:
            symbols = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die het meest waarschijnlijk horen bij de decisievariabelen
    """
    symbols = np.array([])

    if type == 'PAM':
        symbols = np.where(u.real < -2 / np.sqrt(5), -3 / np.sqrt(5), np.where(u.real < 0, -1 / np.sqrt(5),np.where(u.real < 2 / np.sqrt(5), 1 / np.sqrt(5), 3 / np.sqrt(5))))
    
    elif type == 'PSK':
        symbols = np.where(u.real < 0, -1, 1)

    elif type == 'QAM':
        I = np.where(u.real < 0, -1 / np.sqrt(2), 1 / np.sqrt(2))
        Q = np.where(u.imag < 0, -1 / np.sqrt(2), 1 / np.sqrt(2))
        symbols = I + 1j * Q

    return symbols

#################################################
############## Basisbandmodulatie ###############
#################################################


def moduleerBB(a,T,Ns,alpha,Lf):
    """
        Functie die de symboolsequentie a omzet in een basisbandsignaal
        Input:
            a = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die gemoduleerd moeten worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            alpha = roll-off factor van de square-root raised cosine puls
            Lf = aantal symboolintervallen van puls voordat we deze afknotten (aan een zijde); de pulse bestaat dus uit 2Lf symboolintervallen
        Output:
            sBB = 1D numpy array die de samples van het gemoduleerde signaal sBB(t) bevat
    """
    K = len(a)
    num_samples = K * Ns + 2 * Lf * Ns  # Total number of samples (accounting for pulse truncation)
    sBB = np.zeros(num_samples, dtype=complex)

    print(num_samples)
    print(K)
    for n in range(num_samples):
        t = (n - Lf * Ns) * (T / Ns)
        for k in range(K):
            sBB[n] += a[k] * pulse(t - k*T, T, alpha)

    return sBB

def basisband_kanaal(sBB,sigma,A0,theta):
    """
        Functie die het continue-tijd basisbandkanaal simuleert
        Input:
            sBB = 1D numpy array die de samples van het gemoduleerde signaal sBB(t) bevat
            sigma = standaard deviatie van de witte ruis
            A0 = de schalingsfactor van het kanaal
            theta = de faserotatie (in radialen)
        Output:
            rBB = 1D numpy array die de samples van het ontvangen signaal rBB(t) bevat (aan de ontvanger)
    """
    rBB=np.array([])

    length = len(sBB)

    h_ch = A0 * np.exp(1j * theta)

    # Genereer de witte ruis: complexe circulair symmetrische Gaussiaanse ruis
    # De reële en imaginaire delen hebben beide een standaarddeviatie van sigma/sqrt(2)
    nBB = np.random.normal(0, sigma / np.sqrt(2), len(sBB)) + 1j * np.random.normal(0, sigma / np.sqrt(2), len(sBB))

    rBB = h_ch * sBB + nBB

    return rBB

def demoduleerBB(rBB,T,Ns,alpha,Lf):
    """
        Functie die het ontvangen signaal rBB(t) demoduleert
        Input:
            rBB = 1D numpy array die de samples van het ontvangen signaal rBB(t) bevat dat gedemoduleerd moet worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            alpha = roll-off factor van de square-root raised cosine puls
            Lf = aantal symboolintervallen van puls voordat we deze afknotten (aan 1 zijde); de pulse bestaat dus uit 2Lf symboolintervallen
        Output:
            y = 1D numpy array die de samples van het gedemoduleerde signaal rBB(t) bevat
    """
    # Tijdstip array voor de puls
    t = np.linspace(-Lf*T, Lf*T, num = 2*Lf*Ns + 1)
    
    # Genereer de zenderpuls (ontvangerfilter)
    hrec = pulse(t, T, alpha)
    
    # Convolutie
    y = T * np.convolve(rBB, hrec)

    # schaal door Ns om het effect van oversampling te corrigeren
    # convolutie zorgt impliciet voor factor Ns
    y /= Ns
    
    # Afkappen om de juiste lengte te verkrijgen
    start_index = 2 * Lf * Ns #factor 2 belangrijk!
    y = y[start_index:-start_index]

    return y

def decimatie(y,Ns,Lf):
    """
        Functie die de decimatie uitvoert op het gedemoduleerd signaal y(t)
        Input:
            y = 1D numpy array die de samples van het gedemoduleerde signaal y(t) bevat
            Ns = aantal samples per symboolinterval
            Lf = aantal symboolintervallen van puls voordat we deze afknotten (aan 1 zijde); de pulse bestaat dus uit 2Lf symboolintervallen
        Output:
            z = 1D numpy array die de samples na decimatie bevat
    """
    
    tau_opt = 0  
    #num_symbols = len(y) // Ns  # This should match the number of original symbols

    # Decimatie: één sample per symboolperiode (bij k*T + tau_opt), de afkapping is al gebeurd bij demodulatie
    z = y[tau_opt::Ns] # Selecteer elk Ns-de sample, startend bij tau_opt
    
    return z

#################################################
############## Draaggolfmodulatie ###############
#################################################

def moduleer(sBB,T,Ns,frequentie):
    """
        Functie die de (complexe) symbolen in symbols moduleert
        Input:
            a = 1D numpy array bestaande uit (complexe) symbolen van de constellatie die gemoduleerd moeten worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            frequentie = draaggolfrequentie
        Output:
            s = 1D numpy array die de samples van het gemoduleerde signaal s(t) bevat
    """
    s=np.array([])

    # voeg hier je code toe

    return s

def kanaal(s,sigma,A0):
    """
        Functie die het kanaal simuleert
        Input:
            s = 1D numpy array die de samples van het gemoduleerde signaal s(t) bevat
            sigma = standaard deviatie van de witte ruis
            A0 = schalingsfactor van het kanaal
        Output:
            r = 1D numpy array die de samples van het ontvangen signaal r(t) bevat (aan de ontvanger)
    """
    r=np.array([])

    # voeg hier je code toe

    return r

def demoduleer(r,T,Ns,frequentie,theta):
    """
        Functie die het ontvangen signaal r(t) demoduleert
        Input:
            r = 1D numpy array die de samples van het ontvangen signaal r(t) bevat dat gedemoduleerd moet worden
            T = symboolinterval
            Ns = aantal samples per symboolinterval
            frequentie = draaggolfrequentie
            theta = onbekende faserotatie die de demodulator introduceert (in radialen)
        Output:
            rBB = 1D numpy array die de samples van het gedemoduleerde signaal r(t) bevat
    """
    rBB=np.array([])

    # voeg hier je code toe

    return rBB

def pulse(t,T,alpha):
    """ Niet aanpassen!
        Functie die de square-root raised cosine puls genereert
        Input:
            t = 1D numpy array met tijdstippen waarop de puls gesampled moet worden
            T = symboolinterval
            alpha = roll-off factor
        Output:
            p = 1D numpy array met de samples van de puls op de tijdstippen in t
    """
    een = (1-alpha)*np.sinc(t*(1-alpha)/T)
    twee = (alpha)*np.cos(math.pi*(t/T-0.25))*np.sinc(alpha*t/T-0.25)
    drie = (alpha)*np.cos(math.pi*(t/T+0.25))*np.sinc(alpha*t/T+0.25)
    p = 1/np.sqrt(T)*(een+twee+drie)
    return p

# plot fouriergetransformeerde pulse voor verschillende alpha
def plot_fourier_transform(alpha_values, T, t):
    plt.figure(figsize=(12, 8))
    
    for alpha in alpha_values:
        pulse_samples = pulse(t, T, alpha)
        freqs = np.fft.fftfreq(len(t), d=t[1]-t[0])
        fourier_transform = np.fft.fft(pulse_samples)
        
        plt.plot(freqs, np.abs(fourier_transform), label=f'α={alpha}')
    
    plt.title('Fourier Transform of Square-Root Raised Cosine Pulses')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.legend()
    plt.xlim(-5, 5)  # Adjust as necessary for your specific use case
    plt.show()

# Sample parameters
T = 1.0  # Example symbol interval
alpha_values = [0.05, 0.5, 0.95]  # Different roll-off factors
t = np.linspace(-4, 4, 1000)  # Time instances

#plot_fourier_transform(alpha_values, T, t)




##########################
##tests #################

def plot_scatter(symbols, title="Scatter Plot"):
    """
    Plot de ontvangen symbolen in een scatterplot.
    """
    plt.figure()
    plt.scatter(symbols.real, symbols.imag, color='blue', marker='o')
    plt.title(title)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.grid(True)
    plt.show()


###TEST VRAAG 4###
def vraag4(mod_type = 'QAM'):
    bit_array = np.random.randint(0, 2, 10000)  # Random bits
    M = 4 
    A0 = 1 
    theta = 0 
    sigma = 0

    # Map de bits naar symbolen
    symbols = mapper(bit_array, M, mod_type)

    Eb_N0_dB_range = np.arange(0, 40, 10)

    received_symbols = discreet_geheugenloos_kanaal(symbols, sigma, A0, theta)

    na_decisie = maak_decisie_variabele(received_symbols, A0, theta)

    plot_scatter(na_decisie, title="4-QAM Scatter Plot zonder ruis")

    for Eb_N0_dB in Eb_N0_dB_range:

        Eb_N0_linear = 10 ** (Eb_N0_dB / 10)

        sigma = math.sqrt(1/(Eb_N0_linear*math.log2(M)*2))

        received_symbols_with_noise = discreet_geheugenloos_kanaal(symbols, sigma, A0, theta)

        na_decisie_with_noise = maak_decisie_variabele(received_symbols_with_noise, A0, theta)

        plot_scatter(na_decisie_with_noise, title=f"4-QAM Scatter Plot bij Eb/N0 = {Eb_N0_dB} dB")

###TEST VRAAG 5###
def vraag5(mod_type='QAM'):
    bit_array = np.random.randint(0, 2, 1000)  # Random bits
    M = 4
    A0 = 1  
    theta = 0 
    sigma = 0 

    symbols = mapper(bit_array, M, mod_type)

    received_symbols = discreet_geheugenloos_kanaal(symbols, sigma, A0, theta)

    na_decisie = maak_decisie_variabele(received_symbols, A0, theta)

    estimated_symbols = decisie(na_decisie, M, mod_type)

    correct_estimation = np.allclose(symbols, estimated_symbols)
    print(f"Correcte constellatiepunten teruggevonden voor N0 = 0: {correct_estimation}")

    # Scatterplot maken om visueel te inspecteren
    plot_scatter(na_decisie, title="4-QAM Scatter Plot zonder ruis (N0 = 0)")




# Q-function
def q_function(x):
    return 0.5 * erfc(x / np.sqrt(2))

def theoretical_ber(M, mod_type, Eb_N0_range):
    
    if mod_type == 'PSK':
        return q_function(np.sqrt(2 * Eb_N0_range))
    
    elif mod_type == 'QAM':
        return  q_function(np.sqrt(2*Eb_N0_range))
    
    elif mod_type == 'PAM':
        return (3/4) * q_function(np.sqrt(2*Eb_N0_range*0.4))
    
    return None

def ber_simulation(M, mod_type, Eb_N0_range,epsilon=0,phi=0):
    bit_errors = []
    num_bits = 100000  # Number of bits per simulation

    A = 1
    A_ = A*epsilon + A

    theta = np.pi/6
    theta_ = phi +  theta


    
    for Eb_N0 in Eb_N0_range:
        total_bit_errors = 0
        total_bits = 0

        # sigma
        sigma = math.sqrt(1/(Eb_N0*math.log2(M)*2))
        print(sigma)
        
        while total_bit_errors < 100:
            bit_array = np.random.randint(0, 2, num_bits)
            
            symbols = mapper(bit_array, M, mod_type)
            received_symbols = discreet_geheugenloos_kanaal(symbols, sigma, A, theta)
            decision_variables = maak_decisie_variabele(received_symbols, A_, theta_)
            estimated_symbols = decisie(decision_variables, M, mod_type)
            demapped_bits = demapper(estimated_symbols, M, mod_type)
            
            bit_errors_current = np.sum(bit_array != demapped_bits)
            total_bit_errors += bit_errors_current
            total_bits += len(bit_array)
        
        ber = total_bit_errors / total_bits
        bit_errors.append(ber)
    
    return bit_errors

def plot_ber(Eb_N0_dB_range, ber_data):
    plt.figure()
    
    for modulation, (simulated_ber, theoretical_ber) in ber_data.items():
        plt.semilogy(Eb_N0_dB_range, simulated_ber, 'o-', label=f'{modulation.upper()} Simulated')
        
        plt.semilogy(Eb_N0_dB_range, theoretical_ber, '--', label=f'{modulation.upper()} Theoretical')
    
    plt.xlabel('$E_b/N_0$ (dB)')
    plt.ylabel('BER')
    plt.title('BER vs. $E_b/N_0$ (Simulated vs. Theoretical)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.ylim([1e-4, 1e0])
    plt.show()


def vraag6(r):
    Eb_N0_dB_range = np.arange(0, r, 1)
    
    # Convert Eb/N0 from dB to linear scale
    Eb_N0_range_linear = 10 ** (Eb_N0_dB_range / 10)

    ber_bpsk = ber_simulation(2, 'PSK', Eb_N0_range_linear)
    ber_qam = ber_simulation(4, 'QAM', Eb_N0_range_linear)
    ber_pam = ber_simulation(4, 'PAM', Eb_N0_range_linear)
    
    # theoretische waarde
    theory_bpsk = theoretical_ber(2, 'PSK', Eb_N0_range_linear)
    theory_qam = theoretical_ber(4, 'QAM', Eb_N0_range_linear)
    theory_pam = theoretical_ber(4, 'PAM', Eb_N0_range_linear)

    ber_data = {
    'bpsk': (ber_bpsk, theory_bpsk),
    'qam': (ber_qam, theory_qam),
    'pam': (ber_pam, theory_pam)
}   

    # Plot
    plot_ber(Eb_N0_dB_range, ber_data)


##vraag 7
def vraag7():
    r = 8
    Eb_N0_dB_range = np.arange(0, r, 1)
    Eb_N0_range_linear = 10 ** (Eb_N0_dB_range / 10)

    phi = 0
    l = [0,0.1,0.2]
    for epsilon in l:
        ber_bpsk = ber_simulation(2, 'PSK', Eb_N0_range_linear,epsilon,phi)
        ber_pam = ber_simulation(4, 'PAM', Eb_N0_range_linear,epsilon,phi)

        # originele waarde
        theory_bpsk = theoretical_ber(2, 'PSK', Eb_N0_range_linear)
        theory_pam = theoretical_ber(4, 'PAM', Eb_N0_range_linear)

        ber_data = {
            'bpsk': (ber_bpsk, theory_bpsk),
            'pam': (ber_pam, theory_pam)
        }

        plot_ber(Eb_N0_dB_range, ber_data)

##vraag 8
def vraag8():
    r = 8
    Eb_N0_dB_range = np.arange(0, r, 1)
    Eb_N0_range_linear = 10 ** (Eb_N0_dB_range / 10)

    epsilon = 0
    l = [0,np.pi/16,np.pi/8,np.pi/4]
    for phi in l:
        ber_qam = ber_simulation(4, 'QAM', Eb_N0_range_linear,epsilon,phi)

        # originele waarde
        theory_qam = theoretical_ber(4, 'QAM', Eb_N0_range_linear)

        ber_data = {
            'qam': (ber_qam,theory_qam)
        }

        plot_ber(Eb_N0_dB_range, ber_data)

# Functie om het oogdiagram te plotten
def plot_oogdiagram(y, T, Ns, num_symbols=2):
    samples_per_symbol = int(Ns * T)
    interval = samples_per_symbol * num_symbols
    num_intervals = len(y) // interval

    plt.figure(figsize=(10, 6))
    for i in range(num_intervals):
        print(i)
        start = i * interval
        end = start + interval
        plt.plot(np.arange(0, interval) / Ns, np.real(y[start:end]), 'b', alpha=0.3)

    plt.xlabel("Tijd (in symboolperioden)")
    plt.ylabel("Amplitude van y(t)")
    plt.title("Oogdiagram")
    plt.grid(True)
    plt.show()

# Parameters
T = 1               # Symboolduur
Ns = 6           # Samples per symboolperiode
alpha_values = [0.05, 0.5, 0.95]  # Verschillende roll-off factoren
Lf = 10              # Puls afknotting

# Bitsequentie genereren en mappen naar BPSK-symbolen
num_bits = 200
bits = np.random.randint(0, 2, num_bits)
a = mapper(bits,2,'PSK')

# Oogdiagram genereren voor verschillende alpha-waarden
for alpha in alpha_values:
    # Moduleren en door een ruisvrij kanaal sturen
    print(alpha)
    sBB = moduleerBB(a, T, Ns, alpha, Lf)
    print(sBB)
    y = basisband_kanaal(sBB,0,1,0)  # h_ch is 1 zonder amplitudeverandering

    # Plot oogdiagram voor de roll-off factor alpha
    print(f"Oogdiagram voor alpha = {alpha}")
    plot_oogdiagram(y, T, Ns, num_symbols=2)
    






#vraag6(8)
#vraag7()
#vraag8()
#vraag4('QAM')
#vraag5('QAM')



