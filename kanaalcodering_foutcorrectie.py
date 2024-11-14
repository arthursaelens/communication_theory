import numpy as np
import random
import matplotlib.pyplot as plt

####Vraag1 (bij C)


H = np.array([
    [0,1,0,0,1,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,0,1,0,1,1,0,0,0,0,1],
    [0,0,1,0,1,0,0,0,0,0,1,0,1,0],
    [0,0,0,0,0,1,0,0,1,1,0,1,0,0],
    [1,0,1,0,1,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,1,0,1,0,0,0,1,0,0]])
n = H.shape[1]
k = H.shape[0]


Hsys=H.copy()
Hsys[:,[0,1,3,8,9,11,12,13]] = Hsys[:,[12,8,11,1,13,9,0,3]]
Ptransposed = Hsys[:,:n-k]
Gsys = np.hstack((Ptransposed.T, np.eye(n-k)))

#G=Gsys.copy()
#G[:, [12, 8, 11, 1, 13, 9, 0, 3]] = G[:, [0, 1, 3, 8, 9, 11, 12, 13]]
#print("nieuwe Hsys\n")
#print(Hsys)
#print("PT:")
#print(Ptransposed)
#print("Gsys:")
#print(Gsys)
#print (G@H.T%2)
#
#print(Gsys@Hsys.T%2)


def encodeer_lineaire_blokcode(bit_array,G):
    """
            Functie die bit_array encodeert met een lineaire blokcode met generatormatrix G
            Input:
                bit_array = 1D numpy array met bits (0 of 1) met de ongecodeerde bits; merk op dat bit_array meerdere informatiewoorden kan bevatten die allemaal geconcateneerd zijn
                G = 2D numpy array met bits (0 of 1) die de generator matrix van de lineaire blokcode bevat (dimensies kxn)
            Output:
                bitenc = 1D numpy array met bits (0 of 1) die de gecodeerde bits bevat; ook hier zijn de bits van de codewoorden geconcateneerd in een 1D numpy array
    """
    bitenc = np.array([])
    k,n = G.shape
    while len(bit_array>= k):
        informatiewoord = bit_array[:k]
        bitenc = np.concatenate((bitenc,(informatiewoord@G)%2))
        bit_array = bit_array[k:]
    return bitenc

# functie die de decoder van de uitwendige code implementeert
def decodeer_lineaire_blokcode(bit_array,H):
    """
        Functie die bit_array decodeert met een lineaire blokcode met checkmatrix H
        Input:
            bit_array = 1D numpy array met bits (0 of 1) met de gecodeerde bits; merk op dat bit_array meerdere codewoorden kan bevatten die allemaal geconcateneerd zijn
            H = 2D numpy array met bits (0 of 1) die de check matrix van de lineaire blokcode bevat (dimensies (n-k)xn)
        Output:
            bitdec = 1D numpy array met bits (0 of 1) die de ongecodeerde bits bevat; ook hier zijn de bits van de informatiewoorden geconcateneerd in een 1D numpy array
            bool_fout = 1D numpy array die voor elke informatiewoord/codewoord aangeeft of er een fout is gedetecteerd
    """  
    p, q = H.shape  # p = n - k, q = n
    bool_fout = []
    bitdec = []

    # Genereer de syndroomtabel
    syndroomtabel = genereer_syndroom_table(H)

    while len(bit_array) >= q:
        codewoord = bit_array[:q]
        bit_array = bit_array[q:]
        
        syndroom = (codewoord @ H.T) % 2
        fout = not np.all(syndroom == 0)
        bool_fout.append(fout)

        if fout:  # Er is een fout gedetecteerd
            foutvector = syndroomtabel[tuple(syndroom)]
            codewoord = (codewoord + foutvector) % 2
        
        bitdec.append(codewoord[:q - p])

    bool_fout = np.array(bool_fout, dtype=int)
    bitdec = np.concatenate(bitdec)  # Concateneer de resultaten naar een 1D array
    return bitdec, bool_fout

def genereer_syndroom_table(H):
    p, q = H.shape  # p = n - k, q = n
    max_fouten = 2 ** q  # we zoeken een vector r voor elk mogelijk syndroom
    #foutvector heeft shape 1* n en minimaal gewicht
    tabel = {}
    for gewicht in range(1, q + 1):
        for r in range(max_fouten):
            if bin(r).count('1') == gewicht:
                binair = bin(r)[2:].zfill(q)
                binair_array = np.array([int(bit) for bit in binair[:q]])
                s = (binair_array @ H.T) % 2
                #syndroom heeft lengte n-k
                # Voeg het syndroom toe aan de tabel als het nog niet bestaat
                if tuple(s) not in tabel.keys() and not np.all(s==0):
                    tabel[tuple(s)] = binair_array
                if len(tabel) >= 2**p-1:
                    return tabel

    return tabel

### testje genereer syndroomtabel

"""
H = np.array([[1, 0, 1, 1],
              [0, 1, 0, 1],
              [1, 1, 0, 0]])

# Genereer de syndroomtabel
syndroom_tabel = genereer_syndroom_table(H)
print("Syndroomtabel:")
for s, foutpatronen in syndroom_tabel.items():
    print(f"Syndroom {s}: Foutpatronen {foutpatronen}")

"""


G = np.array([
    [1,0,0,0,1,1],
    [0,1,0,1,0,1],
    [0,0,1,1,1,0]])
H = np.array([
    [0,1,1,1,0,0],
    [1,0,1,0,1,0],
    [1,1,0,0,0,1]])


"""
def test_lineaire_blokcode():
    # Voorbeeld bit_array met bits (bijv. 110)
    bit_array = np.array([1,1,0,0,0,0])
    # Encodeer de bits
    gecodeerde_bits = encodeer_lineaire_blokcode(bit_array, G)
    print("Gecodeerde bits:", gecodeerde_bits)

    # Simuleer een fout in de gecodeerde bits (bijv. toggle de laatste bit)
    gecodeerde_bits[5] = 1 - gecodeerde_bits[5]

    # Decodeer de bits
    gedecodeerde_bits, fout_gedetecteerd = decodeer_lineaire_blokcode(gecodeerde_bits, H)
    print("Gecodeerde bits met fout:", gecodeerde_bits)
    print("Fout gedetecteerd:", fout_gedetecteerd)
    print("Gedecodeerde bits:", gedecodeerde_bits)

    # Test de syndroomtabel
    syndroom_tabel = genereer_syndroom_table(H)
    print("Syndroomtabel:")
    for s, foutpatronen in syndroom_tabel.items():
        print(f"Syndroom {s}: Foutpatronen {foutpatronen}")

# Voer de test uit
test_lineaire_blokcode()

"""

### Vraag 3 ###

def decodeerfout_sim(p):
    corr = 0
    for _ in range(1000):
        codewoord = np.random.randint(0, 2, 3)
        gecodeerd = encodeer_lineaire_blokcode(codewoord, G)
        codemetruis = np.array([1 - x if random.random() < p else x for x in gecodeerd])
        gedecodeerd, fout = decodeer_lineaire_blokcode(codemetruis, H)
        if not np.array_equal(codewoord, gedecodeerd):
            corr += 1
    return corr / 1000


### exacte berekening kans op decodeerfout bij volledige decodering
def decodeerfout_exact(p):
    pe = 0
    syntabel = genereer_syndroom_table(H)
    for foutvec in syntabel.values():
        w = np.sum(foutvec)
        pe += p**w * (1-p)**(n-w)
    return pe


def decodeerfout_benadering(p):
    pe = 0
    syntabel = genereer_syndroom_table(H)
    for foutvec in syntabel.values():
        w = np.sum(foutvec)
        pe += p**w
    return pe



p_values = np.arange(0.05, 0.55, 0.05)
simulatie_resultaten = [decodeerfout_sim(p) for p in p_values]
exacte_resultaten = [decodeerfout_exact(p) for p in p_values]
benaderde_resultaten = [decodeerfout_benadering(p) for p in p_values]

# Plot de resultaten
plt.figure(figsize=(10, 6))
plt.plot(p_values, simulatie_resultaten, label="Simulatie", marker='o')
plt.plot(p_values, exacte_resultaten, label="Exact", linestyle="--")
plt.plot(p_values, benaderde_resultaten, label="Benadering", linestyle=":")
plt.xlabel("Kans p")
plt.ylabel("Kans op Decodeerfout")
plt.title("Vraag C.3")
plt.legend()
plt.grid(True)
plt.show()