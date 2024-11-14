import numpy as np

def determine_CRC_bits(bit_array, generator):
    """
            Functie die de CRC bits (komt overeen met r(x) uit de opgave) van een bit array bepaalt
            Input:
                bit_array = 1D numpy array met bits (0 of 1) waarvoor de CRC bits moeten berekend worden; de meest linkse bit komt overeen met de coëfficient van de hoogste graad
                generator = generator polynoom van de CRC code (numpy array); de meest linkse bit komt overeen met de coëfficient van de hoogste graad
            Output:
                crc_bits = 1D numpy array met bits (0 of 1) die de crc bits bevat; de meest linkse bit komt overeen met de coëfficient van de hoogste graad
    """

    crc_bits=np.asarray([])

    # Verleng bit_array door het aantal nullen dat nodig is voor de CRC-bits
    n = len(generator) - 1  # Graad van de generator polynoom
    dividend = np.concatenate([bit_array, np.zeros(n, dtype=int)])  # Voeg n nullen toe aan het einde van het bit_array

    # Stap 2: XOR-delingsproces om de rest te vinden
    for i in range(len(bit_array)):
        # Alleen als de huidige bit 1 is, voeren we een XOR uit met de generator
        if dividend[i] == 1:
            # Voer XOR uit tussen het huidige deel van het dividend en het generator-polynoom
            dividend[i:i+len(generator)] ^= generator  # XOR op slice van dividend

    crc_bits = dividend[-n:]  # Laatste n bits zijn de rest (CRC-bits)

    return crc_bits