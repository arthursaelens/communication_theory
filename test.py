import numpy as np
import broncodering
import PNG
import kanaalcodering_foutdetectie

# Testcaseje 
alfabet = np.array([0, 1, 2, 3])
waarschijnlijkheden = np.array([0.4, 0.3, 0.2, 0.1])
code_lengtes = np.array([1,2,3,3])

expected_dictionary = {
    0: '0',
    1: '10',
    2: '110',
    3: '111'
}
expected_code_lengths = {
    0: 1,  
    1: 2,
    2: 3,
    3: 3  
}
expected_average_length = (0.4 * 1 + 0.3 * 2 + 0.2 * 3 + 0.1 * 3)
expected_entropy = -((0.4 * np.log2(0.4)) + (0.3 * np.log2(0.3)) + (0.2 * np.log2(0.2)) + (0.1 * np.log2(0.1)))

#Huffman coding function van broncodering.py
dictionary, gem_len, entropie = broncodering.maak_codetabel_Huffman(waarschijnlijkheden, alfabet)

#test broncodering.maak_codetabel_Huffman() met asserts
for symbool, code in dictionary.items():
    assert len(code) == expected_code_lengths[symbool], f"Expected length {expected_code_lengths[symbool]} for symbol {symbool}, but got {len(code)}, Expected {expected_dictionary}, but got {dictionary}"
assert abs(gem_len - expected_average_length) < 1e-5, f"Expected average length {expected_average_length}, but got {gem_len}"
assert abs(entropie - expected_entropy) < 1e-5, f"Expected entropy {expected_entropy}, but got {entropie}"

print("geen fout in maak_codetabel_Huffman()")

expected_canonical_codes = {
        0: '0',
        1: '10',
        2: '110',
        3: '111'
    }

#canonische huffman: met zelfde case (we gebriuken codelengtes ingegeven in expected code lenghts)
canonieke_codes = broncodering.genereer_canonische_Huffman(code_lengtes, alfabet)

print("ouput canoniek: ",canonieke_codes)

for symbool, code in canonieke_codes.items():
    expected_code = expected_canonical_codes.get(symbool, None)
    
    # Check of symbool in de verwachte dictionary zit en dat de codes overeenkomen
    assert expected_code is not None, f"Unexpected symbol {symbool} in the result."
    assert code == expected_code, f"Expected code {expected_code} for symbol {symbool}, but got {code}."

print('geen fouten gedetecteerd in canunik van der paele')

# Laad RGB-waarden van afbeelding1.pkl en afbeelding2.pkl
rgb_image1 = PNG.read_RGB_values('afbeelding1.pkl')
rgb_image2 = PNG.read_RGB_values('afbeelding2.pkl')

png_encoder = PNG.PNG_encoder()
png_encoder.encode(rgb_image1,'afbeelding1.png')

print("----------------------------------")
print(png_encoder.Huffmancode1['entropie'])
print("----------------------------------")
print(png_encoder.Huffmancode2['entropie'])
print("----------------------------------")
print(png_encoder.Huffmancode3['entropie'])


