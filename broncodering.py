import numpy as np
import math


def  maak_codetabel_Huffman(waarschijnlijkheden,alfabet):
    """
        Functie die een (gewone) Huffman codetabel maakt
        Input:
            waarschijnlijkheden : 1D numpy array die de waarschijnlijkheid van elk symbool bevat ( een symbool kan niet voorkomen en waarschijnlijkheid 0 hebben; deze symbolen moeten geen codewoord hebben)
            alfabet : 1D numpy array met alle mogelijke symbolen in dezelfde volgorde als rel_freq ; in dit project is het alfabet alle getallen van 0 tot lengte alfabet-1
        Output:
            dictionary: dictionary met symbolen van het alfabet als key en codewoord als value
            gem_len: gemiddelde codewoordlengte
            entropie: entropie van symbolen
    """


    dictionary = {}
    lijst = []
    for i in range(len(waarschijnlijkheden)):
        if waarschijnlijkheden[i]>0:
            lijst.append((waarschijnlijkheden[i],[alfabet[i]]))
            dictionary[alfabet[i]] = ''


    while len(lijst)>1:
        lijst.sort()
        node1 = lijst[0]
        node2 = lijst[1]
        lijst = lijst[2:] # verwijder eerste 2 
        nieuwenode = (node1[0]+node2[0], node1[1]+node2[1])
        lijst.append(nieuwenode)

        for symbool in node1[1]:
            code = dictionary[symbool]
            dictionary[symbool] = '0' + code

        for symbool in node2[1]:
            code = dictionary[symbool]
            dictionary[symbool] = '1' + code  
        

    gem_len = sum([waarschijnlijkheden[i] * len(dictionary[alfabet[i]]) for i in range(len(alfabet)) if waarschijnlijkheden[i] > 0])
    
    entropie= -np.sum([p*math.log2(p) for p in waarschijnlijkheden if p > 0]) # ignore symbol 0 probabi


   # voeg hier je code toe test1234

    return dictionary,gem_len,entropie



def genereer_canonische_Huffman(code_lengtes, alfabet):
    """
        Functie die een canonische Huffman codetabel maakt
        Input:
            code_lengtes : 1D numpy array met de lengte van het Huffmancodewoord voor elk symbool uit het alfabet (merk op dat de codewoordlengte van een symbool ook 0 kan zijn als dit symbool niet voorkomt)
            alfabet : 1D numpy array met alle mogelijke symbolen in dezelfde volgorde als code_lengtes ; in dit project is het alfabet alle getallen van 0 tot lengte alfabet-1
        Output:
            dictionary: dictionary met symbolen van het alfabet als key en het canonische codewoord als value
    """
    #print("alfabet:" , alfabet)
    #print("codelengtes", code_lengtes)

    #zou logischer zijn om in de paired list en gesorteerde lijst, lengte en symbool om te wisselen... boeie is nu zo

    paired_list = [(code_lengtes[i], alfabet[i]) for i in range(len(alfabet))]
    #print(paired_list)
    gesorteerd = sorted(paired_list, key=lambda x: (x[0], x[1]))

    canonische_code_tabel = {}
    code = 0
    vorige_lengte = 0

    for lengte, symb in gesorteerd:
        #print(lengte, symb)
        if lengte == 0:
            continue #skip bij code lengte 0


        #als codelengte toeneemt..
        if lengte > vorige_lengte:
            code <<= (lengte - vorige_lengte)

        #code toewijzen
        canonische_code_tabel[symb] = f"{code:0{lengte}b}"

        code += 1
        vorige_lengte = lengte
    
    
    return canonische_code_tabel


