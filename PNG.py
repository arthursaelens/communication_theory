import struct
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.signal import deconvolve
from zlib import adler32
import matplotlib.pyplot as plt
import time
import cv2
import broncodering
import kanaalcodering_foutdetectie as kanaalcodering
import pickle


def plot_afbeelding(img):
    """ Niet aanpassen!
    Functie die een 3D matrix met RGB waarden plot
    Input:
        img = 3D numpy array met de RGB waarden van een afbeelding
    """
    plt.figure()
    plt.imshow(img.astype(int))
    plt.show()


def convert_bytes_to_bits(byte_array):
    """ Niet aanpassen!
    Functie die bytes omzet naar bits
    Input:
        byte_array = 1D numpy array met waarden tussen 0 en 255 waarbij elke waarde een byte voorstelt
    Output:
        bit_array = 1D numpy array waarbij elke waarde in byte_array is omgezet naar 8 bits (0 of 1)
    """
    bit_array=np.unpackbits(np.array(byte_array,dtype=np.uint8))
    return bit_array

def convert_bits_to_bytes(bit_array):
    """ Niet aanpassen!
    Functie die bits omzet naar bytes
    Input:
        bit_array = 1D numpy array met waarden 0 of 1 waarbij elke waarde een bit voorstelt
    Output:
        byte_array = 1D numpy array waarbij elke 8 waarden in bit_array zijn omgezet naar een byte (waarden tussen 0 en 255)
    """
    byte_array=np.packbits(np.array(bit_array,dtype=np.uint8))
    return byte_array

def read_RGB_values(filename):
    """ Niet aanpassen!
    Functie die de RGB waarden van een afbeelding inleest; in dit project beschouwen we enkel RGB afbeeldingen en geen RGBA of grayscale
    Input:
        filename = naam van het bestand met de RGB waarden van een afbeelding; in dit geval een .pkl bestand (str)
    Output:
        img_rgb = 3D matrix met dimensie [h,w,3] waarbij h en w respectievelijk de hoogte en breedte van de afbeelding zijn (in pixels) en elke pixel voorgesteld wordt door 3 waarden (de RGB waarden)
    """
    file=open(filename,'rb')
    img_rgb=pickle.load(file)
    file.close()

    return img_rgb


def determine_CRC_bytes(byte_array):
    """ Niet aanpassen!
    Functie die de CRC bytes berekend voor een byte array met de te encoderen bytes
    Input:
        byte_array = 1D numpy array met de te encoderen bytes
    Output:
        crc = list met de crc bytes horende bij de byte_array (list)
    """
    CRC_polynomial=[1,1,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1]

    bit_array=np.unpackbits(np.asarray(byte_array,dtype=np.uint8)[-1::-1])#bytes omzetten naar bits
    bit_array[(-min(32,len(bit_array)))::]=bit_array[-min(32,len(bit_array))::]+1#nodig om CRC berekening van cursus te laten overeenkomen met het CRC-32 algoritme
    bit_array=bit_array%2

    crc_bits=np.flip(kanaalcodering.determine_CRC_bits(np.flip(bit_array),np.flip(CRC_polynomial)))

    crc_bits[:min(32,len(bit_array))]=(crc_bits[:min(32,len(bit_array))]+1)%2#nodig om CRC berekening van cursus te laten overeenkomen met het CRC-32 algoritme
    crc_bits=np.asarray(crc_bits).astype(int)
    crc=[]
    for i in range(0,len(crc_bits),8):
        crc.append(int("".join(str(el) for el in crc_bits[i:i+8]),2)) #bits terug omzetten naar bytes
    return crc

class PNG_encoder:
    def __init__(self):
        # definieer parameters PNG afbeelding
        self.colour_type=2# 0 = greyscale, 2 = true colour, 3 = indexed-colour; 4= greyscale with alpha, 6 = true colour with alpha
        self.filter_type= 0 # 0 = None, 1 =  Sub, 2 = Up, 3 = Average, 4 = Paeth
        self.interlace=False
        self.bit_depth=8 #1, 2, 4, 8 or 16
        self.compression_method=0 #0
        self.filter_method=0 #0


        self.data=[]#deze variabele bevat de bits van de PNG afbeelding na encode())

        #deze variabelen bevatten de gegevens van de 3 Huffmancodes in het PNG-bestand
        self.Huffmancode1={}
        self.Huffmancode2={}
        self.Huffmancode3={}


    def encode(self,img, filename):
        """ Niet aanpassen!
            Functie die de RGB waarden van een afbeelding encodeert in een PNG bestand
            Input:
                img = RGB waarden van een afbeelding; numpy array met dimensies (h,w,3) waarbij h en w respectievelijk de hoogte en breedte zijn van de afbeelding en elke pixel voorgesteld wordt door een RGB waarde
                filename = naam van het PNG bestand waarin de afbeelding wordt geëncodeerd (str)
            Output:
                data = 1D numpy array die de bits van het PNG-bestand bevat
        """
        self.data=[0x89,0x50,0x4e,0x47,0x0D,0x0a,0x1a,0x0a]#PNG signature
        self.h,self.w,_=np.shape(img)
        self.encode_header()
        self.encode_IDAT(img)
        self.encode_IEND()
        try:
            with open(filename, "wb") as f:
                f.write(bytes(self.data))
        except IOError:
             print('Error While Opening the file!')

        self.data=convert_bytes_to_bits(self.data)
        return self.data

    def encode_header(self):
        """ Niet aanpassen!
            Functie die header informatie aan het PNG-bestand toevoegt
        """
        self.data.extend([0,0,0,0xD,0x49,0x48,0x44,0x52])
        self.data.extend([el for el in struct.pack('>IIBBBBB',self.w,self.h,self.bit_depth,self.colour_type,self.compression_method,self.filter_method,self.interlace)])
        self.data.extend(determine_CRC_bytes(self.data[-17:]))
        pass

    def encode_IDAT(self,img):
        """ Niet aanpassen!
            Functie die de data van de afbeelding aan het PNG bestand toevoegt
            Input:
                img = 3D matrix met dimensie [h,w,3] waarbij h en w respectievelijk de hoogte en breedte van de afbeelding zijn (in pixels) en elke pixel voorgesteld wordt door 3 waarden (de RGB waarden)
        """
        chunk_type_flags=[0x49,0x44,0x41,0x54]
        chunk_type_flags.extend([0x8,0xD7])
        img2=img.reshape((-1))
        img2=np.insert(img2,np.arange(0,len(img2),self.w*3),0)
        compressed_datastream,rel_freq_literals,rel_freq_distances=self.compress(img2)
        hlit=np.where(rel_freq_literals>0)[0][-1]+1-257
        hdist=np.where(rel_freq_distances>0)[0][-1]+1-1

        dictionary_literals,gem_len,entropy=broncodering.maak_codetabel_Huffman(rel_freq_literals,np.arange(len(rel_freq_literals)))
        self.Huffmancode1["gem_len"]=gem_len
        self.Huffmancode1["entropie"]=entropy
        self.Huffmancode1["alfabet"]=np.arange(len(rel_freq_literals))
        self.Huffmancode1["waarschijnlijkheden"]=rel_freq_literals
        self.Huffmancode1["dictionary"]=dictionary_literals

        dictionary_distances,gem_len,entropy=broncodering.maak_codetabel_Huffman(rel_freq_distances,np.arange(len(rel_freq_distances)))
        self.Huffmancode2["gem_len"]=gem_len
        self.Huffmancode2["entropie"]=entropy
        self.Huffmancode2["alfabet"]=np.arange(len(rel_freq_distances))
        self.Huffmancode2["waarschijnlijkheden"]=rel_freq_distances
        self.Huffmancode2["dictionary"]=dictionary_distances

        literals_lengths=np.asarray([len(dictionary_literals[el]) for el in dictionary_literals.keys()])
        distances_lengths=np.asarray([len(dictionary_distances[el]) for el in dictionary_distances.keys()])

        keys_literals=np.asarray(list(dictionary_literals.keys()))
        keys_distances=np.asarray(list(dictionary_distances.keys()))
        dict_literals=broncodering.genereer_canonische_Huffman(literals_lengths,keys_literals)
        dict_distances=broncodering.genereer_canonische_Huffman(distances_lengths,keys_distances)

        for el in dict_literals.values():
            assert len(el)<16,f'length of huffman codeword too long:{len(el)}'
        for el in dict_distances.values():
            assert len(el)<16,f'length of huffman codeword too long:{len(el)}'

        code_symbols_literals,count_literals=self.determine_code_symbols(dict_literals)
        code_symbols_distances,count_distances=self.determine_code_symbols(dict_distances)
        count=count_literals+count_distances

        dictionary_code,gem_len,entropy=broncodering.maak_codetabel_Huffman(np.asarray(count/np.sum(count)),np.arange(len(count)))
        self.Huffmancode3["gem_len"]=gem_len
        self.Huffmancode3["entropie"]=entropy
        self.Huffmancode3["alfabet"]=np.arange(len(count))
        self.Huffmancode3["waarschijnlijkheden"]=np.asarray(count/np.sum(count))
        self.Huffmancode3["dictionary"]=dictionary_code

        code_lengths=np.asarray([len(dictionary_code[el]) for el in dictionary_code.keys()])
        keys=np.asarray(list(dictionary_code.keys()))

        dict_code=broncodering.genereer_canonische_Huffman(code_lengths,keys)
        order=[16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
        temp=count[order]
        hclen=np.where(temp>0)[0][-1]+1-4

        bitstream=''
        bitstream+='101'#lastblock+block type
        temp='{0:05b}'.format(hlit)
        bitstream+=temp[::-1]
        temp='{0:05b}'.format(hdist)
        bitstream+=temp[::-1]
        temp='{0:04b}'.format(hclen)
        bitstream+=temp[::-1]

        for el in order[:hclen+4]:
            if el in dict_code.keys():
                temp=len(dict_code[el])
            else:
                temp=0
            temp='{0:03b}'.format(temp)

            bitstream+=temp[::-1]

        i=0
        while i <len(code_symbols_literals):
            bitstream+=dict_code[code_symbols_literals[i]]
            if code_symbols_literals[i]==16:
                temp='{0:03b}'.format(code_symbols_literals[i+1]-3)
                bitstream+=temp[::-1]
                i+=2
            elif code_symbols_literals[i]==17:
                temp='{0:03b}'.format(code_symbols_literals[i+1]-3)
                bitstream+=temp[::-1]
                i+=2
            elif code_symbols_literals[i]==18:
                temp='{0:07b}'.format(code_symbols_literals[i+1]-11)
                bitstream+=temp[::-1]
                i+=2
            else:
                i+=1

        i=0
        while i <len(code_symbols_distances):
            bitstream+=dict_code[code_symbols_distances[i]]
            if code_symbols_distances[i]==16:
                temp='{0:03b}'.format(code_symbols_distances[i+1]-3)
                bitstream+=temp[::-1]
                i+=2
            elif code_symbols_distances[i]==17:
                temp='{0:03b}'.format(code_symbols_distances[i+1]-3)
                bitstream+=temp[::-1]
                i+=2
            elif code_symbols_distances[i]==18:
                temp='{0:07b}'.format(code_symbols_distances[i+1]-11)
                bitstream+=temp[::-1]
                i+=2
            else:
                i+=1

        i=0

        while i<len(compressed_datastream):
            bitstream+=dict_literals[compressed_datastream[i]]
            if compressed_datastream[i]>256:
                bits=compressed_datastream[i+2]
                if bits>0:
                    temp=('{0:0'+str(bits)+'b}').format(compressed_datastream[i+1])
                    bitstream+=temp[::-1]

                bitstream+=dict_distances[compressed_datastream[i+3]]
                bits=compressed_datastream[i+5]
                if bits>0:
                    temp=('{0:0'+str(bits)+'b}').format(compressed_datastream[i+4])
                    bitstream+=temp[::-1]
                i+=6
            else:
                i+=1
        temp=8-len(bitstream)%8
        bitstream+='0'*temp

        data=[int(bitstream[i*8:(i+1)*8][::-1],2) for i in range (int(len(bitstream)/8))]

        data.extend([int(el) for el in struct.pack('>I',adler32(bytes(list(img2))))])
        temp=len(data)+2
        data2=list(struct.pack('>I',temp))
        data2.extend(chunk_type_flags)
        data2.extend(data)

        self.data.extend(data2)
        self.data.extend(determine_CRC_bytes(data2[4:]))
        print(f'Length IDAT chunk:{len(data2)+4}bytes')
        pass

    def encode_IEND(self):
        """ Niet aanpassen!
            Functie die IEND chunk aan het PNG-bestand toevoegt
        """
        self.data.extend([0,0,0,0,0x49,0x45,0x4e,0x44,0xae,0x42,0x60,0x82])
        pass

    @staticmethod
    def determine_code_symbols(dict):
        """ Niet aanpassen!
            Functie die dictionary van Huffmancode omzet naar symbolen van een andere Huffmancode (nodig voor de derde Huffmancode)
            Input:
                dict = dictionary van de (canonische) Huffmancode
            Output:
                code_symbols = lijst met symbolen voor de derde Huffmancode (list)
                count = 1D numpy array met voor elk symbool (in code_symbols) het aantal keer dat het voorkomt
        """
        dict={k: v for k, v in dict.items() if v!=''}
        code_symbols=[]
        count=np.zeros(19,dtype=int)
        literals_keys=list(dict.keys())
        i=0
        previous_symbol=-1
        while i <len(literals_keys):
            key=literals_keys[i]
            if i==0 and key>0:
                temp=key
                while temp>10:
                    code_symbols.append(18)
                    count[18]+=1
                    code_symbols.append(min(138,temp))
                    temp=temp-min(138,temp)
                    previous_symbol=18
                while temp>2:
                    code_symbols.append(17)
                    count[17]+=1
                    code_symbols.append(min(10,temp))
                    temp=temp-min(10,temp)
                    previous_symbol=17
                while temp>0:
                    code_symbols.append(0)
                    count[0]+=1
                    temp=temp-1
                    previous_symbol=0
            temp=key-literals_keys[i-1]-1
            if i>0 and temp>0:
                while temp>10:
                    code_symbols.append(18)
                    count[18]+=1
                    code_symbols.append(min(138,temp))
                    temp=temp-min(138,temp)
                    previous_symbol=18
                while temp>2:
                    code_symbols.append(17)
                    count[17]+=1
                    code_symbols.append(min(10,temp))
                    temp=temp-min(10,temp)
                    previous_symbol=17
                while temp>0:
                    code_symbols.append(0)
                    count[0]+=1
                    temp=temp-1
                    previous_symbol=0
            if len(code_symbols)>0 and previous_symbol==len(dict[key]):
                temp=key+1
                while temp in literals_keys and len(dict[key])==len(dict[temp]):
                    temp+=1
                if 3<=temp-key<=6:
                    code_symbols.append(16)
                    count[16]+=1
                    code_symbols.append(temp-key)
                    i+=(temp-key)
                    previous_symbol=16
                else:
                    code_symbols.append(len(dict[key]))
                    count[len(dict[key])]+=1
                    i+=1
                    previous_symbol=len(dict[key])
            else:
                code_symbols.append(len(dict[key]))
                count[len(dict[key])]+=1
                i+=1
                previous_symbol=len(dict[key])
        return code_symbols, count

    def compress(self, datastream):
        """ Niet aanpassen!
            Functie die de datastream comprimeert met LZ77 codering
            Input:
                datastream = 1D numpy array met de datastream die gecomprimeerd moet worden
            Output:
                compressed_datastream = list met de gecomprimeerde datastream
                rel_freq_literal = 1D numpy array met de relatieve frequentie van de symbolen van de Huffmancode voor de RGB-waarden + lengte van de herhalingen voor LZ77 codering
                rel_freq_distance = 1D numpy array met de relatieve frequentie van de symbolen van de Huffmancode voor de afstand waar de herhaling begint voor LZ77 codering
        """
        history_window=128
        rel_freq_literal=np.zeros(286,dtype=int)
        rel_freq_distance=np.zeros(30,dtype=int)
        compressed_datastream=[]
        i=0
        lengths_symbols=[257,258,259,260,261,262,263,264]+[265]*2+[266]*2+[267]*2+[268]*2+[269]*4+[270]*4+[271]*4+[272]*4+[273]*8+[274]*8+[275]*8+[276]*8+[277]*16+[278]*16+[279]*16+[280]*16+[281]*32+[282]*32+[283]*32+[284]*32+[285]
        lengths_bits=[0,0,0,0,0,0,0,0]+[1]*8+[2]*16+[3]*32+[4]*64+[5]*128+[0]
        lengths_helper=[3,4,5,6,7,8,9,10]+[11]*2+[13]*2+[15]*2+[17]*2+[19]*4+[23]*4+[27]*4+[31]*4+[35]*8+[43]*8+[51]*8+[59]*8+[67]*16+[83]*16+[99]*16+[115]*16+[131]*32+[163]*32+[195]*32+[227]*32+[258]

        distance_symbols=[0,1,2,3]+[4]*2+[5]*2+[6]*4+[7]*4+[8]*8+[9]*8+[10]*16+[11]*16+[12]*32+[13]*32+[14]*64+[15]*64+[16]*128+[17]*128+[18]*256+[19]*256+[20]*512+[21]*512+[22]*1024+[23]*1024+[24]*2048+[25]*2048+[26]*4096+[27]*4096+[28]*8192+[29]*8192

        distance_bits=[[0]]*4+[[i]*2**(i+1) for i in range(1,14)]
        distance_bits=[el for sublist in distance_bits for el in sublist]
        distance_helper=[1,2,3,4]+[5]*2+[7]*2+[9]*4+[13]*4+[17]*8+[25]*8+[33]*16+[49]*16+[65]*32+[97]*32+[129]*64+[193]*64+[257]*128+[385]*128+[513]*256+[769]*256+[1025]*512+[1537]*512+[2049]*1024+[3073]*1024+[4097]*2048+[6145]*2048+[8193]*4096+[12289]*4096+[16385]*8192+[24577]*8192
        while i < len(datastream):

            horizon=1
            temp=self.find_sub_list(datastream[i:i+horizon],datastream[max(0,i-history_window):i])
            while i+ horizon <=len(datastream) and len(temp)>0 and horizon<=258:
                horizon+=1
                temp=self.find_sub_list(datastream[i:i+horizon],datastream[max(0,i-history_window):i])
            if horizon<5:
                compressed_datastream.append(datastream[i])
                rel_freq_literal[datastream[i]]+=1
                i+=1
            else:
                temp=self.find_sub_list(datastream[i:i+horizon-1],datastream[max(0,i-history_window):i])
                length=temp[-1][1]-temp[-1][0]+1
                compressed_datastream.append(lengths_symbols[length-3])
                compressed_datastream.append(length-lengths_helper[length-3])
                compressed_datastream.append(lengths_bits[length-3])
                distance=i-(temp[-1][0]+max(0,i-history_window))
                compressed_datastream.append(distance_symbols[distance-1])
                compressed_datastream.append(distance-distance_helper[distance-1])
                compressed_datastream.append(distance_bits[distance-1])
                rel_freq_literal[lengths_symbols[length-3]]+=1
                rel_freq_distance[distance_symbols[distance-1]]+=1
                i+=length
        compressed_datastream.append(256)
        rel_freq_literal[256]=1
        rel_freq_literal=rel_freq_literal/np.sum(rel_freq_literal)
        rel_freq_distance=rel_freq_distance/np.sum(rel_freq_distance)
        return compressed_datastream,rel_freq_literal,rel_freq_distance

    @staticmethod
    def find_sub_list(sl,l):
        """ Niet aanpassen!
            Functie die een patroon zoekt in een numpy array  (helper functie voor LZ77 codering)
            Input:
                sl = 1D numpy array met het patroon dat moet gevonden worden in l
                l =  1D numpy array waarin het patroon gevonden moet worden
            Output:
                results = lijst met tuples waar het patroon begint en eindigt in l
        """
        results=[]
        sll=len(sl)
        if len(l)>0 and sll>0:
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if ind+sll <= len(l) and np.all(l[ind:ind+sll]==sl):
                    results.append((ind,ind+sll-1))
        return results



class PNG_decoder:
    def __init__(self, bit_array):
        #Input:
        #   bit_array = 1D numpy array bestaande uit bits (0 of 1) en die overeenkomt met de bytestream van het PNG bestand

        byte_array=convert_bits_to_bytes(bit_array) # zet bits om naar bytes voor de rest van de klasse
        self.byte_array=byte_array
        self.supported_chunk_types=['IHDR','IDAT','IEND'] # chunktypes die de klasse ondersteunt
        self.data=[] # gedecodeerde bytestream; met get_image() wordt deze omgezet in de RGB waarden van de afbeelding

    def decode(self):
        """ Niet aanpassen!
            Functie die de bit_array van een PNG-bestand (als input aan init methode gegeven) decodeert
            Output:
                succes = geeft aan of de decodering succesvol is verlopen
        """
        byte_array=self.byte_array.copy()
        byte_array, success=PNG_decoder.read_signature(byte_array)
        if not success:
            return success

        byte_array,success= self.read_header(byte_array)
        if not success:
            return success

        byte_array,success=self.read_chunks(byte_array)
        return success

    @staticmethod
    def read_signature(byte_array):
        """ Niet aanpassen!
            Functie die kijkt of de eerste 8 bytes overeenkomen met de signature van een PNG-bestand
            Input:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand
            Output:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand zonder de signature
                success = geeft aan of de signature juist is
        """
        success=(np.all(byte_array[:8]==[137,80,78,71,13,10,26,10]))
        byte_array = byte_array[8:]
        return byte_array,success


    def read_header(self,byte_array):
        """ Niet aanpassen!
            Functie die de header chunk van het PNG-bestand inleest en decodeert
            Input:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand
            Output:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand zonder de header chunk
                success = geeft aan of de header succesvol gedecodeerd is
        """
        success=np.all((byte_array[:4]==[0,0,0,13])) # lengte van header chunk moet 13 zijn
        success=success and np.all((byte_array[4:8]==[73,72,68,82]))
        temp=bytes(byte_array[8:21])
        (self.width,self.height,self.bit_depth,self.colour_type,self.compression_method, self.filter_method,self.interlace_method)=struct.unpack('>IIBBBBB',temp)
        crc=determine_CRC_bytes(byte_array[4:21])
        if np.all((crc==byte_array[21:25])):
            pass
        else:
            success=False
        byte_array=byte_array[25:]
        return  byte_array,success


    def read_chunks(self,byte_array):
        """ Niet aanpassen!
            Functie die de chunks van het PNG-bestand inleest en decodeert
            Input:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand
            Output:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand zonder de gedecodeerde chunks
                success = geeft aan of de chunks succesvol gedecodeerd zijn
        """
        success=True
        IDAT_read=False
        while True:
            if len(byte_array)>=8:
                try:
                    chunk_length=struct.unpack('>I',bytes(byte_array[:4]))[0]
                    chunk_type=struct.unpack('4s',bytes(byte_array[4:8]))[0].decode("ascii")
                except:
                    chunk_length=None
                    chunk_type=None
            else:
                return byte_array,False
            if chunk_type=='IEND':
                return byte_array, (success and IDAT_read)
            if chunk_type=='IDAT':
                IDAT_read=True
            if chunk_type not in self.supported_chunk_types:
                byte_array=byte_array[1:]
            else:
                method_name='read_'+chunk_type
                method=getattr(self, method_name)

                byte_array, success_temp= method(byte_array, chunk_length)
                success=success and success_temp

        pass


    def read_IDAT(self,byte_array,chunk_length):
        """ Niet aanpassen!
            Functie die een IDAT chunk van het PNG-bestand inleest en decodeert
            Input:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand
            Output:
                byte_array = 1D numpy array met de bytestream van het PNG-bestand zonder het gedecodeerde IDAT chunk
                success = geeft aan of de IDAT chunk succesvol gedecodeerd is
        """
        if len(byte_array)<12+chunk_length:
            return byte_array[1:],False
        success=True
        zlib_compression_method_flag='{0:08b}'.format(byte_array[8])
        compression_method=int(zlib_compression_method_flag[:4],2)
        compression_info=int(zlib_compression_method_flag[4:],2)

        additional_flags='{0:08b}'.format(byte_array[9])
        fdict=int(additional_flags[-6],2)
        compression_level=int(additional_flags[0:2],2)
        fcheck=int(additional_flags[-5:])
        if (int(zlib_compression_method_flag,2)*256+int(additional_flags,2)) %31 !=0:
            # print('check fails')
            pass
        data_bytes=byte_array[10:12+chunk_length-4]

        bitstring=''.join(str(el) for el in np.unpackbits(np.asarray(data_bytes,dtype=np.uint8),bitorder='little'))
        adler32_checksum=self.decompress_png(bitstring)
        del bitstring

        temp=struct.unpack('>I',bytes(byte_array[chunk_length+4:chunk_length+8]))[0]
        crc=determine_CRC_bytes(byte_array[4:8+chunk_length])
        if np.all((crc==byte_array[8+chunk_length:12+chunk_length])) and (temp==adler32_checksum):
            # print('crc IDAT OK')
            pass
        else:
            success=False
        byte_array=byte_array[12+chunk_length:]
        return byte_array,success

    def decompress_png(self,bitstream):
        """ Niet aanpassen!
            Functie die de bitstream in een IDAT chunk decodeert
            Input:
                bitstream = 1D numpy array met de bitstream van de IDAT chunk
            Output:
                adler32_checksum = checksum op gedecodeerde data die gebruikt kan worden voor foutdetectie
        """
        data=[]
        while True:
            if len(bitstream)<17:
                return -1
            lastblock=int(bitstream[0])

            btype=bitstream[1:3]
            btype=int(btype[-1::-1],2)

            hlit=bitstream[3:8]
            hlit=int(hlit[-1::-1],2)+257

            hdist=bitstream[8:13]
            hdist=int(hdist[-1::-1],2)+1

            hclen=bitstream[13:17]
            hclen=int(hclen[-1::-1],2)+4

            bitstream=bitstream[17:]

            if hlit>286 or hdist>30 or hclen>19:
                return -1

            order=[16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15]
            cl_code_length=np.zeros((19),dtype=int)

            for i in range(hclen):
                if len(bitstream)<3:
                    return -1
                temp=bitstream[:3]
                cl_code_length[order[i]]=int(temp[-1::-1],2)
                bitstream=bitstream[3:]

            dict_huffmann=broncodering.genereer_canonische_Huffman(cl_code_length,np.arange(len(cl_code_length)))
            dict_huffmann = {v: k for k, v in dict_huffmann.items() if v!=''}

            literals_length=np.zeros((286),dtype=int)
            distances_length=np.zeros((30),dtype=int)


            nr_symbols_read=0
            last_symbol=-1
            while nr_symbols_read <hlit:
                horizon=1
                while ''.join(bitstream[:horizon]) not in dict_huffmann.keys():
                    horizon+=1
                    if horizon>np.max(cl_code_length) or len(bitstream)<horizon:
                        return -1
                temp=dict_huffmann[''.join(bitstream[:horizon])]

                if temp<16:
                    literals_length[nr_symbols_read]=temp
                    bitstream=bitstream[horizon:]
                    nr_symbols_read+=1
                    last_symbol=temp
                elif temp==16:
                    if last_symbol==-1 or len(bitstream)<horizon+2:
                        return -1
                    temp2=bitstream[horizon:horizon+2]
                    temp2=int(temp2[-1::-1],2)+3
                    literals_length[nr_symbols_read:nr_symbols_read+temp2]=last_symbol
                    nr_symbols_read+=temp2

                    bitstream=bitstream[horizon+2:]
                elif temp==17:
                    if len(bitstream)<horizon+3:
                        return -1
                    temp2=bitstream[horizon:horizon+3]
                    temp2=int(temp2[-1::-1],2)

                    nr_symbols_read+=3+temp2
                    bitstream=bitstream[horizon+3:]
                    last_symbol=0
                elif temp==18:
                    if len(bitstream)<horizon+7:
                        return -1
                    temp2=bitstream[horizon:horizon+7]
                    temp2=int(temp2[-1::-1],2)

                    nr_symbols_read+=11+temp2
                    bitstream=bitstream[horizon+7:]
                    last_symbol=0

            nr_symbols_read=0
            last_symbol=-1
            while nr_symbols_read <hdist:

                horizon=1

                while ''.join(bitstream[:horizon]) not in dict_huffmann.keys():
                    horizon+=1
                    if horizon>np.max(cl_code_length) or len(bitstream)<horizon:
                        return -1
                temp=dict_huffmann[''.join(bitstream[:horizon])]
                if temp<16:
                    distances_length[nr_symbols_read]=temp
                    bitstream=bitstream[horizon:]
                    nr_symbols_read+=1
                    last_symbol=temp
                elif temp==16:
                    if last_symbol==-1 or len(bitstream)<horizon+2:
                        return -1
                    temp2=bitstream[horizon:horizon+2]
                    temp2=int(temp2[-1::-1],2)+3
                    distances_length[nr_symbols_read:nr_symbols_read+temp2]=last_symbol
                    nr_symbols_read+=temp2
                    bitstream=bitstream[horizon+2:]
                elif temp==17:
                    if len(bitstream)<horizon+3:
                        return -1
                    temp2=bitstream[horizon:horizon+3]
                    temp2=int(temp2[-1::-1],2)
                    nr_symbols_read+=3+temp2
                    bitstream=bitstream[horizon+3:]
                    last_symbol=0
                elif temp==18:
                    if len(bitstream)<horizon+7:
                        return -1
                    temp2=bitstream[horizon:horizon+7]
                    temp2=int(temp2[-1::-1],2)
                    nr_symbols_read+=11+temp2
                    bitstream=bitstream[horizon+7:]
                    last_symbol=0

            if np.sum(literals_length)==0:
                return -1
            if np.sum(distances_length)==0:
                return -1
            dict_literals=broncodering.genereer_canonische_Huffman(literals_length,np.arange(len(literals_length)))
            dict_literals = {v: k for k, v in dict_literals.items() if v!=''}
            dict_distances=broncodering.genereer_canonische_Huffman(distances_length,np.arange(len(distances_length)))
            dict_distances = {v: k for k, v in dict_distances.items() if v!=''}

            lengths_length=[3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258]
            lengths_bits=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
            distances_length=[1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577]
            distances_bits=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13]

            min1=np.min(literals_length)
            min2=np.min(distances_length)
            max1=np.max(literals_length)
            max2=np.max(distances_length)

            while True:

                horizon=min1
                while ''.join(bitstream[:horizon]) not in dict_literals.keys():
                    horizon+=1
                    if horizon>max1 or len(bitstream)<horizon:
                        return -1
                temp=dict_literals[''.join(bitstream[:horizon])]

                if temp<256:
                    data.append(temp)
                    bitstream=bitstream[horizon:]

                elif temp==256:
                    bitstream=bitstream[horizon:]
                    break #end of block
                else:

                    bits=lengths_bits[temp-257]
                    length=lengths_length[temp-257]
                    if bits>0:
                        if len(bitstream)<horizon+bits:
                            return -1
                        temp2=bitstream[horizon:horizon+bits]
                        temp2=int(temp2[-1::-1],2)
                        length+=temp2

                    bitstream=bitstream[horizon+bits:]

                    horizon=min2
                    while ''.join(bitstream[:horizon]) not in dict_distances.keys():
                        horizon+=1
                        if horizon>max2 or len(bitstream)<horizon:
                            return -1
                    temp=dict_distances[''.join(bitstream[:horizon])]
                    bits=distances_bits[temp]
                    distance=distances_length[temp]

                    if bits>0:
                        if len(bitstream)<horizon+bits:
                            return -1
                        temp2=bitstream[horizon:horizon+bits]
                        temp2=int(temp2[-1::-1],2)

                        distance+=temp2

                    bitstream=bitstream[horizon+bits:]
                    if length<distance:
                        data.extend(data[-distance:-distance+length])
                    elif distance==length:
                         data.extend(data[-distance:])
                    else:
                        data.extend(data[-distance:])
                        length=length-distance
                        while length>0:
                            if length<distance:
                                data.extend(data[-distance:-distance+length])
                                length=0
                            else:
                                data.extend(data[-distance:])
                                length=length-distance

                        data.extend(data[-distance:-distance+length-distance])

            if lastblock:
                break


        self.data.extend(data)

        adler32_checksum=adler32(bytes(data))
        return adler32_checksum


    def get_image(self):
        """ Niet aanpassen!
            Functie die de RGB waarden van de gedecodeerde PNG-afbeelding genereert
            Output:
                img = 3D numpy array met de RGB waarden van de afbeelding
        """
        w=self.width
        h=self.height
        data=self.data.copy()

        if self.colour_type==2:
            del data[0::(w*3+1)]
            img=np.asarray(data)
        elif self.colour_type==3:
            del data[0::(w+1)]
            img=self.palette[data]
        if not(self.colour_type ==2):
            print("unsupported colour type")

        img=img.reshape((h,w,3))
        return img



if __name__=='__main__':
    #voorbeeld hoe code gebruikt kan worden

    img=read_RGB_values('afbeelding1.pkl')
    encoder=PNG_encoder()
    png_bits=encoder.encode(img,'test.png')

    decoder=PNG_decoder(png_bits)
    succes=decoder.decode()

    img2=decoder.get_image()

    plot_afbeelding(img2)

