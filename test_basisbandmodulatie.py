import numpy as np
import moddet
import matplotlib.pyplot as plt


def plot_signal(t, sBB, title):
    """
    Function to plot the baseband signal sBB over time t.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, np.real(sBB), label='Real Part')
    plt.plot(t, np.imag(sBB), label='Imaginary Part', linestyle='--')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()


# Test 1: Simple case with a single symbol
def test_single_symbol():
    T = 1.0  # Symbol interval
    Ns = 6  # Samples per symbol interval
    alpha = 0.5  # Roll-off factor
    Lf = 2  # Pulse spans 2 symbol intervals on each side
    symbols = np.array([1 + 1j])  # Single complex symbol

    sBB = moddet.moduleerBB(symbols, T, Ns, alpha, Lf)
    t = np.arange(-Lf * T, (len(symbols) + Lf) * T, T / Ns)

    print("Test 1: Single symbol output")
    print(sBB)
    print("\n")

    plot_signal(t, sBB, "Test 1: Single Symbol Baseband Signal")

def test_multiple_symbols():
    T = 1.0
    Ns = 6
    alpha = 0.5
    Lf = 2
    symbols = np.array([1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j])  # 4 symbols

    sBB = moddet.moduleerBB(symbols, T, Ns, alpha, Lf)

    # Time vector
    t = np.arange(-Lf * T, (len(symbols) + Lf) * T, T / Ns)
    
    plot_signal(t, sBB, "Test 2: Multiple Symbols Baseband Signal")

def test_modulation_and_demodulation():
    T = 1.0  # Symbol interval
    Ns = 6  # Samples per symbol interval
    alpha = 0.5  # Roll-off factor
    Lf = 2  # Pulse truncation length
    symbols = np.array([1 + 1j, -1 - 1j, 1 - 1j, -1 + 1j])  # Symbol sequence

    # Step 1: Modulate the symbols to create a baseband signal
    sBB = moddet.moduleerBB(symbols, T, Ns, alpha, Lf)
    
    # Step 2: Demodulate the received baseband signal
    y = moddet.demoduleerBB(sBB, T, Ns, alpha, Lf)

    #Step 3: decimated (downsampled y to get z, an estimation of "symbols")
    z = moddet.decimatie(y, Ns, Lf) 

    # Time vector for plotting
    t = np.arange(-Lf * T  , (len(symbols) + Lf ) * T, T / Ns)
    t_demod = np.linspace(0, len(y) * (T / Ns), len(y))

    # Plot the modulated signal
    plot_signal(t, sBB, "Modulated Signal (Baseband)")

    # Plot the demodulated signal
    plot_signal(t_demod, y, "Demodulated Signal")

    print_symbols(symbols)
    print("sbb:")
    print_symbols(sBB)
    print("y:")
    print_symbols(y)
    print_symbols(z)

def print_symbols(symbols):
    """
    Function to print complex symbols in a nicely formatted way.
    Input:
        symbols: 1D numpy array or list of complex numbers
    """
    print("Formatted Output:")
    for i, symbol in enumerate(symbols):
        # Format the real and imaginary parts with 3 decimal precision
        real_part = f"{symbol.real:.3f}"
        imag_part = f"{abs(symbol.imag):.3f}"  # Take absolute value for formatting
        sign = "+" if symbol.imag >= 0 else "-"  # Determine the sign of the imaginary part
        print(f"Symbol {i+1}: {real_part} {sign} {imag_part}j")


#test_single_symbol()
#test_multiple_symbols()
test_modulation_and_demodulation()


