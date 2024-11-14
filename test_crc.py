import numpy as np
import broncodering
import PNG
import kanaalcodering_foutdetectie as kcfd
from scipy.special import comb
import matplotlib.pyplot as plt


#generator polynomial for CRC-5: g(x) = x^5 + x^4 + x^2 + 1 => 110101
generator = np.array([1, 1, 0, 1, 0, 1])

def test_crc():
    # Example message
    message = np.array([1, 0, 1, 1, 0])  # A random message of length k = 5

    # Determine CRC bits
    crc_bits = kcfd.determine_CRC_bits(message, generator)

    # Display the CRC result
    print(f"Message: {message}")
    print(f"CRC bits: {crc_bits}")

    expected_crc_bits = np.array([0, 1, 1, 0, 1])
    assert np.array_equal(crc_bits, expected_crc_bits), f"Expected {expected_crc_bits}, but got {crc_bits}"

# Simulating binary symmetric channel (BSC) errors
def simulate_bsc_errors(message, error_probability):

    # Maak een mask array waar fouten moeten worden toegepast
    flip_mask = np.random.rand(len(message)) < error_probability
    
    # Flip de bits door XOR toe te passen met de flip_mask
    flipped_message = message ^ flip_mask.astype(int)
    
    return flipped_message

def run_simulation(error_probability, num_trials):
    # Run multiple simulations to estimate the probability of non-detectable errors
    
    undetected_errors = 0

    for _ in range(num_trials):
        CRC_check_passed = False

        message = np.random.randint(0, 2, size=5)  # Random message of length k=5
        crc_bits = kcfd.determine_CRC_bits(message, generator)
        message_with_crc = np.concatenate([message, crc_bits])

        # Simulate transmission with errors
        received_message = simulate_bsc_errors(message_with_crc, error_probability)

        # Check if CRC detects the error
        remainder = kcfd.determine_CRC_bits(received_message, generator)
        if np.all(remainder == 0):
            CRC_check_passed = True

        if np.any(message_with_crc != received_message) and CRC_check_passed:
            #print("UNDETECTED!")
            #print("og message:          ", message_with_crc)
            #print("message with errors: ", received_message)

            undetected_errors += 1

    #print(f"Estimated probability of undetected errors for p={error_probability}: {undetected_errors / num_trials}")

    return undetected_errors/num_trials
# Function to calculate the probability of non-detectable errors
def analytical_prob_undetected(p, k=5, w=4):
    # Total number of bits transmitted including CRC bits
    n = k + len(generator) - 1  # k message bits + degree of generator
    prob = 0.0
    # Calculate the probability for all k mod w == 0
    for i in range(n + 1):
        if i % w == 0:  # Undetectable if the number of errors is a multiple of the weight
            prob += comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return prob


if __name__ == "__main__":
    p_values = np.linspace(0.05, 0.5, 100)  # Probability values from 0.05 to 0.5
    analytical_probs = []
    simulated_probs = []

    # Calculate analytical probabilities
    for p in p_values:
        analytical_probs.append(analytical_prob_undetected(p))

    # Run simulations for different probabilities
    for p in p_values:
        simulated_prob = run_simulation(p, 10000)
        simulated_probs.append(simulated_prob)
    
    #print(simulated_probs)
    #print(analytical_probs)

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(p_values, simulated_probs, label='Simulated Probability', color='red')
    #plt.plot(p_values, analytical_probs, label='Analytical Probability', color='blue')
    plt.xlabel('Error Probability (p)')
    plt.ylabel('Probability of Undetectable Errors')
    plt.title('Probability of Undetectable Errors vs. Error Probability')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.04)
    plt.xlim(0.05, 0.5)
    plt.show()