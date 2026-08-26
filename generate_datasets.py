import numpy as np
import os

def generate_genome(size, filename):
    """
    Generates a mock genome sequence of given size and saves it as a .npy file.
    Encoding: 0=A, 1=C, 2=G, 3=T
    """
    print(f"Generating {filename} with size {size} bases...")
    # Generate random bases
    data = np.random.randint(0, 4, size, dtype=np.int8)
    np.save(filename, data)
    print(f"Saved {filename} ({os.path.getsize(filename) / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    
    # 1. Small Dataset: 10 Million bases (~10MB)
    generate_genome(10_000_000, "datasets/small_genome.npy")
    
    # 2. Medium Dataset: 100 Million bases (~100MB)
    generate_genome(100_000_000, "datasets/medium_genome.npy")
    
    # 3. Large Dataset: 1 Billion bases (~1GB)
    # generate_genome(1_000_000_000, "datasets/large_genome.npy") 
    # NOTE: Uncomment the line above to generate the large dataset. 
    # It takes significant disk space (1GB), so we keep it small for initial tests.
    # We will generate a 250MB large dataset for practical initial benchmarking:
    generate_genome(250_000_000, "datasets/large_genome.npy")
    
    print("All test datasets generated successfully.")
