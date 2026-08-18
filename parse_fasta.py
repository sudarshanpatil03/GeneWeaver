from Bio import SeqIO

def chunk_sequence(sequence, chunk_size):
    """Yield successive chunks from sequence."""
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:i + chunk_size]

def parse_fasta(file_path, chunk_size=50):
    print(f"Parsing {file_path} with chunk size {chunk_size}...\n")
    try:
        for record in SeqIO.parse(file_path, "fasta"):
            print(f"ID: {record.id}")
            print(f"Description: {record.description}")
            print(f"Sequence Length: {len(record.seq)}")
            
            # Chunk the sequence into manageable arrays (strings here, but can be lists of chars)
            sequence_str = str(record.seq)
            chunks = list(chunk_sequence(sequence_str, chunk_size))
            
            print(f"Total Chunks: {len(chunks)}")
            print(f"First 2 chunks:")
            for i, chunk in enumerate(chunks[:2]):
                print(f"  Chunk {i+1} (length {len(chunk)}): {chunk}")
            print("-" * 40)
    except Exception as e:
        print(f"Error parsing FASTA file: {e}")

if __name__ == "__main__":
    parse_fasta("mock_genome.fasta", chunk_size=50)
