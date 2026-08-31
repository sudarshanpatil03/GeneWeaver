import json
from Bio import SeqIO
import sys
import argparse

def chunk_sequence(sequence: str, chunk_size: int):
    """Yield successive chunks from sequence."""
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:i + chunk_size]

def parse_and_chunk(file_path: str, chunk_size: int = 50, output_file: str = None):
    try:
        records = list(SeqIO.parse(file_path, "fasta"))
        pipeline_output = []
        
        for record in records:
            sequence_str = str(record.seq)
            chunks = list(chunk_sequence(sequence_str, chunk_size))
            
            record_data = {
                "id": record.id,
                "description": record.description,
                "sequence_length": len(record.seq),
                "total_chunks": len(chunks),
                "chunks": chunks
            }
            pipeline_output.append(record_data)
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(pipeline_output, f, indent=4)
            print(f"Successfully processed {len(records)} records. Output saved to {output_file}")
        else:
            print(json.dumps(pipeline_output, indent=4))
            
    except Exception as e:
        print(f"Error parsing FASTA file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse FASTA and chunk sequences.")
    parser.add_argument("--input", default="mock_genome.fasta", help="Input FASTA file")
    parser.add_argument("--chunk-size", type=int, default=50, help="Chunk size for the sequence")
    parser.add_argument("--output", default="chunked_genome.json", help="Output JSON file (optional)")
    
    args = parser.parse_args()
    
    print(f"Starting Data Pipeline...")
    print(f"Input File: {args.input}")
    print(f"Chunk Size: {args.chunk_size}")
    
    parse_and_chunk(args.input, args.chunk_size, args.output)
