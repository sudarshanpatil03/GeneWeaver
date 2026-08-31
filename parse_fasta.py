import time
from Bio import SeqIO
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, ProgressBar, RichLog

def chunk_sequence(sequence, chunk_size):
    """Yield successive chunks from sequence."""
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:i + chunk_size]

class FastaParserApp(App):
    """A Textual app to parse FASTA files and track chunking progress."""
    
    CSS = """
    ProgressBar {
        margin: 1 2;
    }
    RichLog {
        margin: 0 2 1 2;
        height: 1fr;
        border: solid green;
    }
    """
    
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)
        yield ProgressBar(id="progress", show_eta=True)
        yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        """Called when app starts."""
        self.title = "FASTA Parser & Chunker"
        self.log_widget = self.query_one(RichLog)
        self.progress = self.query_one(ProgressBar)
        
        # Start the parsing in a background worker thread
        self.parse_and_chunk("mock_genome.fasta", chunk_size=50)

    @work(exclusive=True, thread=True)
    def parse_and_chunk(self, file_path: str, chunk_size: int = 50) -> None:
        """Parse the FASTA file and chunk sequences, updating the UI."""
        self.app.call_from_thread(self.log_widget.write, f"[bold green]Parsing {file_path} with chunk size {chunk_size}...[/bold green]\n")
        
        try:
            records = list(SeqIO.parse(file_path, "fasta"))
            
            # Calculate total chunks for the progress bar across all records
            total_chunks = 0
            for record in records:
                total_chunks += (len(record.seq) + chunk_size - 1) // chunk_size
                
            self.app.call_from_thread(self.progress.update, total=total_chunks)
            
            for record in records:
                self.app.call_from_thread(self.log_widget.write, f"[bold cyan]ID:[/bold cyan] {record.id}")
                self.app.call_from_thread(self.log_widget.write, f"[bold cyan]Description:[/bold cyan] {record.description}")
                self.app.call_from_thread(self.log_widget.write, f"[bold cyan]Sequence Length:[/bold cyan] {len(record.seq)}")
                
                sequence_str = str(record.seq)
                chunks = list(chunk_sequence(sequence_str, chunk_size))
                
                self.app.call_from_thread(self.log_widget.write, f"Total Chunks: {len(chunks)}")
                self.app.call_from_thread(self.log_widget.write, "First 2 chunks:")
                
                for i, chunk in enumerate(chunks):
                    # Simulate processing time so the user can visually see the progress bar advance
                    time.sleep(0.3)
                    
                    if i < 2:
                        self.app.call_from_thread(self.log_widget.write, f"  Chunk {i+1} (length {len(chunk)}): {chunk}")
                    
                    # Advance progress bar
                    self.app.call_from_thread(self.progress.advance, 1)
                    
                self.app.call_from_thread(self.log_widget.write, "-" * 40)
            
            self.app.call_from_thread(self.log_widget.write, "\n[bold yellow]Processing Complete! Press 'q' to quit.[/bold yellow]")
                
        except Exception as e:
            self.app.call_from_thread(self.log_widget.write, f"[bold red]Error parsing FASTA file:[/bold red] {e}")

if __name__ == "__main__":
    app = FastaParserApp()
    app.run()
