import argparse
import time
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, ProgressBar, DataTable, Static, Log
from textual import work

import geneweaver_final

class GeneWeaverApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #dashboard {
        width: 35%;
        border: solid green;
        padding: 1 2;
    }
    #results {
        width: 65%;
        border: solid blue;
        padding: 1 2;
    }
    #status_label {
        margin-bottom: 1;
        text-style: bold;
    }
    ProgressBar {
        margin-bottom: 2;
    }
    Log {
        height: 1fr;
        border: solid ascii #333;
    }
    """
    
    BINDINGS = [("q", "quit", "Quit Application")]

    def __init__(self, fasta_path, sgrna, max_mismatches):
        super().__init__()
        self.fasta_path = fasta_path
        self.sgrna = sgrna
        self.max_mismatches = max_mismatches

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="dashboard"):
                yield Static("GeneWeaver Dashboard\nWaiting to start...", id="status_label")
                yield ProgressBar(id="progress_bar")
                yield Static("\nPipeline Logs:", classes="section-title")
                yield Log(id="log_panel")
            with Vertical(id="results"):
                yield Static("Off-Target Mismatches Found", classes="section-title")
                yield DataTable(id="results_table")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results_table", DataTable)
        table.add_columns("Index", "Context (23 bp)", "Mismatches", "Severity Score")
        table.zebra_stripes = True
        self.run_pipeline_worker()

    @work(thread=True)
    def run_pipeline_worker(self):
        log = self.query_one("#log_panel", Log)
        status = self.query_one("#status_label", Static)
        pbar = self.query_one("#progress_bar", ProgressBar)
        table = self.query_one("#results_table", DataTable)
        
        self.call_from_thread(log.write_line, f"Starting pipeline on {self.fasta_path}...")
        self.call_from_thread(log.write_line, f"Target: {self.sgrna} | Max Mismatches: {self.max_mismatches}")
        
        for update in geneweaver_final.run_pipeline(self.fasta_path, self.sgrna, self.max_mismatches):
            if update["status"] == "init":
                mode = "[green]GPU (CUDA)[/green]" if update.get("gpu") else "[yellow]CPU Fallback[/yellow]"
                self.call_from_thread(pbar.update, total=update["total_chunks"])
                self.call_from_thread(status.update, f"Pipeline Running ({mode})\nTotal Chunks: {update['total_chunks']}\nDask Dashboard: {update['dashboard']}")
                self.call_from_thread(log.write_line, f"Pipeline initialized. Mode: {'GPU' if update.get('gpu') else 'CPU'}")
            
            elif update["status"] == "progress":
                self.call_from_thread(pbar.update, advance=1)
                self.call_from_thread(log.write_line, f"Processed chunk {update['completed']}/{update['total']}")
                
                # Render results for this chunk
                for res in update["results"]:
                    score = res['score']
                    # Severity coloring based on score
                    color = "red" if score < 50 else ("orange" if score < 75 else "green")
                    fmt_score = f"[{color} bold]{score}[/]"
                    
                    # Highlight mutated base pairs in red
                    target_full = self.sgrna + "AGG"
                    context = res['context']
                    formatted_context = ""
                    for i, char in enumerate(context):
                        if i < len(target_full) and char != target_full[i]:
                            # "N" in PAM is a wildcard, so technically NGG matches anything for first PAM base
                            if i == 20: # 21st base is N in NGG
                                formatted_context += char
                            else:
                                formatted_context += f"[red bold]{char}[/red bold]"
                        else:
                            formatted_context += char

                    self.call_from_thread(
                        table.add_row,
                        f"{res['index']:,}",
                        formatted_context,
                        str(res['mismatches']),
                        fmt_score
                    )
                    
            elif update["status"] == "error":
                self.call_from_thread(log.write_line, f"[red bold]Error:[/red bold] {update['message']}")
                self.call_from_thread(status.update, "Pipeline [red]FAILED[/red]")
                
            elif update["status"] == "done":
                self.call_from_thread(log.write_line, "[green bold]Pipeline completed successfully![/green bold]")
                self.call_from_thread(status.update, "Pipeline [green]COMPLETED[/green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeneWeaver Textual Interface")
    parser.add_argument("--fasta", default="mock_genome.fasta", help="Path to input FASTA file")
    parser.add_argument("--sgrna", default="GAGTCCGAGCAGAAGAAGAA", help="Target sgRNA sequence (20bp)")
    parser.add_argument("--mismatches", type=int, default=4, help="Max mismatches threshold")
    args = parser.parse_args()
    
    app = GeneWeaverApp(args.fasta, args.sgrna, args.mismatches)
    app.run()
