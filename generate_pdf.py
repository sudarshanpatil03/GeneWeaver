import markdown
import os
import subprocess
import argparse
import time

def convert_md_to_pdf(md_file, pdf_file):
    print(f"Converting {md_file} to {pdf_file}...")
    
    # 1. Read Markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()
        
    # 2. Convert to HTML
    html_body = markdown.markdown(md_text, extensions=['fenced_code', 'tables'])
    
    # Add GitHub-like styling for a professional look
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ 
                font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji";
                line-height: 1.6; 
                color: #24292f;
                padding: 40px;
                max-width: 900px;
                margin: 0 auto;
            }}
            h1, h2, h3, h4 {{ 
                margin-top: 24px;
                margin-bottom: 16px;
                font-weight: 600;
                line-height: 1.25;
                border-bottom: 1px solid #hsla(210,18%,87%,1);
                padding-bottom: 0.3em;
            }}
            h1 {{ font-size: 2em; }}
            h2 {{ font-size: 1.5em; }}
            code {{ 
                background-color: rgba(175,184,193,0.2); 
                padding: 0.2em 0.4em; 
                border-radius: 6px; 
                font-family: ui-monospace,SFMono-Regular,SF Mono,Menlo,Consolas,Liberation Mono,monospace;
                font-size: 85%;
            }}
            pre {{ 
                background-color: #f6f8fa; 
                padding: 16px; 
                border-radius: 6px; 
                overflow: auto; 
                font-family: ui-monospace,SFMono-Regular,SF Mono,Menlo,Consolas,Liberation Mono,monospace;
                font-size: 85%;
                line-height: 1.45;
            }}
            pre code {{ background-color: transparent; padding: 0; }}
            table {{ 
                border-collapse: collapse; 
                width: 100%; 
                margin-bottom: 20px; 
            }}
            th, td {{ 
                border: 1px solid #d0d7de; 
                padding: 6px 13px; 
            }}
            th {{ font-weight: 600; background-color: #f6f8fa; }}
            tr:nth-child(2n) {{ background-color: #f6f8fa; }}
            hr {{
                height: 0.25em;
                padding: 0;
                margin: 24px 0;
                background-color: #d0d7de;
                border: 0;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """
    
    html_file = md_file.replace('.md', '.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    # 3. Use Edge Headless to print to PDF
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
    ]
    
    edge_exe = None
    for p in edge_paths:
        if os.path.exists(p):
            edge_exe = p
            break
            
    if not edge_exe:
        print("Microsoft Edge not found. Cannot generate PDF.")
        return
        
    abs_html = os.path.abspath(html_file)
    abs_pdf = os.path.abspath(pdf_file)
    
    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={abs_pdf}",
        abs_html
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Wait a moment to ensure file lock is released
    time.sleep(1)
    
    # Clean up HTML
    try:
        os.remove(html_file)
    except Exception as e:
        print(f"Warning: Could not remove temporary html file: {e}")
        
    print(f"Successfully generated {pdf_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=str, required=True, help="Input Markdown file")
    parser.add_argument("--pdf", type=str, required=True, help="Output PDF file")
    args = parser.parse_args()
    
    convert_md_to_pdf(args.md, args.pdf)
