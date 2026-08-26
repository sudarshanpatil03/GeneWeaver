import os
try:
    os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64")
except AttributeError:
    pass # Not on Windows or Python >= 3.8

from numba import cuda
import time
import threading
import functools

class VRAMMonitor:
    """
    A background threaded monitor to track GPU VRAM usage.
    Helps identify memory leaks or CUDA_OUT_OF_MEMORY risks during large transfers.
    """
    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False
        self.thread = None
        self.peak_memory_used = 0
        self.memory_history = []

    def _monitor_loop(self):
        try:
            if not cuda.is_available():
                return

            while self.running:
                ctx = cuda.current_context()
                free_mem, total_mem = ctx.get_memory_info()
                used_mem = total_mem - free_mem
                
                self.memory_history.append(used_mem)
                if used_mem > self.peak_memory_used:
                    self.peak_memory_used = used_mem
                
                time.sleep(self.interval)
        except Exception as e:
            print(f"VRAM Monitor Error: {e}")

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        return self.peak_memory_used

def track_memory(func):
    """
    Decorator to automatically track peak VRAM usage around a function call.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not cuda.is_available():
            print("CUDA not available. Executing without VRAM monitoring.")
            return func(*args, **kwargs)
            
        monitor = VRAMMonitor(interval=0.01) # Poll quickly for fast kernels
        print(f"\n[VRAM Monitor] Starting memory tracking for '{func.__name__}'...")
        monitor.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            peak_bytes = monitor.stop()
            print(f"[VRAM Monitor] Finished '{func.__name__}'. Peak VRAM Used: {peak_bytes / (1024**2):.2f} MB")
            
        return result
    return wrapper

if __name__ == "__main__":
    print("--- Testing Kernel Memory Transfer Patterns ---")
    
    @track_memory
    def test_host_to_device_transfer():
        if not cuda.is_available():
            print("Skipping transfer test: No GPU available on this system.")
            return
            
        import numpy as np
        
        # Allocate 100MB array on host (RAM)
        print("Allocating 100MB array on Host (RAM)...")
        data = np.ones(25_000_000, dtype=np.float32) 
        
        print("Transferring Host -> Device (VRAM)...")
        # This is where VRAM usage should spike
        d_data = cuda.to_device(data)
        
        # Simulate some kernel compute time where data sits in VRAM
        time.sleep(0.5)
        
        print("Transferring Device -> Host (RAM)...")
        h_data = d_data.copy_to_host()
        
        # Explicitly delete the device array to free VRAM
        print("Freeing Device Memory...")
        del d_data
        time.sleep(0.1) # Give monitor a moment to catch the drop

    test_host_to_device_transfer()
