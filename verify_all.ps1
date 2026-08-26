Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   GENEWEAVER FINAL - VERIFICATION RUN   " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Environment Verification
Write-Host "`n[1/4] Verifying Python Environment & Dependencies..." -ForegroundColor Yellow
.\venv\Scripts\python.exe -c "import numba, dask, numpy, pandas; print('SUCCESS: All core dependencies loaded perfectly!')"
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to load dependencies." -ForegroundColor Red; exit }

# 2. Dataset Generation (Week 1)
Write-Host "`n[2/4] Verifying Dataset Generation (Week 1)..." -ForegroundColor Yellow
.\venv\Scripts\python.exe generate_datasets.py
if ($LASTEXITCODE -ne 0) { Write-Host "Failed to generate datasets." -ForegroundColor Red; exit }

# 3. Benchmark Framework (Week 1)
Write-Host "`n[3/4] Running Week 1 Benchmarks (CPU vs Numba vs Dask)..." -ForegroundColor Yellow
.\venv\Scripts\python.exe benchmark.py --dataset small
if ($LASTEXITCODE -ne 0) { Write-Host "Failed benchmark tests." -ForegroundColor Red; exit }

# 4. Dask Scaffolding & Memory Monitor (Week 2)
Write-Host "`n[4/4] Verifying Week 2 Scaffolding (VRAM Monitor & Dask Pipeline)..." -ForegroundColor Yellow
Write-Host "-> Testing Memory Monitor:" -ForegroundColor Gray
.\venv\Scripts\python.exe week2/memory_monitor.py
Write-Host "-> Testing Dask Distributed Integration:" -ForegroundColor Gray
.\venv\Scripts\python.exe week2/dask_scaffold.py
if ($LASTEXITCODE -ne 0) { Write-Host "Failed Dask scaffolding." -ForegroundColor Red; exit }

Write-Host "`n=========================================" -ForegroundColor Green
Write-Host " ALL SYSTEMS VERIFIED AND WORKING PERFECTLY! " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
