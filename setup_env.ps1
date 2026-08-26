Write-Host "Setting up Python virtual environment..."
python -m venv venv
.\venv\Scripts\Activate.ps1

Write-Host "Installing pip dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Checking Numba CUDA availability..."
python -c "from numba import cuda; print('CUDA available:', cuda.is_available())"

Write-Host "Environment setup complete. To activate the environment in the future, run '.\venv\Scripts\Activate.ps1'."
