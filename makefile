# Folder-based conda environment, created in .venv
ENV_DIR = .venv
PYTHON  = python
PIP     = pip

.PHONY: init install clean run

# 1. Create a new conda environment in folder ".venv" 
#    and install a specific Python version (e.g., 3.9).
init:
	conda create -p $(ENV_DIR) python=3.9 -y

# 2. Install the required packages using pip inside that conda environment.
install: init
	conda run -p $(ENV_DIR) conda install cudatoolkit cudnn -y
	conda run -p $(ENV_DIR) $(PIP) install -r requirements.txt

# 3. Remove the entire environment folder.
clean:
	rm -rf $(ENV_DIR)

# 4. Run your Python module "test.py" (or "test" as a module) inside the conda environment.
run:
	conda run -p $(ENV_DIR) $(PYTHON) -u -m test