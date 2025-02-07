# Folder-based conda environment, created in .venv
ENV_DIR = .venv
PYTHON  = python
PIP     = pip

.PHONY: init install clean run

# Create a new conda environment in folder $(ENV_DIR).
init: clean
	conda create -p $(ENV_DIR) python=3.9 -y

# Install the required packages
install: init
	conda run -p $(ENV_DIR) conda install nvidia::libcublas -y
	conda run -p $(ENV_DIR) conda install cudatoolkit cudnn -y
	conda run -p $(ENV_DIR) $(PIP) install -r requirements.txt

# Remove the entire environment folder.
clean:
	rm -rf $(ENV_DIR)


run-test:
	conda run --live-stream -p $(ENV_DIR) $(PYTHON) -u -m test