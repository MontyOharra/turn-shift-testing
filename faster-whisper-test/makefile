PYAUDIO_INSTALL_COMMAND := conda install pyaudio alsa-lib alsa-plugins -y
CUBLAS_INSTALL_COMMAND := conda install nvidia::libcublas -y
CUDNN_INSTALL_COMMAND := conda install cudatoolkit cudnn -y

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
	conda run -p $(ENV_DIR) $(PYAUDIO_INSTALL_COMMAND)	
	conda run -p $(ENV_DIR) $(CUBLAS_INSTALL_COMMAND)
	conda run -p $(ENV_DIR) $(CUDNN_INSTALL_COMMAND)
	conda run -p $(ENV_DIR) $(PIP) install -r requirements.txt

# Remove the entire environment folder.
clean:
	rm -rf $(ENV_DIR)

transcribe:
	conda run --live-stream -p $(ENV_DIR) $(PYTHON) -u -m applications.transcribe_audio

record:
	conda run --live-stream -p $(ENV_DIR) $(PYTHON) -u -m applications.record_audio

get-audio-devices:
	conda run -p $(ENV_DIR) $(PYTHON) -u -m applications.get_audio_devices
