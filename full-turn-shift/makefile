OSTYPE :=
ifeq ($(OS),Windows_NT)
	OSTYPE := Windows
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		OSTYPE := Linux
	endif
	ifeq ($(UNAME_S),Darwin)
		OSTYPE := MacOS
	endif
endif

PYAUDIO_LIBS := 
ifeq ($(OSTYPE),Windows)
	PYAUDIO_INSTALL_COMMAND := python -m pip install pyaudio
	PYTORCH_INSTALL_COMMAND := -c pytorch pytorch torchvision torchaudio cpuonly -y
	CUBLAS_INSTALL_COMMAND := nvidia::libcublas -y
	CUDNN_INSTALL_COMMAND := -c nvidia cudatoolkit cudnn -y
else
	PYAUDIO_INSTALL_COMMAND := -c conda-forge pyaudio alsa-lib alsa-plugins -y
	PYTORCH_INSTALL_COMMAND := pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
	CUBLAS_INSTALL_COMMAND := nvidia::libcublas -y
	CUDNN_INSTALL_COMMAND := cudatoolkit cudnn -y
endif


# Folder-based conda environment, created in .venv
# or whatever you name the .venv dir.
ENV_DIR = .venv
# Python and pip commands. 
# Probably should implement cross-OS functionality
PYTHON  = python
PIP     = pip

# Repo names and git links
TURN_GPT = TurnGPT
TURN_GPT_REPO_LINK = https://github.com/ErikEkstedt/TurnGPT.git
DATASETS = datasets_turntaking
DATASETS_REPO_LINK = https://github.com/ErikEkstedt/datasets_turntaking.git
VAP = VoiceActivityProjection
VAP_REPO_LINK = https://github.com/ErikEkstedt/VoiceActivityProjection.git

.PHONY: init install fix-turngpt clean run

# Create a new conda environment in folder $(ENV_DIR).
# Remove TurnGPT if it exists, clone the repo, and clone the datasets_turntaking repo. 
# Similarly for VAP.  
init: 
	conda create -p $(ENV_DIR) python=3.9 -y
	rm -rf $(VAP) && \
	git clone $(VAP_REPO_LINK)
	rm -rf $(TURN_GPT) && \
	git clone $(TURN_GPT_REPO_LINK) && \
	cd $(TURN_GPT)/ && \
	git clone $(DATASETS_REPO_LINK) && \
	cd ..

# Install the required packages, The first line are root-dependencies.
# All other dependencies are found in $(TURN_GPT)/requirements.txt and datasets_turntaking/requirements.txt
install: init
	conda run -p $(ENV_DIR) conda install $(PYAUDIO_INSTALL_COMMAND)
	conda run -p $(ENV_DIR) conda install $(PYTORCH_INSTALL_COMMAND)
	conda run -p $(ENV_DIR) conda install $(CUBLAS_INSTALL_COMMAND)
	conda run -p $(ENV_DIR) conda install $(CUDNN_INSTALL_COMMAND)
	conda run -p $(ENV_DIR) $(PIP) install -r requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -r $(TURN_GPT)/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -r $(TURN_GPT)/$(TURN_GPT_DATASETS)/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -e $(TURN_GPT)/$(TURN_GPT_DATASETS)/
	conda run -p $(ENV_DIR) $(PIP) install -e $(TURN_GPT)/
	conda run -p $(ENV_DIR) $(PIP) install -r $(VAP)/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -e $(VAP)/

	sed -i '71s/s\.strip()/[s.strip()]/' $(TURN_GPT)/turngpt/model.py

# Remove the entire environment folder and delete TurnGPT.
clean:
	rm -rf $(VAP)
	rm -rf $(TURN_GPT)
	rm -rf $(ENV_DIR)

# Run test.py in the environment. 

whisper-transcribe:
	conda run --live-stream -p $(ENV_DIR) $(PYTHON) -u -m applications.transcribe_audio

whisper-record:
	conda run --live-stream -p $(ENV_DIR) $(PYTHON) -u -m applications.record_audio

get-audio-devices:
	conda run -p $(ENV_DIR) $(PYTHON) -u -m applications.get_audio_devices

turngpt-test:
	rm -rf output.png
	conda run -p $(ENV_DIR) $(PYTHON) -u -m test

turngpt-with-whisper:
	conda run --live-stream -p $(ENV_DIR) $(PYTHON) -u -m applications.turngpt_with_whisper

turngpt-vap-with-whisper:
	conda run --live-stream -p $(ENV_DIR) $(PYTHON) -u -m applications.turngpt_vap_with_whisper
