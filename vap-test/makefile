# Folder-based conda environment, created in .venv
# or whatever you name the .venv dir.
ENV_DIR = .venv
# Python and pip commands. 
# Probably should implement cross-OS functionality
PYTHON  = python
PIP     = pip
# Name of VAP repo
VAP_REPO_NAME = VoiceActivityProjection
VAP_REPO_LINK = https://github.com/ErikEkstedt/VoiceActivityProjection.git

.PHONY: init install clean run

# Create a new conda environment in folder $(ENV_DIR).
# Remove TurnGPT if it exists, clone the repo, and clone the datasets_turntaking repo.   
init: 
	conda create -p $(ENV_DIR) python=3 -y
	rm -rf $(VAP_REPO_NAME) && \
	git clone $(VAP_REPO_LINK)

# Install the required packages, The first line are root-dependencies.
# All other dependencies are found in TurnGPT/requirements.txt and datasets_turntaking/requirements.txt
install: init
	conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -y
	conda run -p $(ENV_DIR) $(PIP) install -r requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -r $(VAP_REPO_NAME)/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -e $(VAP_REPO_NAME)/

# Remove the entire environment folder and delete TurnGPT.
clean:
	rm -rf $(VAP_REPO_NAME)
	rm -rf $(ENV_DIR)

temp:
	conda run -p $(ENV_DIR) $(PIP) install -r requirements.txt

# Run test.py in the environment. 
run-test:
	conda run -p $(ENV_DIR) $(PYTHON) -u VoiceActivityProjection/run.py \
	-a VoiceActivityProjection/example/student_long_female_en-US-Wavenet-G.wav \
	-sd VoiceActivityProjection/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt \
	-f test.json \
	--plot