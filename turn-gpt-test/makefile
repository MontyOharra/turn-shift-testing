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

.PHONY: init install fix-turngpt clean run

# Create a new conda environment in folder $(ENV_DIR).
# Remove TurnGPT if it exists, clone the repo, and clone the datasets_turntaking repo.   
init: 
	conda create -p $(ENV_DIR) python=3 -y
	rm -rf $(TURN_GPT) && \
	git clone $(TURN_GPT_REPO_LINK) && \
	cd $(TURN_GPT)/ && \
	git clone $(DATASETS_REPO_LINK) && \
	cd ..

# Install the required packages, The first line are root-dependencies.
# All other dependencies are found in $(TURN_GPT)/requirements.txt and datasets_turntaking/requirements.txt
install: init
	conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
	conda run -p $(ENV_DIR) $(PIP) install -r requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -r $(TURN_GPT)/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -r $(TURN_GPT)/$(TURN_GPT_DATASETS)/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -e $(TURN_GPT)/$(TURN_GPT_DATASETS)/
	conda run -p $(ENV_DIR) $(PIP) install -e $(TURN_GPT)/
	sed -i '71s/s\.strip()/[s.strip()]/' $(TURN_GPT)/turngpt/model.py


# Remove the entire environment folder and delete TurnGPT.
clean:
	rm -rf $(TURN_GPT)
	rm -rf $(ENV_DIR)

# Run test.py in the environment. 
run-test:
	rm -rf output.png
	conda run -p $(ENV_DIR) $(PYTHON) -u -m test
