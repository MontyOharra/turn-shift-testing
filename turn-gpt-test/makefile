# Folder-based conda environment, created in .venv
# or whatever you name the .venv dir.
ENV_DIR = .venv
# Python and pip commands. 
# Probably should implement cross-OS functionality
PYTHON  = python
PIP     = pip

.PHONY: init install clean run

# Create a new conda environment in folder $(ENV_DIR).
# Remove TurnGPT if it exists, clone the repo, and clone the datasets_turntaking repo.   
init: 
	conda create -p $(ENV_DIR) python=3 -y
	rm -rf TurnGPT && \
	git clone https://github.com/ErikEkstedt/TurnGPT.git && \
	cd TurnGPT/ && \
	git clone https://github.com/ErikEkstedt/datasets_turntaking.git && \
	cd ..

# Install the required packages, The first line are root-dependencies.
# All other dependencies are found in TurnGPT/requirements.txt and datasets_turntaking/requirements.txt
install: init
	conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
	conda run -p $(ENV_DIR) $(PIP) install -r TurnGPT/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -r TurnGPT/datasets_turntaking/requirements.txt
	conda run -p $(ENV_DIR) $(PIP) install -e TurnGPT/datasets_turntaking/
	conda run -p $(ENV_DIR) $(PIP) install -e TurnGPT/

fix-turngpt:
	sed -i '71s/s\.strip()/[s.strip()]/' TurnGPT/turngpt/model.py


# Remove the entire environment folder and delete TurnGPT.
clean:
	rm -rf TurnGPT
	rm -rf $(ENV_DIR)

# Run test.py in the environment. 
run-test:
	conda run -p $(ENV_DIR) $(PYTHON) -u -m test
