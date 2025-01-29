ENV_DIR = .venv

ifeq ($(OS),Windows_NT)
	PIP = $(ENV_DIR)\Scripts\pip.exe
	PYTHON = python
else
	PIP = $(ENV_DIR)/bin/pip
	PYTHON = python3
endif

.PHONY:

init:
	$(PYTHON) -m venv $(ENV_DIR)

install: init
	$(PIP) install -r requirements.txt

clean:
	rm -rf $(ENV_DIR)