# Set PYTHON variable according to OS
ifeq ($(OS),Windows_NT)
	PYTHON=python
else
	PYTHON=python3
endif

install:
	$(PYTHON) -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -r requirements-optional.txt
	pip install -e .
