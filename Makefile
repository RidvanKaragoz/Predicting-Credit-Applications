# Makefile

# Variables
VENV_NAME?=.venv
PYTHON=${VENV_NAME}/bin/python3
REQUIREMENTS=requirements.txt

# Phony targets
.PHONY: env clean run lint test

# Create virtual environment and install requirements
env:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Activating virtual environment..."
	. $(VENV_NAME)/bin/activate; 
	@echo "Installing requirements..."
	$(PYTHON) -m pip install -r $(REQUIREMENTS)

freeze:
	pip3 freeze > requirements_dev.txt

# Clean up generated files
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	rm -rf __pycache__
	rm -rf *.pyc

# Run the main application
run: env preprocess features train predict
	@echo "Done running all scripts"

preprocess:
	@echo "Preprocessing data..."
	$(PYTHON) src/preprocess.py

features:
	@echo "Engineering features..."
	$(PYTHON) src/create_features.py

train:
	@echo "Training model..."
	$(PYTHON) src/train.py

predict:
	@echo "Making prediction..."
	$(PYTHON) src/predict.py

# Lint the code using Flake8
lint:
	@echo "Linting code..."
	. $(VENV_NAME)/bin/activate; flake8

# Run tests using Pytest
test:
	@echo "Running tests..."
	. $(VENV_NAME)/bin/activate; pytest