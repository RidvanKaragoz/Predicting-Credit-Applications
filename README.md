# Predicting-Credit-Applications
# ABN AMRO Case Study

This is a case study project for predicting future credit applications from monthly client account data. The project was developed using Python version 3.10.0.


## Getting Started

To get started, please follow these instructions:

1. Clone the repository to your local machine
2. Navigate to the project directory
3. Create a virtual environment and install the required packages:

```bash
make env
```

4. Run the main application:

```bash
make run
```

## Makefile

The project uses a Makefile to automate common tasks. The following targets are available:

env: create a virtual environment and install the required packages.

run: run the main application.

preprocess: preprocess the data.

features: engineer features.

train: train a machine learning model.

predict: make predictions.

clean: remove generated files.

The run command will execute several scripts in the src directory that will preprocess the data, engineer features, train a model, and make predictions. The output will be saved in the appropriate projects folders.

## Contributors
This project was developed by Ridvan Karagoz. If you have any questions or suggestions, please feel free to contact me. Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

License-free for employees of ABN AMRO