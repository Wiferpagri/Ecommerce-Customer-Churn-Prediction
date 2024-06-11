# Ecommerce Customer Churn Prediction

## Overview
This project aims to predict customer churn from an ecommerce. The main focus is to build a predictive model with accuracy over 70% for both classes. The project leverages various data science techniques including data cleaning, exploratory data analysis (EDA), SMOTE, feature engineering, and machine learning.

### Author
William Pabon. [LinkedIn](linkedin.com/in/william-ferney-pabon-grimaldi-026846236/)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/Wiferpagri/Ecommerce-Customer-Churn-Prediction.git
cd Ecommerce-Customer-Churn-Prediction
pip install -r requirements.txt
```

## Usage

Run the following command to start the analysis:

```bash
python main.py
```

You can also explore individual notebooks in the `notebooks` directory to see the step-by-step data processing and analysis.

## Project Structure

    ├── LICENSE
    ├── tasks.py           <- Invoke with commands like `notebook`.
    ├── README.md          <- The top-level README for developers using this project.
    ├── install.md         <- Detailed instructions to set up this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries.
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures         <- Generated graphics and figures to be used in reporting.
    │
    ├── requirements.txt    <- The requirements file for reproducing the analysis environment.
    │
    ├── .here              <- File that will stop the search if none of the other criteria
    │                         apply when searching head of project.
    │
    ├── setup.py           <- Makes project pip installable (pip install -e .)
    │                         so ecommerce_customer_churn_prediction can be imported.
    │
    └── ecommerce_customer_churn_prediction               <- Source code for use in this project.
        ├── __init__.py    <- Makes ecommerce_customer_churn_prediction a Python module.
        │
        ├── data           <- Scripts to download or generate data.
        │   └── make_dataset.py
        │
        ├── features       <- Scripts to turn raw data into features for modeling.
        │   └── build_features.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions.
        │   ├── predict_model.py
        │   └── train_model.py
        │
        ├── utils          <- Scripts to help with common tasks.
            └── paths.py   <- Helper functions to relative file referencing across project.
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations.
            └── visualize.py

---

## Datasets
The data used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data). The datasets are stored in the data/ directory:

* `raw/` contains the original, unprocessed data.
* `interim/` contains intermediate data that has been transformed.
* `processed/` contains cleaned and preprocessed data ready for analysis.
* `external/` contains any additional data from external sources. Test data was generated using [Mostly AI](https://mostly.ai/blog/how-to-generate-synthetic-data)

## Results
The results of the analysis and model performance are documented in the `reports/` directory. This includes:

* Final model performance metrics (e.g., accuracy, precision, recall, F1 score)
* Visualizations and plots
* Detailed analysis reports

## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch: git checkout -b feature-branch-name.
3. Make your changes and commit them: git commit -m 'Add new feature'.
4. Push to the branch: git push origin feature-branch-name.
5. Create a pull request detailing your changes.

## License
This project is licensed under the MIT.
