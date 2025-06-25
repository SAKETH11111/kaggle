# Business Traveler Flight Selection Analysis

This repository contains a comprehensive analysis of a flight booking dataset to understand the key drivers of business travelers' choices. The project aims to uncover actionable insights that can be used to build a high-performing predictive model.

## Project Structure

```
.
├── data/
│   ├── train.parquet
│   └── jsons_raw.tar.kaggle
├── processed/
│   ├── jsons_raw/
│   │   └── raw/
│   └── json_features.parquet
├── docs/
│   └── day1_2_report.md
├── json_extractor/
│   ├── json_extractor.py
│   └── README.md
├── final_analysis.py
├── json_feature_extractor.py
├── price_policy_analysis.py
├── group_analysis.py
├── requirements.txt
└── README.md
```

### Key Directories

*   **`data/`**: Contains the raw, immutable input data.
    *   `train.parquet`: The main dataset of flight search results.
    *   `jsons_raw.tar.kaggle`: An archive containing detailed JSON data for each search.
*   **`processed/`**: Contains all processed or intermediate data files.
    *   `jsons_raw/`: The destination for the extracted JSON files.
    *   `json_features.parquet`: A dataset of features extracted from the JSON files.
*   **`docs/`**: Contains project documentation and reports.
*   **`json_extractor/`**: A standalone module for parsing the raw JSON data.

### Key Files

*   **`group_analysis.py`**: The initial exploratory data analysis script.
*   **`price_policy_analysis.py`**: A script that performs a deep-dive into price and policy compliance.
*   **`json_feature_extractor.py`**: A script to extract detailed features from the raw JSON data.
*   **`final_analysis.py`**: The final, comprehensive analysis script that merges all data sources and generates the key insights.
*   **`requirements.txt`**: A list of all Python dependencies required to run the project.
*   **`README.md`**: This file, providing an overview of the project.

## Setup and Execution

### 1. Environment Setup

It is recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Before running the analysis, the raw JSON data must be extracted.

```bash
# Create the destination directory
mkdir -p processed/jsons_raw

# Extract the JSON files
tar -xf data/jsons_raw.tar.kaggle -C processed/jsons_raw/
```

### 3. Running the Analyses

The analyses are designed to be run in a specific order, as each script builds upon the previous one.

#### Step 1: Initial Group Analysis (Optional)

This script performs the initial exploratory data analysis.

```bash
python3 group_analysis.py
```

#### Step 2: Price and Policy Deep-Dive (Optional)

This script provides a more detailed look at pricing and policy compliance.

```bash
python3 price_policy_analysis.py
```

#### Step 3: JSON Feature Extraction (Required)

This is a critical step that extracts detailed features from the raw JSON data. It is parallelized to run on all available CPU cores.

```bash
python3 json_feature_extractor.py
```

#### Step 4: Final Comprehensive Analysis (Required)

This script merges all data sources and performs the final, in-depth analysis to generate the key insights that will inform the modeling phase.

```bash
python3 final_analysis.py
```

## Key Insights

The comprehensive analysis has revealed several powerful, actionable insights that are critical for building a high-performing predictive model:

*   **Positional Bias**: A flight's rank in the original search results is a massively predictive feature.
*   **Flexibility is Non-Negotiable**: Business travelers almost exclusively select fares with zero cancellation fees.
*   **Policy Compliance is Dominant**: Travelers are highly likely to choose a policy-compliant flight when one is available.
*   **Price Sensitivity is Contextual**: Travelers are willing to pay a significant premium for convenience, especially for long-haul flights and if they are VIPs.
*   **The "Ideal" Flight**: The data shows a clear preference for direct, policy-compliant, and fully refundable flights during standard business hours.