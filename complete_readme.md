# 🏠 House Rent Prediction - Machine Learning Project

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ML Zoomcamp 2024 - Midterm Project**

---

## 📋 Table of Contents

- [Problem Description](#-problem-description)
- [Dataset](#-dataset)
- [Project Architecture](#-project-architecture)
- [Technologies Used](#-technologies-used)
- [Installation Guide](#-installation-guide)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker Setup](#docker-setup)
- [Usage](#-usage)
  - [Training the Model](#1-training-the-model)
  - [Running the Web Service](#2-running-the-web-service)
  - [Making Predictions](#3-making-predictions)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Deployment](#-deployment)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Description

### The Challenge

Finding the right rental price is a complex challenge that affects millions of people in India:

**For Tenants:**
- 🏘️ Difficulty determining if a rental price is fair or inflated
- 💰 Risk of overpaying due to lack of market knowledge
- 📊 Need for data-driven price comparisons across different locations

**For Landlords:**
- 📈 Challenge in setting competitive rental prices
- ⚖️ Risk of underpricing and losing potential income
- 🎯 Uncertainty about property value factors

**For Real Estate Agents:**
- 🤝 Need to provide accurate estimates to clients
- ⏱️ Time-consuming manual price research
- 📍 Difficulty tracking prices across multiple cities

### The Solution

This project provides a **machine learning-powered REST API** that predicts house rental prices based on property characteristics. The model analyzes:

- 🛏️ Property features (BHK, size, bathrooms)
- 📍 Location (6 major Indian cities)
- 🪑 Furnishing status
- 👥 Tenant preferences
- 📞 Contact type

### Real-World Applications

1. **Rental Price Estimation Tool** - Integrate into property listing platforms
2. **Market Analysis Dashboard** - Help real estate agencies analyze pricing trends
3. **Tenant Budget Planner** - Assist tenants in financial planning
4. **Dynamic Pricing System** - Enable landlords to optimize rental income

---

## 📊 Dataset

**Source:** [Kaggle - House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)

### Dataset Statistics

- **Total Records:** 4,700+ properties
- **Time Period:** 2022-2023
- **Geographic Coverage:** 6 major Indian cities
- **Features:** 12 attributes

### Features Description

| Feature | Type | Description | Example Values |
|---------|------|-------------|----------------|
| **BHK** | Numerical | Number of Bedrooms, Hall, Kitchen | 1, 2, 3, 4+ |
| **Size** | Numerical | Property size in square feet | 500 - 8000 sq ft |
| **City** | Categorical | Location of property | Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata |
| **Furnishing Status** | Categorical | Level of furnishing | Furnished, Semi-Furnished, Unfurnished |
| **Tenant Preferred** | Categorical | Preferred tenant type | Bachelors, Family, Bachelors/Family |
| **Bathroom** | Numerical | Number of bathrooms | 1, 2, 3, 4+ |
| **Area Type** | Categorical | Type of area measurement | Super Area, Carpet Area, Built Area |
| **Point of Contact** | Categorical | Contact person | Contact Owner, Contact Agent |
| **Rent** | Numerical (Target) | Monthly rent in INR (₹) | 10,000 - 300,000+ |

### Data Preprocessing

- Missing values handled (totalcharges filled with 0)
- Log transformation applied to target variable (rent) for better model performance
- Categorical variables encoded using DictVectorizer
- Train/Validation/Test split: 60%/20%/20%

---

## 🏗️ Project Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client    │─────▶│  Flask API   │─────▶│    Model    │
│  (User/App) │      │  (predict.py)│      │ (model.bin) │
└─────────────┘      └──────────────┘      └─────────────┘
       │                     │                      │
       │                     │                      │
       ▼                     ▼                      ▼
  JSON Request          Processing            Prediction
  {property data}    DictVectorizer        Linear Regression
                     Transformation        Returns: rent value
```

### Workflow

1. **Training Phase** (`train.py`):
   - Load data from Kaggle
   - Preprocess and clean data
   - Train Linear Regression model
   - Save model and vectorizer to `model.bin`

2. **Prediction Phase** (`predict.py`):
   - Load trained model
   - Receive property data via POST request
   - Transform input using DictVectorizer
   - Return predicted rent price

3. **Deployment**:
   - Containerized with Docker
   - Can be deployed to AWS, Railway, or any cloud platform
   - Scalable REST API

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.11 | Core programming language |
| **scikit-learn** | 1.4.0 | Machine learning (Linear Regression, DictVectorizer) |
| **Flask** | 3.0.0 | Web framework for REST API |
| **Pandas** | 2.2.0 | Data manipulation and analysis |
| **NumPy** | 1.26.4 | Numerical computations |
| **Gunicorn** | 21.2.0 | Production WSGI server |
| **Docker** | Latest | Containerization |
| **Pipenv** | Latest | Dependency management |
| **KaggleHub** | 0.2.5 | Dataset download |

---

## 📦 Installation Guide

### Prerequisites

Before you begin, ensure you have the following installed:

- ✅ **Python 3.11** or higher ([Download](https://www.python.org/downloads/))
- ✅ **pip** (comes with Python)
- ✅ **pipenv** for dependency management
- ✅ **Docker** (optional, for containerization) ([Download](https://www.docker.com/))
- ✅ **Git** (for cloning the repository)

### Local Setup

#### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd house-rent-midterm-project
```

#### Step 2: Install Pipenv

```bash
pip install pipenv
```

#### Step 3: Install Dependencies

```bash
# Install all project dependencies
pipenv install

# For development (includes Jupyter, matplotlib, seaborn)
pipenv install --dev
```

This will:
- Create a virtual environment
- Install all packages from `Pipfile`
- Generate `Pipfile.lock` for reproducibility

#### Step 4: Activate Virtual Environment

```bash
pipenv shell
```

Your terminal prompt should change to indicate you're in the virtual environment:
```
(house-rent-midterm-project) user@machine:~/project$
```

#### Step 5: Verify Installation

```bash
# Check Python version
python --version
# Output: Python 3.11.x

# Check installed packages
pip list
```

---

### Docker Setup

Docker provides a consistent environment across different systems.

#### Step 1: Install Docker

- **Windows/Mac**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux**: 
  ```bash
  curl -fsSL https://get.docker.com -o get-docker.sh
  sudo sh get-docker.sh
  ```

#### Step 2: Verify Docker Installation

```bash
docker --version
# Output: Docker version 24.x.x

docker run hello-world
# Should download and run a test container
```

#### Step 3: Build Docker Image

Navigate to the project directory and build the image:

```bash
# Build the image (this may take 2-5 minutes)
docker build -t house-rent-prediction .

# Verify the image was created
docker images
```

You should see:
```
REPOSITORY               TAG       IMAGE ID       CREATED         SIZE
house-rent-prediction    latest    abc123def456   2 minutes ago   500MB
```

#### Step 4: Run Docker Container

```bash
# Run the container
docker run -it --rm -p 9696:9696 house-rent-prediction
```

**Explanatio