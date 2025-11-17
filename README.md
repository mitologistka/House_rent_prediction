# House Rent Prediction - Machine Learning Project

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ML Zoomcamp 2024 - Midterm Project**

---

## Table of Contents

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

## Problem Description

### The Challenge

Finding the right rental price is a complex challenge that affects millions of people in India:

**For Tenants:**
- Difficulty determining if a rental price is fair or inflated
- Risk of overpaying due to lack of market knowledge
- Need for data-driven price comparisons across different locations

**For Landlords:**
- Challenge in setting competitive rental prices
- Risk of underpricing and losing potential income
- Uncertainty about property value factors

**For Real Estate Agents:**
- Need to provide accurate estimates to clients
- Time-consuming manual price research
- Difficulty tracking prices across multiple cities

### The Solution

This project provides a **machine learning-powered REST API** that predicts house rental prices based on property characteristics. The model analyzes:

- Property features (BHK, size, bathrooms)
- Location (6 major Indian cities)
- Furnishing status
- Tenant preferences
- Contact type

### Real-World Applications

1. **Rental Price Estimation Tool** - Integrate into property listing platforms
2. **Market Analysis Dashboard** - Help real estate agencies analyze pricing trends
3. **Tenant Budget Planner** - Assist tenants in financial planning
4. **Dynamic Pricing System** - Enable landlords to optimize rental income

---

## Dataset

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

## Technologies Used

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

## Installation Guide

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11** or higher ([Download](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **pipenv** for dependency management
- **Docker** (optional, for containerization) ([Download](https://www.docker.com/))
- **Git** (for cloning the repository)

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

**Explanation of flags:**
- `-it`: Interactive terminal
- `--rm`: Automatically remove container when it stops
- `-p 9696:9696`: Map port 9696 from container to host

The service will be available at `http://localhost:9696`

#### Step 5: Run Container in Background (Detached Mode)

```bash
# Run in background
docker run -d -p 9696:9696 --name rent-predictor house-rent-prediction

# Check running containers
docker ps

# View logs
docker logs rent-predictor

# Stop container
docker stop rent-predictor

# Remove container
docker rm rent-predictor
```

---

## Usage

### 1. Training the Model

If you want to retrain the model with fresh data:

```bash
# Activate environment
pipenv shell

# Run training script
python train.py
```

**Expected output:**
```
Loading data from Kaggle...
Dataset loaded: (4746, 12)
Preparing data...
Train size: 2847
Validation size: 950
Test size: 949
Vectorizing features...
Training the final model...
Test RMSE: 0.2734
Test R²: 0.7421
Saving model to model.bin...
Model saved successfully!
```

This creates/updates `model.bin` (approximately 2-3 KB).

### 2. Running the Web Service

#### Option A: Local (using Flask development server)

```bash
# Activate environment
pipenv shell

# Run the service
python predict.py
```

**Output:**
```
 * Serving Flask app 'rent-prediction'
 * Debug mode: on
 * Running on http://127.0.0.1:9696
Press CTRL+C to quit
```

**Note:** Keep this terminal open while the service is running.

#### Option B: Docker (recommended for production)

```bash
# Build and run
docker build -t house-rent-prediction .
docker run -it --rm -p 9696:9696 house-rent-prediction
```

#### Option C: Production with Gunicorn

```bash
pipenv shell
gunicorn --bind 0.0.0.0:9696 predict:app
```

### 3. Making Predictions

#### Option A: Using the Test Script

**In a new terminal** (keep the service running in the first terminal):

```bash
# Activate environment
pipenv shell

# Run test
python test_predict.py
```

**Expected output:**
```
Sending request to: http://localhost:9696/predict
Property data: {'bhk': 2, 'size': 1100, ...}

Response:
  Predicted rent: ₹40,954.34
  Predicted rent (log): 10.6202
```

#### Option B: Using curl

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bhk": 2,
    "size": 1100,
    "bathroom": 2,
    "area_type": "super_area",
    "city": "mumbai",
    "furnishing_status": "semi-furnished",
    "tenant_preferred": "bachelors/family",
    "point_of_contact": "contact_owner"
  }'
```

#### Option C: Using Python requests

```python
import requests

url = 'http://localhost:9696/predict'

property_data = {
    "bhk": 2,
    "size": 1100,
    "bathroom": 2,
    "area_type": "super_area",
    "city": "mumbai",
    "furnishing_status": "semi-furnished",
    "tenant_preferred": "bachelors/family",
    "point_of_contact": "contact_owner"
}

response = requests.post(url, json=property_data)
result = response.json()

print(f"Predicted Rent: ₹{result['predicted_rent']:,.2f}")
```

---

## API Documentation

### Base URL

```
http://localhost:9696
```

### Endpoints

#### 1. Health Check

**GET** `/health`

Check if the service is running.

**Response:**
```json
{
  "status": "healthy"
}
```

**Example:**
```bash
curl http://localhost:9696/health
```

---

#### 2. Predict Rent

**POST** `/predict`

Predict house rent based on property features.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "bhk": 2,
  "size": 1100,
  "bathroom": 2,
  "area_type": "super_area",
  "city": "mumbai",
  "furnishing_status": "semi-furnished",
  "tenant_preferred": "bachelors/family",
  "point_of_contact": "contact_owner"
}
```

**Field Descriptions:**

| Field | Type | Required | Valid Values | Description |
|-------|------|----------|--------------|-------------|
| `bhk` | integer | Yes | 1-6 | Number of Bedrooms, Hall, Kitchen |
| `size` | integer | Yes | 100-10000 | Property size in square feet |
| `bathroom` | integer | Yes | 1-6 | Number of bathrooms |
| `area_type` | string | Yes | `"super_area"`, `"carpet_area"`, `"built_area"` | Type of area measurement |
| `city` | string | Yes | `"mumbai"`, `"delhi"`, `"bangalore"`, `"hyderabad"`, `"chennai"`, `"kolkata"` | City location |
| `furnishing_status` | string | Yes | `"furnished"`, `"semi-furnished"`, `"unfurnished"` | Furnishing level |
| `tenant_preferred` | string | Yes | `"bachelors"`, `"family"`, `"bachelors/family"` | Preferred tenant type |
| `point_of_contact` | string | Yes | `"contact_owner"`, `"contact_agent"` | Contact type |

**Response:**
```json
{
  "predicted_rent": 40954.34,
  "predicted_rent_log": 10.6202
}
```

**Status Codes:**
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid input data
- `500 Internal Server Error`: Server error

**Example:**
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "bhk": 3,
    "size": 1500,
    "bathroom": 2,
    "area_type": "super_area",
    "city": "bangalore",
    "furnishing_status": "furnished",
    "tenant_preferred": "family",
    "point_of_contact": "contact_owner"
  }'
```

---

## Model Performance

### Metrics on Test Set

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 0.2734 | Average error on log scale |
| **R² Score** | 0.7421 | Model explains 74.21% of variance |
| **MAE** | ~0.21 | Median absolute error |

### Model Selection Process

We evaluated multiple models:

| Model | RMSE | R² Score | Training Time |
|-------|------|----------|---------------|
| **Linear Regression** | 0.2734 | 0.7421 | < 1s |
| Ridge Regression | 0.2758 | 0.7389 | < 1s |
| Decision Tree | 0.3124 | 0.6821 | 2s |
| Random Forest | 0.2891 | 0.7245 | 15s |

**Linear Regression** was chosen for its:
- Best RMSE and R² scores
- Fast training and prediction
- Interpretability
- Small model size (2.5 KB)

### Feature Importance

Top factors affecting rent prices:

1. **City** (Mumbai > Bangalore > Delhi)
2. **Size** (square footage)
3. **BHK** (number of bedrooms)
4. **Furnishing Status**
5. **Area Type**

### Cross-Validation Results

5-Fold Cross-Validation RMSE: **0.2715 ± 0.008**

This indicates the model is stable and generalizes well.

---

## Project Structure

```
house-rent-midterm-project/
│
├── notebook.ipynb              # Jupyter notebook with EDA and model training
│
├── train.py                    # Script to train and save the model
├── predict.py                  # Flask web service for predictions
├── test_predict.py             # Script to test the API
│
├── model.bin                   # Trained model (pickle file)
│
├── Pipfile                     # Python dependencies
├── Pipfile.lock                # Locked dependencies for reproducibility
│
├── Dockerfile                  # Docker configuration
├── .dockerignore               # Files to exclude from Docker build
│
├── README.md                   # This file
│
└──  data/                       # (Optional) Data folder
    └── House_Rent_Dataset.csv     # Downloaded by kagglehub
```

### File Descriptions

- **`notebook.ipynb`**: 
  - Exploratory Data Analysis (EDA)
  - Feature engineering
  - Model training and evaluation
  - Visualization of results

- **`train.py`**: 
  - Downloads dataset from Kaggle
  - Preprocesses data
  - Trains Linear Regression model
  - Saves model to `model.bin`

- **`predict.py`**: 
  - Loads trained model
  - Creates Flask REST API
  - Handles prediction requests
  - Returns JSON responses

- **`test_predict.py`**: 
  - Tests the API with sample data
  - Validates predictions

- **`model.bin`**: 
  - Serialized model (DictVectorizer + LinearRegression)
  - Size: ~2.5 KB

- **`Pipfile` & `Pipfile.lock`**: 
  - Dependency management
  - Ensures reproducible environments

- **`Dockerfile`**: 
  - Instructions to build Docker image
  - Based on Python 3.11-slim
  - Installs dependencies and runs app

---

## Deployment

### Option 1: Railway.app (Easiest, Free)

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Initialize project
railway init

# Deploy
railway up

# Get URL
railway domain
```


---

### Option 2: AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Configure AWS credentials
aws configure

# Initialize application
eb init -p docker house-rent-prediction --region us-east-1

# Create environment and deploy
eb create house-rent-env

# Open application
eb open
```

---

### Option 3: AWS ECS + ECR

#### Step 1: Push to Amazon ECR

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Create repository
aws ecr create-repository --repository-name house-rent-prediction

# Tag image
docker tag house-rent-prediction:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/house-rent-prediction:latest

# Push image
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/house-rent-prediction:latest
```

#### Step 2: Deploy on ECS

1. Go to AWS Console → ECS
2. Create Cluster (Fargate)
3. Create Task Definition
4. Create Service
5. Access via Load Balancer URL

---

### Option 4: Render.com

1. Connect GitHub repository
2. Select "Docker" as environment
3. Set build command: `docker build`
4. Deploy automatically

---

## Examples

### Example 1: Cheap Apartment in Delhi

**Input:**
```json
{
  "bhk": 1,
  "size": 500,
  "bathroom": 1,
  "area_type": "carpet_area",
  "city": "delhi",
  "furnishing_status": "unfurnished",
  "tenant_preferred": "bachelors",
  "point_of_contact": "contact_owner"
}
```

**Expected Output:**
```json
{
  "predicted_rent": 12500.45,
  "predicted_rent_log": 9.4335
}
```

---

### Example 2: Luxury Apartment in Mumbai

**Input:**
```json
{
  "bhk": 3,
  "size": 2000,
  "bathroom": 3,
  "area_type": "super_area",
  "city": "mumbai",
  "furnishing_status": "furnished",
  "tenant_preferred": "family",
  "point_of_contact": "contact_owner"
}
```

**Expected Output:**
```json
{
  "predicted_rent": 85000.22,
  "predicted_rent_log": 11.3509
}
```

---

### Example 3: Mid-Range Apartment in Bangalore

**Input:**
```json
{
  "bhk": 2,
  "size": 1200,
  "bathroom": 2,
  "area_type": "super_area",
  "city": "bangalore",
  "furnishing_status": "semi-furnished",
  "tenant_preferred": "bachelors/family",
  "point_of_contact": "contact_agent"
}
```

**Expected Output:**
```json
{
  "predicted_rent": 28000.75,
  "predicted_rent_log": 10.2398
}
```

---

## Troubleshooting

### Issue 1: "Connection Refused" Error

**Problem:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Solution:**
1. Make sure the Flask service is running:
   ```bash
   python predict.py
   ```
2. Check if port 9696 is available:
   ```bash
   lsof -i :9696
   ```
3. If port is occupied, kill the process:
   ```bash
   kill -9 <PID>
   ```

---

### Issue 2: "model.bin not found"

**Problem:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'model.bin'
```

**Solution:**
Train the model first:
```bash
python train.py
```

---

### Issue 3: Docker Build Fails

**Problem:**
```
ERROR: failed to solve: failed to compute cache key
```

**Solution:**
1. Ensure `Pipfile.lock` exists:
   ```bash
   pipenv lock
   ```
2. Clean Docker cache:
   ```bash
   docker system prune -a
   ```
3. Rebuild:
   ```bash
   docker build --no-cache -t house-rent-prediction .
   ```

---

### Issue 4: Pipenv Install Errors

**Problem:**
```
pipenv install fails with dependency conflicts
```

**Solution:**
1. Delete existing environment:
   ```bash
   pipenv --rm
   ```
2. Clear cache:
   ```bash
   pipenv --clear
   ```
3. Reinstall:
   ```bash
   pipenv install
   ```

---

### Issue 5: Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'flask'
```

**Solution:**
Activate the virtual environment:
```bash
pipenv shell
```

---

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Areas for Improvement

- Add more machine learning models (XGBoost, LightGBM)
- Create a web interface (Streamlit/Gradio)
- Add visualization of rent prices by location
- Implement API authentication
- Add monitoring and logging
- Add unit tests and integration tests
- Create mobile app integration

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Author

**Your Name**
- GitHub: [@mitologistka](https://github.com/mitologistka)
- LinkedIn: [Dominika Wojtczak](https://www.linkedin.com/in/dominika-wojtczak-004383283)
- Email: dwojtczak9@gmail.com

---

## 🙏 Acknowledgments

- **ML Zoomcamp** by [DataTalks.Club](https://datatalks.club/) for the amazing course
- **Kaggle** for providing the dataset
- **Sourav Banerjee** for curating the House Rent dataset
- The open-source community for the amazing tools and libraries

---

## 📚 References

- [ML Zoomcamp Course](https://datatalks.club/blog/machine-learning-zoomcamp.html)
- [Kaggle Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Docker Documentation](https://docs.docker.com/)


---

**⭐ If you find this project helpful, please give it a star!**

**Made with ❤️ for ML Zoomcamp 2024**