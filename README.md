
# Iris Flower Classification REST API using FastAPI

## 1. Project Overview

This project implements a REST API using FastAPI to serve predictions from a trained Machine Learning model.

The API accepts four Iris flower measurements in JSON format and predicts the flower species along with its confidence score and class probabilities.

The model is trained using the Iris dataset from Scikit-learn and uses the Random Forest Classification algorithm.

## 2. Technologies Used

- Python 3.12
- FastAPI
- Scikit-learn
- NumPy
- Joblib
- Pydantic
- Uvicorn
- Pytest
- HTTPX

## 3. Machine Learning Model

**Algorithm:** Random Forest Classifier

**Dataset:** Iris Dataset

**Number of Features:** 4

**Number of Classes:** 3

The four input features are:
1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

The model classifies flowers into:
- Setosa
- Versicolor
- Virginica

**Model Configuration:**
- Number of estimators: 100
- Random state: 42
- Training data: 80%
- Testing data: 20%

**Test Accuracy:** 90%

## 4. Project Structure

```text
Assignment-5/
│
├── artifacts/
│   └── iris_model.joblib
│
├── tests/
│   └── test_api.py
│
├── main.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 5. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Welcome message |
| GET | /health | Check API and model status |
| GET | /model-info | Retrieve model information |
| POST | /predict | Predict Iris flower species |

## 6. Installation and Setup

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Assignment-5
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the environment

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Train the Machine Learning model

```bash
python train_model.py
```

This generates the trained model at:

`artifacts/iris_model.joblib`

### Step 6: Start the FastAPI server

```bash
uvicorn main:app --reload
```

The API will be available at:

http://127.0.0.1:8000

## 7. API Documentation

FastAPI automatically generates interactive API documentation.

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- OpenAPI Schema: http://127.0.0.1:8000/openapi.json

## 8. Prediction Request Example

**Endpoint:** POST /predict

**Request Body:**

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

**Example Response:**

```json
{
    "predicted_class": 0,
    "predicted_species": "setosa",
    "confidence": 1.0,
    "probabilities": {
        "setosa": 1.0,
        "versicolor": 0.0,
        "virginica": 0.0
    },
    "model_version": "1.0.0"
}
```

Probability values may vary depending on the trained model.

## 9. Input Validation and Error Handling

Pydantic validates incoming request data.

The API rejects:
- Negative or zero measurements.
- Measurements greater than 20.
- Missing required features.
- Incorrect data types.

Important HTTP status codes:

| Status Code | Description |
|-------------|-------------|
| 200 | Request successful |
| 404 | Endpoint not found |
| 422 | Input validation failed |
| 500 | Prediction failure |

## 10. Automated Testing

Automated testing is implemented using Pytest and FastAPI TestClient.

Run the tests using:

```bash
python -m pytest -v
```

### Test Cases

1. Root endpoint test.
2. Health-check endpoint test.
3. Model information test.
4. Valid prediction test.
5. Negative feature value test.
6. Missing feature test.
7. Incorrect data type test.
8. Unknown endpoint test.

**Testing Result: 8 Tests Passed**

## 11. Project Workflow

1. Load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Train a Random Forest classification model.
4. Evaluate the model using accuracy.
5. Save the trained model using Joblib.
6. Load the model during FastAPI application startup.
7. Receive and validate JSON input.
8. Generate predictions using the trained model.
9. Return the predicted species and confidence score.
10. Test API functionality using Pytest.

## 12. Conclusion

This project demonstrates how a trained Machine Learning model can be integrated with FastAPI and exposed through REST API endpoints.

It covers model training, model serialization, API development, input validation, prediction serving, automatic documentation, and automated testing.

The implementation successfully demonstrates the deployment of a Machine Learning model through a REST API.
