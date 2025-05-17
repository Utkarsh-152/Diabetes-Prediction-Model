# Diabetes Prediction Model

![Diabetes Prediction](https://img.shields.io/badge/Health-Diabetes%20Prediction-blue)
![Python](https://img.shields.io/badge/Python-3.10-brightgreen)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.0.1-red)

## 🔗 Live Demo

The application is deployed and accessible at: [https://diabetes-prediction-model-z1i5.onrender.com/](https://diabetes-prediction-model-z1i5.onrender.com/)

## 📋 Overview

This project is an end-to-end machine learning application that predicts the risk of diabetes based on various health indicators. The application uses a machine learning model trained on health data to provide personalized risk assessments and identify potential risk factors.

The system features:
- Interactive web interface for user input
- Real-time prediction using an XGBoost classifier
- Detailed risk assessment with identified risk factors
- Model performance evaluation dashboard

## 🏗️ Project Structure

```
diabetes_prediction_model/
├── artifacts/                  # Model artifacts and processed data
├── logs/                       # Application logs
├── model_evaluation_report/    # Model performance metrics
├── src/
│   └── ml_model/
│       ├── components/         # ML pipeline components
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   └── model_monitering.py
│       ├── pipelines/          # ML pipelines
│       │   ├── prediction_pipeline.py
│       │   └── training_pipeline.py
│       ├── exception.py        # Custom exception handling
│       ├── logger.py           # Logging configuration
│       └── utils.py            # Utility functions
├── templates/                  # HTML templates for web interface
│   ├── hello.html              # Home page
│   ├── predict.html            # Prediction form and results
│   └── model_evaluation.html   # Model performance dashboard
├── app.py                      # FastAPI application
├── main.py                     # ML pipeline execution script
├── pyproject.toml              # Project dependencies (uv)
└── requirements.txt            # Project dependencies
```

## 🚀 Technologies Used

- **Python 3.10**: Core programming language
- **FastAPI**: Web framework for building the API
- **Jinja2**: Templating engine for HTML pages
- **uv**: Modern Python package manager for dependency management
- **Scikit-Learn**: Machine learning library for data preprocessing and model evaluation
- **XGBoost**: Gradient boosting framework for the prediction model
- **Pandas & NumPy**: Data manipulation and numerical computing
- **MySQL**: Database for storing and retrieving health data
- **Tailwind CSS**: Utility-first CSS framework for styling
- **Chart.js**: JavaScript library for data visualization
- **Render**: Cloud platform for deployment

## ✨ Features

### Data Processing Pipeline
- Data ingestion from MySQL database
- Feature selection using SelectKBest with Chi-Square
- Handling class imbalance with NearMiss undersampling
- Data standardization with StandardScaler

### Machine Learning Models
- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- XGBoost Classifier (best performing model)

### Web Application
- Interactive form with sliders and button-style inputs
- Real-time prediction with risk factor identification
- Model performance dashboard with metrics visualization
- Responsive design for all device sizes

## 📊 Model Performance

The XGBoost Classifier achieved the following performance metrics:

- **Accuracy**: 86.5%
- **Precision**: 93.8%
- **Recall**: 78.3%
- **F1 Score**: 85.4%

A detailed visualization of the model's performance is available on the Model Evaluation page of the application.

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.10 or higher
- MySQL database
- uv package manager

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-model.git
   cd diabetes-prediction-model
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies using uv:
   ```bash
   pip install uv
   uv pip install -r requirements.txt
   ```

4. Configure environment variables:
   Create a `.env` file with the following variables:
   ```
   host=your_mysql_host
   user=your_mysql_user
   password=your_mysql_password
   db=your_database_name
   ```

5. Run the ML pipeline to train the model:
   ```bash
   python main.py
   ```

6. Start the FastAPI application:
   ```bash
   uvicorn app:app --reload
   ```

7. Access the application at http://localhost:8000

## 📝 Usage

1. Visit the home page and click on "Start Prediction"
2. Fill out the health information form with your details
3. Submit the form to get your diabetes risk assessment
4. View the identified risk factors and recommendations
5. Explore the model evaluation page to understand the model's performance

## 🧪 Model Training Process

The model training process follows these steps:

1. **Data Ingestion**: Fetch data from MySQL database
2. **Data Transformation**:
   - Clean and preprocess the data
   - Select important features using Chi-Square test
   - Handle class imbalance with NearMiss
   - Split data into training and testing sets
   - Standardize features
3. **Model Training**:
   - Train multiple classifier models
   - Evaluate models based on accuracy and recall
   - Select the best performing model (XGBoost)
4. **Model Monitoring**:
   - Generate performance metrics
   - Create evaluation reports
   - Save confusion matrix and other metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For any questions or feedback, please reach out to:
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/your-profile/)

---

Developed with ❤️ by Utkarsh