# Databricks notebook source
# COMMAND ----------

# Install required packages
!pip install shap pandas beautifulsoup4 matplotlib mlflow

# COMMAND ----------

import pandas as pd
import pickle
from bs4 import BeautifulSoup
import shap
import matplotlib.pyplot as plt
import re
import base64
from io import BytesIO
import mlflow
import mlflow.pyfunc
import json
from typing import Dict, Any
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

# Define the model class that inherits from mlflow.pyfunc.PythonModel
class PromotionPredictorModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def load_context(self, context):
        # Load any additional context if needed
        pass
    
    @staticmethod
    def normalize_text(s, sep_token=" \n "):
        s = s.lower()
        s = re.sub(r'\s+', ' ', s).strip()
        s = re.sub(r". ,","",s)
        s = s.replace("..", ".")
        s = s.replace(". .", ".")
        s = s.replace("\n", "")
        s = s.strip()
        return s

    @staticmethod
    def get_text_to_html_ratio(content, soup):
        text_length = PromotionPredictorModel.get_text_len(soup)
        html_length = len(content)
        return text_length / html_length if html_length != 0 else 0

    @staticmethod
    def get_text_len(soup):
        return len(soup.get_text())

    @staticmethod
    def get_num_links(soup):
        return PromotionPredictorModel.get_tag_nums(soup, 'a')

    @staticmethod
    def get_num_imgs(soup):
        return PromotionPredictorModel.get_tag_nums(soup, 'img')

    @staticmethod
    def get_tag_nums(soup, tag_type):
        return len(soup.find_all(tag_type))

    @staticmethod
    def get_all_tag_nums(soup):
        tags = soup.find_all()
        tag_count = len(tags)
        tag_types = set(tag.name for tag in tags)
        return tag_count, len(tag_types)

    @staticmethod
    def get_max_depth(soup):
        if hasattr(soup, "contents") and soup.contents:
            return max([PromotionPredictorModel.get_max_depth(child) for child in soup.contents]) + 1
        else:
            return 0

    @staticmethod
    def get_external_resource_num(soup):
        return len(soup.find_all(['img', 'link', 'script']))

    @staticmethod
    def get_css_num(soup):
        return PromotionPredictorModel.get_tag_nums(soup, 'style')

    @staticmethod
    def get_css_len(soup):
        style_tags = soup.find_all('style')
        inline_css = ''.join(tag.string for tag in style_tags if tag.string)
        return len(inline_css.encode('utf-8'))

    def extract_html_fea(self, content):
        soup = BeautifulSoup(content, 'html.parser')
        dic = {}
        dic['content_len'] = len(content)
        text_content = soup.get_text()
        dic['text_len'] = len(text_content)
        dic['text_to_html_ratio'] = self.get_text_to_html_ratio(content, soup)
        dic['num_links'] = self.get_num_links(soup)
        dic['num_imgs'] = self.get_num_imgs(soup)

        tag_count, tag_type_cnt = self.get_all_tag_nums(soup)
        dic['tag_count'] = tag_count
        dic['tag_type_cnt'] = tag_type_cnt

        dic['html_max_depth'] = self.get_max_depth(soup)
        dic['external_resources_count'] = self.get_external_resource_num(soup)
        dic['num_css_tag'] = self.get_css_num(soup)

        css_len = self.get_css_len(soup)
        dic['css_to_html_ratio'] = css_len / len(content)

        return dic

    @staticmethod
    def extract_text_fea(text, label):
        return {f'{label}_len': len(text)}
        
    def predict(self, context, model_input):
        # Extract features from HTML content
        html_content = model_input['html_content'].iloc[0]
        subject = model_input['subject'].iloc[0]
        
        # Extract features using the class methods
        html_fea = self.extract_html_fea(html_content)
        subject_fea = self.extract_text_fea(subject, "email_subject")
        
        # Combine features
        combined_fea = {**html_fea, **subject_fea}
        
        # Convert to DataFrame
        df = pd.DataFrame([combined_fea])
        
        # Fix feature names to match model
        df['css_to_html_ration'] = df['css_to_html_ratio']
        df['text_content_len'] = df['text_len']
        
        # Select only the features needed by the model
        df_model = df[self.feature_names]
        
        # Get prediction
        prediction = self.model.predict_proba(df_model)[:, 0][0]
        
        # Get SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(df_model)
        
        # Convert SHAP values to a dictionary
        shap_dict = {
            feature: float(value) 
            for feature, value in zip(self.feature_names, shap_values[0].values)
        }
        
        # Sort features by absolute SHAP value
        sorted_features = dict(sorted(
            shap_dict.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        return {
            'promotion_probability': float(prediction),
            'feature_importance': sorted_features,
            'raw_features': df_model.to_dict('records')[0]
        }

# COMMAND ----------

# Load the model and feature names
model_path = '/dbfs/path/to/your/promotion_xgb.pkl'  # Update this path
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Get feature names
feature_names = None
if hasattr(model, 'feature_names_in_'):
    feature_names = model.feature_names_in_
elif hasattr(model, 'get_booster'):
    booster = model.get_booster()
    feature_names = booster.feature_names

# COMMAND ----------

# Set MLflow registry parameters
catalog = "crm_dev"
schema = "smartlink"
model_name = "smartlink_promotion_pred"
alias = "demo"

# COMMAND ----------

# Create sample training data for model signature
def create_sample_training_data():
    sample_data = [
        {
            'html_content': """
            <html>
            <body>
            <h1>Special Offer!</h1>
            <p>Get 50% off on all products.</p>
            <a href="https://example.com/sale">Click here</a>
            <img src="sale.jpg" alt="Sale banner">
            </body>
            </html>
            """,
            'subject': "Limited Time Offer - 50% Off Everything!"
        },
        {
            'html_content': """
            <html>
            <body>
            <h1>Your Order Status</h1>
            <p>Your order #12345 has been shipped.</p>
            <p>Track your package here.</p>
            </body>
            </html>
            """,
            'subject': "Order #12345 - Shipped"
        },
        {
            'html_content': """
            <html>
            <body>
            <h1>Newsletter</h1>
            <p>Latest updates from our blog.</p>
            <style>
            .newsletter { color: blue; }
            </style>
            <div class="newsletter">
            <p>Read our latest articles.</p>
            </div>
            </body>
            </html>
            """,
            'subject': "Monthly Newsletter - March 2024"
        },
        {
            'html_content': """
            <html>
            <body>
            <h1>Flash Sale!</h1>
            <p>24 hours only - Up to 70% off</p>
            <script>
            function countdown() { /* ... */ }
            </script>
            <div id="countdown"></div>
            </body>
            </html>
            """,
            'subject': "🚨 Flash Sale Alert - 24 Hours Only!"
        },
        {
            'html_content': """
            <html>
            <body>
            <h1>Account Update</h1>
            <p>Your account settings have been updated.</p>
            <p>If you didn't make these changes, please contact support.</p>
            </body>
            </html>
            """,
            'subject': "Important: Account Settings Updated"
        }
    ]
    
    return pd.DataFrame(sample_data)

# Create training data
training_data = create_sample_training_data()

# COMMAND ----------

# Set the MLflow registry URI to Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Register the model with MLflow
with mlflow.start_run(run_name=f"{model_name}_registration") as run:
    # Log model parameters
    mlflow.log_params({
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "catalog": catalog,
        "schema": schema
    })
    
    # Create and log the model
    promotion_model = PromotionPredictorModel(model, feature_names)
    
    # Define input and output schema for the model
    input_schema = Schema([
        ColSpec("string", "html_content"),
        ColSpec("string", "subject")
    ])
    
    output_schema = Schema([
        ColSpec("double", "promotion_probability"),
        ColSpec("string", "feature_importance"),  # Dictionary will be serialized as JSON string
        ColSpec("string", "raw_features")         # Dictionary will be serialized as JSON string
    ])
    
    # Create model signature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    
    # Log example input data
    example_input = pd.DataFrame({
        'html_content': ["""
        <html>
        <body>
        <h1>Special Offer!</h1>
        <p>Get 50% off on all products.</p>
        <a href="https://example.com/sale">Click here</a>
        </body>
        </html>
        """],
        'subject': ["Limited Time Offer - 50% Off Everything!"]
    })
    
    # Get example prediction
    example_output = promotion_model.predict(None, example_input)
    
    # Log example input and output
    mlflow.log_dict(
        {
            "example_input": example_input.to_dict('records')[0],
            "example_output": example_output
        },
        "model_examples.json"
    )
    
    # Log the model with specified catalog, schema, and signature
    registered_model = mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=promotion_model,
        registered_model_name=f"{catalog}.{schema}.{model_name}",
        signature=signature,
        input_example=example_input,
        await_registration_for=600  # Wait for up to 10 minutes for registration
    )
    
    # Set the alias for the newly registered model version
    try:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=f"{catalog}.{schema}.{model_name}",
            alias=alias,
            version=1  # First version will be 1
        )
    except Exception as e:
        print(f"Warning: Could not set alias. Error: {str(e)}")
        print("The model was still registered successfully.")

# COMMAND ----------

# Test the registered model
def test_model():
    # Create sample input
    sample_input = pd.DataFrame({
        'html_content': ["""
        <html>
        <body>
        <h1>Sample HTML</h1>
        <p>This is a test email.</p>
        </body>
        </html>
        """],
        'subject': ["Test Subject"]
    })
    
    try:
        # Load the model from registry using the alias
        model_uri = f"models:/{catalog}.{schema}.{model_name}@{alias}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        
        # Make prediction
        result = loaded_model.predict(sample_input)
        
        # Display results
        print("Model Test Results:")
        print(f"Model URI: {model_uri}")
        print(f"Promotion Probability: {result['promotion_probability']:.4f}")
        print("\nTop 5 Important Features:")
        for feature, importance in list(result['feature_importance'].items())[:5]:
            print(f"{feature}: {importance:.4f}")
    except Exception as e:
        print(f"Warning: Could not test model. Error: {str(e)}")
        print("You may need to wait a few minutes for the model to be fully registered.")

# Run the test
test_model()

# COMMAND ----------

# Example of how to use the model in Airflow
"""
Example Airflow DAG code:

from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksRunNowOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'promotion_prediction',
    default_args=default_args,
    description='Run promotion prediction model',
    schedule_interval=timedelta(days=1),
)

predict_task = DatabricksRunNowOperator(
    task_id='predict_promotion',
    databricks_conn_id='databricks_default',
    job_id=YOUR_JOB_ID,  # Replace with your job ID
    notebook_params={
        'html_content': '{{ dag_run.conf.html_content }}',
        'subject': '{{ dag_run.conf.subject }}'
    },
    dag=dag,
)
""" 