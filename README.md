# Travel Customer Purchase Prediction MLOps Project

## Problem Statement

"Visit with Us," a leading travel company, is revolutionizing the tourism industry by leveraging data-driven strategies to optimize operations and customer engagement. While introducing a new package offering, such as the Wellness Tourism Package, the company faces challenges in targeting the right customers efficiently. The manual approach to identifying potential customers is inconsistent, time-consuming, and prone to errors, leading to missed opportunities and suboptimal campaign performance.

To address these issues, the company aims to implement a scalable and automated system that integrates customer data, predicts potential buyers, and enhances decision-making for marketing strategies. By utilizing an MLOps pipeline, the company seeks to achieve seamless integration of data preprocessing, model development, deployment, and CI/CD practices for continuous improvement. This system will ensure efficient targeting of customers, timely updates to the predictive model, and adaptation to evolving customer behaviors, ultimately driving growth and customer satisfaction.

## Business Context

"Visit with Us" is a travel company aiming to improve customer targeting for new package offerings, specifically the Wellness Tourism Package. The current manual process is inefficient and error-prone. Implementing an MLOps pipeline will automate the process of identifying potential customers, leading to better marketing campaign performance and increased customer satisfaction.

## Objective

The primary objective of this project is to design and deploy an MLOps pipeline on GitHub to automate the end-to-end workflow for predicting customer purchases of the Wellness Tourism Package. This involves building a predictive model, integrating data cleaning, preprocessing, transformation, model building, training, evaluation, and deployment. The pipeline will leverage GitHub Actions for CI/CD to ensure automated updates and streamline deployment.

## Data Description

The dataset contains customer and interaction data used to predict the likelihood of purchasing the Wellness Tourism Package. Key attributes include:

**Customer Details**
- **CustomerID:** Unique identifier for each customer.
- **ProdTaken:** Target variable indicating whether the customer has purchased a package (0: No, 1: Yes).
- **Age:** Age of the customer.
- **TypeofContact:** The method by which the customer was contacted (Company Invited or Self Inquiry).
- **CityTier:** The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3).
- **Occupation:** Customer's occupation (e.g., Salaried, Freelancer).
- **Gender:** Gender of the customer (Male, Female).
- **NumberOfPersonVisiting:** Total number of people accompanying the customer on the trip.
- **PreferredPropertyStar:** Preferred hotel rating by the customer.
- **MaritalStatus:** Marital status of the customer (Single, Married, Divorced).
- **NumberOfTrips:** Average number of trips the customer takes annually.
- **Passport:** Whether the customer holds a valid passport (0: No, 1: Yes).
- **OwnCar:** Whether the customer owns a car (0: No, 1: Yes).
- **NumberOfChildrenVisiting:** Number of children below age 5 accompanying the customer.
- **Designation:** Customer's designation in their current organization.
- **MonthlyIncome:** Gross monthly income of the customer.

**Customer Interaction Data**
- **PitchSatisfactionScore:** Score indicating the customer's satisfaction with the sales pitch.
- **ProductPitched:** The type of product pitched to the customer.
- **NumberOfFollowups:** Total number of follow-ups by the salesperson after the sales pitch.-
- **DurationOfPitch:** Duration of the sales pitch delivered to the customer.

## MLOps Pipeline

The project implements an MLOps pipeline with the following stages:

1.  **Data Registration:** The raw dataset is registered and versioned, likely on Hugging Face Hub as indicated in the notebook.
2.  **Data Preparation:** The data is cleaned, preprocessed, and split into training and testing sets. Categorical features are encoded, and numerical features are scaled.
3.  **Model Training and Registration:** An XGBoost model is trained on the prepared data. Hyperparameter tuning is performed using GridSearchCV. The best model is saved and registered, likely on Hugging Face Hub. Experiment tracking is also mentioned.
4.  **Deployment:** The trained model is deployed as a web application using Streamlit. A Dockerfile is created for containerization.
5.  **Hosting:** The Streamlit application and the model are hosted, likely on Hugging Face Spaces.
6.  **CI/CD with GitHub Actions:** A GitHub Actions workflow is configured to automate the pipeline steps (data registration, data preparation, model training, and deployment) upon pushing changes to the main branch.
