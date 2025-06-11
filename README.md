# AI-Based Multi-Level Product Classification for B2B Catalogs

This project implements an **AI-driven product classification pipeline** designed to enhance the organization, searchability, and usability of B2B e-commerce catalogs.

It provides an automated workflow to process raw product data, apply multi-level classification, and generate structured category outputs — reducing manual effort and improving catalog accuracy.

## Features

- Automated **web scraping** pipeline to gather product data.
- **Data cleaning and preprocessing** of raw text and metadata.
- Multi-level **product classification pipeline** using NLP and traditional machine learning models.
- Structured category outputs to enhance **catalog navigation, search, and analytics**.
- Scalable and modular design for integration into larger B2B catalog systems.

## Repository Structure

Data Cleaning/ # Scripts for cleaning and preprocessing raw product data
Dataset/ # Sample datasets used for training/testing
Modelling/ # Scripts for model training, evaluation, and prediction
Web Scrapping/ # Web scraping scripts to gather product data
README.md # Project documentation


## Technologies Used

- **Python** (Pandas, NumPy, Scikit-learn)
- **Selenium** for automated web scraping
- **NLP preprocessing** (text cleaning, tokenization, feature extraction)
- **Supervised ML models** for classification
- **Data visualization** for performance evaluation

## Results

The AI-based multi-level product classification system delivered significant improvements in catalog accuracy and operational efficiency:

- **25% improvement in catalog accuracy**  
  Automated classification resulted in more accurate category assignments compared to manual processes, improving product searchability and catalog quality.

- **40+ hours/month saved**  
  The automated pipeline reduced manual effort for product categorization by over 40 hours per month.

- **22% increase in labeling accuracy**  
  The NLP-based classification approach achieved a 22% higher accuracy on product category labeling compared to baseline methods.

- **Faster turnaround and scalability**  
  Automated data ingestion, cleaning, and classification workflows improved project turnaround time and enabled the system to scale efficiently across large product datasets.

These results demonstrate the value of AI-driven solutions in enhancing the organization and management of large-scale B2B product catalogs.

## Getting Started

1. Clone this repository:
    ```bash
    git clone https://github.com/shweta7595/AI-Based-Multi-Level-Product-Classification-for-B2B-Catalogs.git
    cd AI-Based-Multi-Level-Product-Classification-for-B2B-Catalogs
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the **web scraping pipeline** (optional, if gathering fresh data):
    ```bash
    python Web\ Scrapping/scrape_products.py
    ```

4. Run **data cleaning and preprocessing**:
    ```bash
    python Data\ Cleaning/clean_data.py
    ```

5. Train and evaluate classification models:
    ```bash
    python Modelling/train_classification_model.py
    ```

## Future Enhancements

- Expand model coverage to more product categories.
- Incorporate additional product attributes and metadata.
- Deploy the system as an API for real-time product classification.
- Integrate with larger B2B catalog management systems.

## License

This project is licensed under the MIT License. Feel free to use and adapt it.
