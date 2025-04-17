# E-commerce Customer Behavior Analysis

## Project Overview
This Python-based data analytics project demonstrates advanced SQL and Python skills for data analysis by examining customer behavior in an e-commerce platform. The project showcases database management, data processing, statistical analysis, customer segmentation, and data visualization techniques.

## Features
- **Database Management**: Creates and populates SQLite database with customers, products, and transactions data
- **SQL Analysis**: Performs complex SQL queries to extract meaningful insights from transaction data
- **Customer Segmentation**: Implements RFM (Recency, Frequency, Monetary) analysis and K-means clustering
- **Data Visualization**: Generates informative charts showing revenue by category, sales trends, and customer segments
- **Business Insights**: Produces actionable business recommendations based on the analysis

## Technologies Used
- **Python 3.12**: Core programming language
- **SQLite**: Database management
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning for customer segmentation
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebooks**: Optional for interactive development

## Project Structure
- `behavior-analysis.py`: Main Python script containing all functions
- `requirements.txt`: List of required Python libraries
- `ecommerce_data.db`: SQLite database (generated when script runs)
- `ecommerce_analysis_results.png`: Output visualizations
- `ecommerce_insights_report.txt`: Generated business insights

## Installation & Usage
1. Clone the repository
2. Create and activate a Python virtual environment:
   ```
   python3.12 -m venv env
   source env/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the analysis:
   ```
   python behavior-analysis.py
   ```

## Analysis Outputs
- **Customer Segmentation**: Identifies key customer groups (High-Value, Loyal, Occasional, At-Risk)
- **Product Analysis**: Reveals top-performing categories and products
- **Sales Trends**: Shows revenue patterns over time
- **Cross-Selling Opportunities**: Identifies products frequently purchased together
- **Strategic Recommendations**: Provides actionable insights for marketing and sales strategies

## Future Enhancements
- Interactive dashboard using Plotly or Dash
- Predictive analytics for customer churn prevention
- Time series forecasting for sales prediction
- Product recommendation engine

## Ideal For
- Data Analyst portfolios
- Business Intelligence demonstrations
- SQL and Python skill showcases
- E-commerce data analysis examples
