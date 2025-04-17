# E-commerce Customer Behavior Analysis Project
# =============================================

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Database Setup
# =================

def create_database():
    """Create the database and tables for the e-commerce analysis."""
    conn = sqlite3.connect('ecommerce_data.db')
    cursor = conn.cursor()

    # Drop existing tables if they exist
    cursor.execute('DROP TABLE IF EXISTS transactions')
    cursor.execute('DROP TABLE IF EXISTS products')
    cursor.execute('DROP TABLE IF EXISTS customers')
    
    # Create customers table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        registration_date TEXT,
        country TEXT
    )
    ''')
    
    # Create products table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        product_id INTEGER PRIMARY KEY,
        product_name TEXT,
        category TEXT,
        price REAL
    )
    ''')
    
    # Create transactions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id INTEGER PRIMARY KEY,
        customer_id INTEGER,
        product_id INTEGER,
        quantity INTEGER,
        transaction_date TEXT,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
        FOREIGN KEY (product_id) REFERENCES products (product_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database and tables created successfully!")

# 2. Generate Sample Data
# ======================

def generate_sample_data(num_customers=1000, num_products=100, num_transactions=5000):
    """Generate sample data for the e-commerce database."""
    conn = sqlite3.connect('ecommerce_data.db')
    
    # Generate customer data
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Japan', 'India']
    customers = []
    
    for i in range(1, num_customers + 1):
        registration_date = (datetime.now() - timedelta(days=np.random.randint(1, 1000))).strftime('%Y-%m-%d')
        customers.append((
            i,
            f"Customer {i}",
            f"customer{i}@example.com",
            registration_date,
            np.random.choice(countries)
        ))
    
    # Insert customers
    cursor = conn.cursor()
    cursor.executemany(
        'INSERT INTO customers (customer_id, name, email, registration_date, country) VALUES (?, ?, ?, ?, ?)',
        customers
    )
    
    # Generate product data
    categories = ['Electronics', 'Clothing', 'Home & Kitchen', 'Books', 'Beauty', 'Sports', 'Toys']
    products = []
    
    for i in range(1, num_products + 1):
        category = np.random.choice(categories)
        if category == 'Electronics':
            price = np.random.uniform(50, 1000)
        elif category == 'Clothing':
            price = np.random.uniform(20, 200)
        else:
            price = np.random.uniform(10, 500)
        
        products.append((
            i,
            f"Product {i}",
            category,
            round(price, 2)
        ))
    
    # Insert products
    cursor.executemany(
        'INSERT INTO products (product_id, product_name, category, price) VALUES (?, ?, ?, ?)',
        products
    )
    
    # Generate transaction data
    transactions = []
    
    for i in range(1, num_transactions + 1):
        customer_id = np.random.randint(1, num_customers + 1)
        product_id = np.random.randint(1, num_products + 1)
        quantity = np.random.randint(1, 6)
        transaction_date = (datetime.now() - timedelta(days=np.random.randint(1, 365))).strftime('%Y-%m-%d')
        
        transactions.append((
            i,
            customer_id,
            product_id,
            quantity,
            transaction_date
        ))
    
    # Insert transactions
    cursor.executemany(
        'INSERT INTO transactions (transaction_id, customer_id, product_id, quantity, transaction_date) VALUES (?, ?, ?, ?, ?)',
        transactions
    )
    
    conn.commit()
    conn.close()
    print(f"Sample data generated: {num_customers} customers, {num_products} products, {num_transactions} transactions")


# 3. SQL Queries for Analysis
# ==========================

def run_sql_analysis():
    """Run SQL queries to analyze the e-commerce data."""
    conn = sqlite3.connect('ecommerce_data.db')
    
    # Query 1: Total revenue by product category
    category_revenue = pd.read_sql_query('''
    SELECT p.category, 
           ROUND(SUM(p.price * t.quantity), 2) as total_revenue,
           COUNT(DISTINCT t.transaction_id) as transaction_count
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    GROUP BY p.category
    ORDER BY total_revenue DESC
    ''', conn)
    
    # Query 2: Customer purchase frequency
    purchase_frequency = pd.read_sql_query('''
    SELECT c.customer_id, c.country,
           COUNT(DISTINCT t.transaction_id) as num_transactions,
           COUNT(DISTINCT t.product_id) as unique_products_purchased,
           ROUND(SUM(p.price * t.quantity), 2) as total_spent,
           ROUND(AVG(p.price * t.quantity), 2) as avg_transaction_value,
           MIN(t.transaction_date) as first_purchase,
           MAX(t.transaction_date) as last_purchase
    FROM customers c
    JOIN transactions t ON c.customer_id = t.customer_id
    JOIN products p ON t.product_id = p.product_id
    GROUP BY c.customer_id
    ORDER BY num_transactions DESC
    ''', conn)
    
    # Query 3: Monthly sales trends
    monthly_sales = pd.read_sql_query('''
    SELECT 
        strftime('%Y-%m', transaction_date) as month,
        ROUND(SUM(p.price * t.quantity), 2) as monthly_revenue,
        COUNT(DISTINCT t.transaction_id) as transaction_count,
        COUNT(DISTINCT t.customer_id) as unique_customers
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    GROUP BY month
    ORDER BY month
    ''', conn)
    
    # Query 4: Products often purchased together
    product_pairs = pd.read_sql_query('''
    SELECT 
        p1.product_name as product1,
        p2.product_name as product2,
        COUNT(*) as pair_count
    FROM transactions t1
    JOIN transactions t2 ON t1.customer_id = t2.customer_id AND t1.transaction_date = t2.transaction_date AND t1.product_id < t2.product_id
    JOIN products p1 ON t1.product_id = p1.product_id
    JOIN products p2 ON t2.product_id = p2.product_id
    GROUP BY p1.product_id, p2.product_id
    HAVING pair_count > 10
    ORDER BY pair_count DESC
    LIMIT 20
    ''', conn)
    
    # Query 5: Customer cohort analysis - count new customers by month
    new_customers_by_month = pd.read_sql_query('''
    SELECT 
        strftime('%Y-%m', registration_date) as cohort_month,
        COUNT(*) as new_customers
    FROM customers
    GROUP BY cohort_month
    ORDER BY cohort_month
    ''', conn)
    
    conn.close()
    
    return {
        'category_revenue': category_revenue,
        'purchase_frequency': purchase_frequency,
        'monthly_sales': monthly_sales,
        'product_pairs': product_pairs,
        'new_customers_by_month': new_customers_by_month
    }


# 4. Python Data Analysis and Customer Segmentation
# ================================================

def perform_customer_segmentation(sql_results):
    """Perform RFM analysis and customer segmentation using K-means clustering."""
    purchase_data = sql_results['purchase_frequency']
    
    # Calculate Recency (days since last purchase)
    purchase_data['last_purchase'] = pd.to_datetime(purchase_data['last_purchase'])
    current_date = datetime.now().date()
    # purchase_data['recency'] = (current_date - purchase_data['last_purchase'].dt.date).dt.days
    purchase_data['recency'] = purchase_data['last_purchase'].apply(lambda x: (current_date - x.date()).days)

    # Extract relevant features for RFM analysis (Recency, Frequency, Monetary)
    rfm_data = purchase_data[['recency', 'num_transactions', 'total_spent']].copy()
    
    # Normalize data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_data)
    
    # Apply K-means clustering to segment customers
    kmeans = KMeans(n_clusters=4, random_state=42)
    purchase_data['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Analyze the clusters
    cluster_analysis = purchase_data.groupby('cluster').agg({
        'recency': 'mean',
        'num_transactions': 'mean',
        'total_spent': 'mean',
        'customer_id': 'count'
    }).rename(columns={'customer_id': 'count'})
    
    # Label the clusters based on characteristics
    cluster_names = {
        0: "High-Value Regular Customers",
        1: "New High-Potential Customers",
        2: "At-Risk Customers",
        3: "Low-Value Occasional Customers"
    }
    
    # Customize based on actual cluster characteristics
    if cluster_analysis.shape[0] == 4:
        sorted_clusters = cluster_analysis.sort_values(['total_spent', 'num_transactions'], ascending=False)
        cluster_mapping = dict(zip(sorted_clusters.index, ['High-Value', 'Loyal', 'Occasional', 'At-Risk']))
        purchase_data['segment'] = purchase_data['cluster'].map(cluster_mapping)
    else:
        purchase_data['segment'] = purchase_data['cluster'].astype(str)
    
    return purchase_data, cluster_analysis


# 5. Visualizations
# ================

def create_visualizations(sql_results, segmentation_results):
    """Create visualizations for the analysis results."""
    segmented_customers, cluster_analysis = segmentation_results
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Revenue by Category
    ax1 = fig.add_subplot(2, 2, 1)
    category_revenue = sql_results['category_revenue']
    sns.barplot(x='total_revenue', y='category', data=category_revenue, ax=ax1)
    ax1.set_title('Revenue by Product Category', fontsize=14)
    ax1.set_xlabel('Total Revenue ($)', fontsize=12)
    ax1.set_ylabel('Category', fontsize=12)
    
    # 2. Monthly Sales Trend
    ax2 = fig.add_subplot(2, 2, 2)
    monthly_sales = sql_results['monthly_sales']
    monthly_sales['month'] = pd.to_datetime(monthly_sales['month'] + '-01')
    ax2.plot(monthly_sales['month'], monthly_sales['monthly_revenue'], marker='o', linewidth=2)
    ax2.set_title('Monthly Sales Trend', fontsize=14)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Monthly Revenue ($)', fontsize=12)
    
    # 3. Customer Segments Distribution
    ax3 = fig.add_subplot(2, 2, 3)
    segment_counts = segmented_customers['segment'].value_counts()
    sns.barplot(x=segment_counts.index, y=segment_counts.values, ax=ax3)
    ax3.set_title('Customer Segment Distribution', fontsize=14)
    ax3.set_xlabel('Segment', fontsize=12)
    ax3.set_ylabel('Number of Customers', fontsize=12)
    
    # 4. Customer Segment Comparison
    ax4 = fig.add_subplot(2, 2, 4)
    segment_metrics = segmented_customers.groupby('segment').agg({
        'total_spent': 'mean',
        'num_transactions': 'mean'
    }).reset_index()
    
    x = segment_metrics['segment']
    width = 0.35
    x_pos = np.arange(len(x))
    
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x_pos - width/2, segment_metrics['total_spent'], width, color='skyblue', label='Avg. Total Spent ($)')
    bars2 = ax4_twin.bar(x_pos + width/2, segment_metrics['num_transactions'], width, color='lightgreen', label='Avg. Transactions')
    
    ax4.set_title('Customer Segment Metrics Comparison', fontsize=14)
    ax4.set_xlabel('Segment', fontsize=12)
    ax4.set_ylabel('Average Total Spent ($)', fontsize=12)
    ax4_twin.set_ylabel('Average Number of Transactions', fontsize=12)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(x)
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('ecommerce_analysis_results.png')
    plt.close()
    
    print("Visualizations created and saved as 'ecommerce_analysis_results.png'")


# 6. Main Analysis Function
# ========================

def main():
    """Main function to run the complete analysis."""
    print("Starting E-commerce Customer Behavior Analysis...")
    
    # Create the database structure
    create_database()
    
    # Generate sample data
    generate_sample_data()
    
    # Run SQL analysis
    print("\nRunning SQL queries for data analysis...")
    sql_results = run_sql_analysis()
    
    # Perform customer segmentation
    print("\nPerforming customer segmentation...")
    segmentation_results = perform_customer_segmentation(sql_results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(sql_results, segmentation_results)
    
    # Generate insights and recommendations
    print("\nGenerating insights and recommendations...")
    generate_insights(sql_results, segmentation_results)
    
    print("\nAnalysis complete! Review the generated insights and visualization file.")


def generate_insights(sql_results, segmentation_results):
    """Generate business insights and recommendations based on the analysis."""
    segmented_customers, cluster_analysis = segmentation_results
    
    # Calculate key metrics
    high_value_pct = (segmented_customers['segment'] == 'High-Value').mean() * 100
    at_risk_pct = (segmented_customers['segment'] == 'At-Risk').mean() * 100
    top_category = sql_results['category_revenue'].iloc[0]['category']
    top_category_revenue = sql_results['category_revenue'].iloc[0]['total_revenue']
    
    # Create insights report
    with open('ecommerce_insights_report.txt', 'w') as f:
        f.write("E-COMMERCE CUSTOMER BEHAVIOR ANALYSIS - INSIGHTS & RECOMMENDATIONS\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        f.write(f"• {top_category} is the highest-performing category, generating ${top_category_revenue:.2f} in revenue.\n")
        f.write(f"• {high_value_pct:.1f}% of customers are in the High-Value segment, representing our most valuable customers.\n")
        f.write(f"• {at_risk_pct:.1f}% of customers are At-Risk and need immediate attention to prevent churn.\n\n")
        
        f.write("CUSTOMER SEGMENTATION INSIGHTS\n")
        f.write("-" * 30 + "\n")
        for segment in segmented_customers['segment'].unique():
            segment_data = segmented_customers[segmented_customers['segment'] == segment]
            avg_spent = segment_data['total_spent'].mean()
            avg_transactions = segment_data['num_transactions'].mean()
            avg_recency = segment_data['recency'].mean()
            
            f.write(f"{segment} Segment:\n")
            f.write(f"  • Average spend: ${avg_spent:.2f}\n")
            f.write(f"  • Average transactions: {avg_transactions:.1f}\n")
            f.write(f"  • Average days since last purchase: {avg_recency:.1f}\n")
            
            # Segment-specific recommendations
            if segment == 'High-Value':
                f.write("  • Recommendation: Implement a premium loyalty program to reward and retain these customers.\n")
            elif segment == 'Loyal':
                f.write("  • Recommendation: Focus on cross-selling related products to increase order value.\n")
            elif segment == 'Occasional':
                f.write("  • Recommendation: Create email marketing campaigns to increase purchase frequency.\n")
            elif segment == 'At-Risk':
                f.write("  • Recommendation: Send targeted win-back campaigns with special offers.\n")
            f.write("\n")
        
        f.write("PRODUCT INSIGHTS\n")
        f.write("-" * 20 + "\n")
        for _, row in sql_results['category_revenue'].iterrows():
            f.write(f"• {row['category']}: ${row['total_revenue']:.2f} revenue from {row['transaction_count']} transactions\n")
        
        f.write("\nCROSS-SELLING OPPORTUNITIES\n")
        f.write("-" * 25 + "\n")
        for _, row in sql_results['product_pairs'].head(5).iterrows():
            f.write(f"• Bundle opportunity: {row['product1']} + {row['product2']} (purchased together {row['pair_count']} times)\n")
        
        f.write("\nACTIONABLE RECOMMENDATIONS\n")
        f.write("-" * 25 + "\n")
        f.write("1. Launch a tiered loyalty program targeting the specific needs of each customer segment\n")
        f.write("2. Develop personalized email campaigns based on purchase history and segment\n")
        f.write("3. Create bundle offers for frequently co-purchased products\n")
        f.write("4. Focus marketing budget on highest-performing product categories\n")
        f.write("5. Implement a win-back campaign for At-Risk customers\n")
    
    print("Insights and recommendations saved to 'ecommerce_insights_report.txt'")


if __name__ == "__main__":
    main()
