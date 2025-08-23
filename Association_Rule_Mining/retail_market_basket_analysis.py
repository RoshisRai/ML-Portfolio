"""
Market Basket Analysis Using Apriori Algorithm and Association Rules
===================================================================
This script performs market basket analysis on retail transaction data to:
1. Discover frequent itemsets using the Apriori algorithm
2. Generate association rules to identify product relationships
3. Provide actionable insights for cross-selling strategies

Dataset: Online Retail dataset with transaction records
Algorithm: Apriori algorithm for frequent pattern mining and association rule generation
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

def load_and_clean_data(filepath='Online_retail.xlsx'):
    """
    Load and clean retail transaction data in one step.
    
    Args:
        filepath (str): Path to the Excel file
        
    Returns:
        pandas.DataFrame: Cleaned transaction data
    """
    print("=== Market Basket Analysis - Association Rule Mining ===\n")
    
    try:
        # Load data with fallback options
        try:
            data = pd.read_excel(filepath, engine='openpyxl')
        except ImportError:
            print("Please install openpyxl: pip install openpyxl")
            return None
        
        print(f"Dataset loaded: {data.shape}")
        
        # Clean data in one pipeline
        cleaned_data = (data
                       .dropna(subset=['InvoiceNo', 'Description'])
                       .assign(InvoiceNo=lambda x: x['InvoiceNo'].astype('str'))
                       .query("~InvoiceNo.str.contains('C', na=False)")
                       .query("Quantity > 0")
                       .assign(Description=lambda x: x['Description'].str.strip())
                       )
        
        print(f"Cleaned dataset: {cleaned_data.shape}")
        print(f"Top countries: {cleaned_data['Country'].value_counts().head(3).to_dict()}")
        
        return cleaned_data
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def create_basket_matrix(data, country='United Kingdom', min_freq=30, max_products=100):
    """
    Create binary market basket matrix for analysis.
    
    Args:
        data (pandas.DataFrame): Cleaned transaction data
        country (str): Country to analyze
        min_freq (int): Minimum product frequency
        max_products (int): Maximum products to include
        
    Returns:
        pandas.DataFrame: Binary encoded market basket
    """
    print(f"\n=== Creating Market Basket for {country} ===")
    
    # Filter data
    country_data = data[data['Country'] == country]
    print(f"Transactions: {len(country_data)}")
    
    # Get top frequent products to speed up processing
    top_products = (country_data['Description']
                   .value_counts()
                   .head(max_products)
                   .index.tolist())
    
    country_data = country_data[country_data['Description'].isin(top_products)]
    print(f"Using top {len(top_products)} products")
    
    # Create basket matrix
    basket = (country_data
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum()
              .unstack(fill_value=0)
              .applymap(lambda x: 1 if x > 0 else 0))  # Binary encoding
    
    # Filter by minimum frequency
    frequent_mask = basket.sum() >= min_freq
    basket = basket.loc[:, frequent_mask]
    
    print(f"Final basket shape: {basket.shape}")
    print(f"Average items per transaction: {basket.sum(axis=1).mean():.1f}")
    
    return basket

def analyze_market_basket(basket, min_support=0.02, min_confidence=0.3, min_lift=1.0):
    """
    Perform complete market basket analysis.
    
    Args:
        basket (pandas.DataFrame): Binary market basket
        min_support (float): Minimum support for frequent itemsets
        min_confidence (float): Minimum confidence for rules
        min_lift (float): Minimum lift for rules
        
    Returns:
        tuple: (frequent_itemsets, association_rules)
    """
    print(f"\n=== Market Basket Analysis ===")
    print(f"Parameters: support≥{min_support}, confidence≥{min_confidence}, lift≥{min_lift}")
    
    # Find frequent itemsets
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True, max_len=3)
    print(f"Frequent itemsets found: {len(frequent_itemsets)}")
    
    if len(frequent_itemsets) == 0:
        print("❌ No frequent itemsets found. Try lower support threshold.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Generate association rules
    try:
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)
        rules = rules[rules['lift'] >= min_lift]  # Filter by lift
        rules = rules.sort_values(['confidence', 'lift'], ascending=False)
        
        print(f"Association rules generated: {len(rules)}")
        
        if len(rules) > 0:
            print(f"Avg confidence: {rules['confidence'].mean():.3f}")
            print(f"Avg lift: {rules['lift'].mean():.3f}")
        
        return frequent_itemsets, rules
        
    except Exception as e:
        print(f"❌ Error generating rules: {str(e)}")
        return frequent_itemsets, pd.DataFrame()

def display_results(frequent_itemsets, rules, top_n=10):
    """
    Display analysis results and insights.
    
    Args:
        frequent_itemsets (pandas.DataFrame): Frequent itemsets
        rules (pandas.DataFrame): Association rules
        top_n (int): Number of top rules to display
    """
    if len(rules) == 0:
        print("No rules to display.")
        return
    
    print(f"\n=== Top {top_n} Association Rules ===")
    print("-" * 80)
    print(f"{'#':<3} {'Antecedent → Consequent':<45} {'Conf':<6} {'Lift':<6}")
    print("-" * 80)
    
    for i, (_, rule) in enumerate(rules.head(top_n).iterrows(), 1):
        ant = ', '.join(list(rule['antecedents']))[:20] + ('...' if len(', '.join(list(rule['antecedents']))) > 20 else '')
        con = ', '.join(list(rule['consequents']))[:15] + ('...' if len(', '.join(list(rule['consequents']))) > 15 else '')
        rule_str = f"{ant} → {con}"
        
        print(f"{i:<3} {rule_str:<45} {rule['confidence']:<6.3f} {rule['lift']:<6.3f}")
    
    # Quick insights
    print(f"\n📊 Quick Insights:")
    high_conf = len(rules[rules['confidence'] >= 0.6])
    high_lift = len(rules[rules['lift'] >= 1.5])
    print(f"• {high_conf} rules with confidence ≥ 60%")
    print(f"• {high_lift} rules with lift ≥ 1.5")
    
    if len(frequent_itemsets) > 0:
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        print(f"• Most common itemset size: {frequent_itemsets['length'].mode().iloc[0]}")

def create_visualizations(rules, frequent_itemsets):
    """
    Create simple visualizations of results.
    
    Args:
        rules (pandas.DataFrame): Association rules
        frequent_itemsets (pandas.DataFrame): Frequent itemsets
    """
    if len(rules) == 0:
        return
    
    print(f"\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Confidence vs Lift
    top_rules = rules.head(30)
    scatter = axes[0].scatter(top_rules['confidence'], top_rules['lift'], 
                             c=top_rules['support'], alpha=0.6, s=50)
    axes[0].set_title('Association Rules: Confidence vs Lift')
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Lift')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Support')
    
    # Plot 2: Top rules by lift
    top_10_rules = rules.head(10)
    rule_names = [f"Rule {i+1}" for i in range(len(top_10_rules))]
    axes[1].barh(rule_names, top_10_rules['lift'], alpha=0.7)
    axes[1].set_title('Top 10 Rules by Lift')
    axes[1].set_xlabel('Lift')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Execute streamlined market basket analysis.
    """
    # Load and clean data
    data = load_and_clean_data('Online_retail.xlsx')
    if data is None:
        return
    
    # Create basket matrix (reduced complexity)
    basket = create_basket_matrix(data, 'United Kingdom', min_freq=50, max_products=50)
    
    if basket.empty:
        print("❌ No data for analysis")
        return
    
    # Perform analysis
    frequent_itemsets, rules = analyze_market_basket(
        basket, 
        min_support=0.05,  # Higher threshold for speed
        min_confidence=0.4,
        min_lift=1.2
    )
    
    # Display results
    display_results(frequent_itemsets, rules, top_n=15)
    
    # Create simple visualizations
    create_visualizations(rules, frequent_itemsets)
    
    # Final summary
    print(f"\n=== Summary ===")
    print(f"Transactions analyzed: {len(basket)}")
    print(f"Products analyzed: {basket.shape[1]}")
    print(f"Frequent itemsets: {len(frequent_itemsets)}")
    print(f"Association rules: {len(rules)}")
    
    if len(rules) > 0:
        print(f"\n🎯 Key Recommendations:")
        print(f"• Use top 5 rules for cross-selling recommendations")
        print(f"• Focus on rules with lift > 1.5 for product bundling")
        print(f"• Implement high-confidence rules for inventory optimization")
    
    print(f"\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()