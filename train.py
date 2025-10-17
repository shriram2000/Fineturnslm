import pandas as pd
import numpy as np
from datetime import datetime
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class FashionRecommendationDataPrep:
    """
    Data Engineering Pipeline for Fashion Retail SLM Training
    Prepares personalized product recommendation training data
    """
    
    def __init__(self):
        self.customers = None
        self.products = None
        self.transactions = None
        self.stores = None
        self.discounts = None
        self.employees = None
        
    def load_data(self, data_path=''):
        """Load all CSV files from the dataset"""
        print("Loading datasets...")
        self.customers = pd.read_csv(f'{data_path}customers.csv')
        self.products = pd.read_csv(f'{data_path}products.csv')
        self.transactions = pd.read_csv(f'{data_path}transactions.csv')
        self.stores = pd.read_csv(f'{data_path}stores.csv')
        self.discounts = pd.read_csv(f'{data_path}discounts.csv')
        self.employees = pd.read_csv(f'{data_path}employees.csv')
        print("✓ All datasets loaded successfully")
        
    def clean_and_prepare_data(self):
        """Clean and prepare data for training"""
        print("\nCleaning and preparing data...")
        
        # Convert date columns
        self.transactions['Date'] = pd.to_datetime(self.transactions['Date'])
        self.customers['Date Of Birth'] = pd.to_datetime(self.customers['Date Of Birth'])
        
        # Calculate customer age
        current_date = self.transactions['Date'].max()
        self.customers['Age'] = ((current_date - self.customers['Date Of Birth']).dt.days / 365.25).astype(int)
        
        # Create age groups
        self.customers['Age Group'] = pd.cut(self.customers['Age'], 
                                              bins=[0, 25, 35, 45, 55, 100],
                                              labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Add product price tier
        self.products['Price Tier'] = pd.cut(self.products['Production Cost'], 
                                              bins=[0, 20, 40, 60, 100, 1000],
                                              labels=['Budget', 'Mid-Range', 'Premium', 'Luxury', 'Ultra-Luxury'])
        
        print("✓ Data cleaning completed")
        
    def create_customer_profiles(self):
        """Generate comprehensive customer shopping profiles"""
        print("\nGenerating customer profiles...")
        
        # Merge transactions with products and customers
        merged = self.transactions.merge(self.products, on='Product ID', how='left')
        merged = merged.merge(self.customers[['Customer ID', 'Gender', 'Age', 'Age Group', 'City', 'Country']], 
                              on='Customer ID', how='left')
        
        # Customer purchase statistics
        customer_stats = merged.groupby('Customer ID').agg({
            'Invoice ID': 'nunique',  # Number of orders
            'Quantity': 'sum',  # Total items purchased
            'Line Total': 'sum',  # Total spending
            'Date': ['min', 'max'],  # First and last purchase
            'Product ID': 'count'  # Number of line items
        }).reset_index()
        
        customer_stats.columns = ['Customer ID', 'Total Orders', 'Total Items', 
                                  'Total Spending', 'First Purchase', 'Last Purchase', 'Line Items']
        
        # Average order value
        customer_stats['Avg Order Value'] = customer_stats['Total Spending'] / customer_stats['Total Orders']
        
        # Favorite categories
        fav_categories = merged.groupby('Customer ID')['Category'].agg(
            lambda x: x.value_counts().index[0] if len(x) > 0 else None
        ).reset_index()
        fav_categories.columns = ['Customer ID', 'Favorite Category']
        
        # Favorite sub-categories (top 3)
        fav_subcats = merged.groupby('Customer ID')['Sub Category'].agg(
            lambda x: ', '.join(x.value_counts().head(3).index.tolist())
        ).reset_index()
        fav_subcats.columns = ['Customer ID', 'Top SubCategories']
        
        # Favorite colors
        fav_colors = merged.groupby('Customer ID')['Color'].agg(
            lambda x: ', '.join(x.value_counts().head(3).index.tolist()) if x.notna().any() else 'N/A'
        ).reset_index()
        fav_colors.columns = ['Customer ID', 'Favorite Colors']
        
        # Favorite sizes
        fav_sizes = merged.groupby('Customer ID')['Size'].agg(
            lambda x: ', '.join(x.value_counts().head(2).index.tolist()) if x.notna().any() else 'N/A'
        ).reset_index()
        fav_sizes.columns = ['Customer ID', 'Favorite Sizes']
        
        # Purchase frequency
        customer_stats['Purchase Frequency (days)'] = (
            (customer_stats['Last Purchase'] - customer_stats['First Purchase']).dt.days / 
            customer_stats['Total Orders']
        ).fillna(0)
        
        # Merge all profiles
        profiles = customer_stats.merge(fav_categories, on='Customer ID', how='left')
        profiles = profiles.merge(fav_subcats, on='Customer ID', how='left')
        profiles = profiles.merge(fav_colors, on='Customer ID', how='left')
        profiles = profiles.merge(fav_sizes, on='Customer ID', how='left')
        profiles = profiles.merge(self.customers[['Customer ID', 'Gender', 'Age', 'Age Group', 'City', 'Country']], 
                                 on='Customer ID', how='left')
        
        print(f"✓ Created profiles for {len(profiles)} customers")
        return profiles
    
    def generate_training_prompts(self, profiles, num_samples=1000):
        """
        Generate training prompts in instruction-tuning format
        Format: Customer profile -> Product recommendations
        """
        print("\nGenerating training prompts for SLM...")
        
        training_data = []
        
        # Get customer purchase history
        merged = self.transactions.merge(self.products[['Product ID', 'Category', 'Sub Category', 
                                                         'Description EN', 'Color', 'Sizes']], 
                                        on='Product ID', how='left')
        
        for idx, customer in profiles.head(num_samples).iterrows():
            cust_id = customer['Customer ID']
            
            # Get customer's purchase history
            cust_transactions = merged[merged['Customer ID'] == cust_id].sort_values('Date')
            
            if len(cust_transactions) < 2:
                continue
            
            # Split: Use 80% for context, 20% for target recommendations
            split_point = int(len(cust_transactions) * 0.8)
            history = cust_transactions.iloc[:split_point]
            target = cust_transactions.iloc[split_point:]
            
            if len(history) == 0 or len(target) == 0:
                continue
            
            # Create instruction prompt
            instruction = self._create_instruction_prompt(customer, history)
            
            # Create target recommendations
            response = self._create_response(target)
            
            training_data.append({
                'instruction': instruction,
                'response': response,
                'customer_id': cust_id
            })
        
        print(f"✓ Generated {len(training_data)} training samples")
        return training_data
    
    def _create_instruction_prompt(self, customer, history):
        """Create instruction prompt from customer profile and history"""
        
        prompt = f"""Customer Profile:
- Gender: {customer['Gender']}
- Age: {customer['Age']} years ({customer['Age Group']})
- Location: {customer['City']}, {customer['Country']}
- Total Orders: {customer['Total Orders']}
- Total Spending: ${customer['Total Spending']:.2f}
- Average Order Value: ${customer['Avg Order Value']:.2f}
- Favorite Category: {customer['Favorite Category']}
- Preferred Subcategories: {customer['Top SubCategories']}
- Favorite Colors: {customer['Favorite Colors']}
- Preferred Sizes: {customer['Favorite Sizes']}

Recent Purchase History:
"""
        
        # Add recent purchases (last 5)
        for idx, row in history.tail(5).iterrows():
            prompt += f"- {row['Category']} > {row['Sub Category']}: {row['Description EN']} ({row['Color']}, Size: {row['Size']}) - ${row['Unit Price']:.2f}\n"
        
        prompt += "\nBased on this customer's profile and shopping history, suggest 5 personalized product recommendations:"
        
        return prompt
    
    def _create_response(self, target):
        """Create response with recommended products"""
        
        response = "Recommended Products:\n\n"
        
        for idx, (i, row) in enumerate(target.head(5).iterrows(), 1):
            response += f"{idx}. {row['Category']} - {row['Sub Category']}\n"
            response += f"   Description: {row['Description EN']}\n"
            response += f"   Color: {row['Color']}, Available Sizes: {row['Sizes']}\n"
            response += f"   Price: ${row['Unit Price']:.2f}\n"
            response += f"   Why: Matches your preference for {row['Category']} items"
            if row['Color'] != 'N/A':
                response += f" in {row['Color']}"
            response += "\n\n"
        
        return response.strip()
    
    def create_product_catalog_embeddings(self):
        """Create structured product catalog for recommendations"""
        print("\nCreating product catalog...")
        
        catalog = []
        for _, product in self.products.iterrows():
            catalog.append({
                'product_id': product['Product ID'],
                'category': product['Category'],
                'subcategory': product['Sub Category'],
                'description': product['Description EN'],
                'color': product['Color'],
                'sizes': product['Sizes'],
                'price': product['Production Cost'],
                'price_tier': product['Price Tier']
            })
        
        print(f"✓ Created catalog with {len(catalog)} products")
        return catalog
    
    def export_training_data(self, training_data, output_path='training_data.jsonl'):
        """Export training data in JSONL format for SLM fine-tuning"""
        print(f"\nExporting training data to {output_path}...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in training_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✓ Exported {len(training_data)} training samples")
        
    def generate_summary_statistics(self, profiles, training_data):
        """Generate summary statistics for the dataset"""
        print("\n" + "="*70)
        print("DATASET SUMMARY STATISTICS")
        print("="*70)
        
        print(f"\n📊 Data Overview:")
        print(f"   Total Customers: {len(self.customers):,}")
        print(f"   Total Products: {len(self.products):,}")
        print(f"   Total Transactions: {len(self.transactions):,}")
        print(f"   Total Stores: {len(self.stores):,}")
        print(f"   Date Range: {self.transactions['Date'].min()} to {self.transactions['Date'].max()}")
        
        print(f"\n👥 Customer Demographics:")
        print(f"   Gender Distribution:")
        for gender, count in self.customers['Gender'].value_counts().items():
            print(f"      {gender}: {count:,} ({count/len(self.customers)*100:.1f}%)")
        
        print(f"\n🛍️ Purchase Behavior:")
        print(f"   Average Orders per Customer: {profiles['Total Orders'].mean():.2f}")
        print(f"   Average Spending per Customer: ${profiles['Total Spending'].mean():.2f}")
        print(f"   Average Order Value: ${profiles['Avg Order Value'].mean():.2f}")
        
        print(f"\n📦 Product Categories:")
        top_categories = self.products['Category'].value_counts().head(5)
        for cat, count in top_categories.items():
            print(f"   {cat}: {count:,} products")
        
        print(f"\n🎓 Training Data:")
        print(f"   Total Training Samples: {len(training_data):,}")
        print(f"   Customers with Training Data: {len(set([d['customer_id'] for d in training_data])):,}")
        
        print("\n" + "="*70)


# Example Usage
if __name__ == "__main__":
    print("="*70)
    print("FASHION RETAIL SLM TRAINING PIPELINE")
    print("Personalized Product Recommendation System")
    print("="*70)
    
    # Initialize pipeline
    pipeline = FashionRecommendationDataPrep()
    
    # Step 1: Load data
    # pipeline.load_data('path/to/your/data/')  # Specify your data path
    
    # For demonstration, showing the workflow:
    print("\n📋 WORKFLOW STEPS:")
    print("\n1. Load Data:")
    print("   pipeline.load_data('path/to/kaggle/dataset/')")
    
    print("\n2. Clean and Prepare:")
    print("   pipeline.clean_and_prepare_data()")
    
    print("\n3. Create Customer Profiles:")
    print("   profiles = pipeline.create_customer_profiles()")
    
    print("\n4. Generate Training Data:")
    print("   training_data = pipeline.generate_training_prompts(profiles, num_samples=5000)")
    
    print("\n5. Export for Training:")
    print("   pipeline.export_training_data(training_data, 'fashion_recommendations_train.jsonl')")
    
    print("\n6. Create Product Catalog:")
    print("   catalog = pipeline.create_product_catalog_embeddings()")
    
    print("\n7. View Statistics:")
    print("   pipeline.generate_summary_statistics(profiles, training_data)")
    
    print("\n" + "="*70)
    print("💡 NEXT STEPS FOR SLM TRAINING:")
    print("="*70)
    print("""
1. Fine-tune a Small Language Model (Recommended: GPT-2, Llama-2-7B, Mistral-7B)
   - Use the generated JSONL file with instruction-tuning
   - Libraries: Hugging Face Transformers, PEFT (LoRA), TRL
   
2. Training Configuration:
   - Model: microsoft/phi-2 or meta-llama/Llama-2-7b-chat-hf
   - Method: LoRA fine-tuning (efficient for small datasets)
   - Batch size: 4-8
   - Learning rate: 2e-4
   - Epochs: 3-5
   
3. Inference:
   - Load fine-tuned model
   - Input: Customer profile + purchase history
   - Output: Top 5 personalized product recommendations
   
4. Evaluation Metrics:
   - Hit Rate @K
   - NDCG (Normalized Discounted Cumulative Gain)
   - Customer satisfaction scores
    """)
    print("="*70)
