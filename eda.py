import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    print("Shape of dataset:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDescription:\n", df.describe())
    
    # Top 15 Ad Groups by Impressions
    top_15 = df.groupby('Ad Group')['Impressions'].sum().sort_values(ascending=False).head(15).index
    filtered_df = df[df['Ad Group'].isin(top_15)]
    
    plt.figure(figsize=(12,6))
    sns.barplot(data=filtered_df, x='Ad Group', y='Impressions', estimator=sum)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 15 Ad Groups by Impressions")
    plt.tight_layout()
    plt.show()
    
    # Correlation Heatmap
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Matrix")
    plt.show()
    
    # Numerical Feature Distribution
    num_cols = ['Impressions', 'Clicks', 'CTR', 'Conversions', 'Conv Rate', 
                'Cost', 'CPC', 'Revenue', 'Sale Amount', 'P&L']
    df[num_cols].hist(bins=20, figsize=(16,12), edgecolor='black')
    plt.suptitle('Distribution of Numerical Features', fontsize=16)
    plt.show()
    
    # Categorical Feature Distribution
    cat_cols = ['Ad Group', 'Month']
    for col in cat_cols:
        plt.figure(figsize=(8,4))
        sns.countplot(data=df, x=col, palette='Set2')
        plt.title(f'Count Plot of {col}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    
    # Revenue by Ad Group
    revenue_by_adgroup = df.groupby('Ad Group')['Revenue'].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12,6))
    sns.barplot(x=revenue_by_adgroup.values, y=revenue_by_adgroup.index, palette='viridis')
    plt.title('Top 10 Ad Groups by Total Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel('Ad Group')
    plt.show()
    
    # Revenue vs Cost Scatter
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='Cost', y='Revenue', hue='Ad Group', alpha=0.7)
    plt.title('Revenue vs. Cost')
    plt.xlabel('Ad Cost')
    plt.ylabel('Revenue')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = pd.read_csv("data/final_shop_6modata.csv")
    perform_eda(df)

if __name__ == "__main__":
    main()
