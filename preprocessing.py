import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def encode_columns(df):
    le_adgroup = LabelEncoder()
    le_month = LabelEncoder()
    
    df['Ad_Group_Encoded'] = le_adgroup.fit_transform(df['Ad Group'])
    df['Month_Encoded'] = le_month.fit_transform(df['Month'])
    
    return df, le_adgroup, le_month

def main():
    df = load_data("data/final_shop_6modata.csv")
    df, le_adgroup, le_month = encode_columns(df)
    print("Data loaded and encoded successfully")
    print(df.head())

if __name__ == "__main__":
    main()
