def add_features(df, is_training=True):
    # Basic features
    df['Conv Rate'] = df['Conversions'] / (df['Clicks'] + 1e-6)
    df['CPC'] = df['Cost'] / (df['Clicks'] + 1e-6)
    df['CPM'] = (df['Cost'] / (df['Impressions'] + 1e-6)) * 1000

    # Always compute KPIs for training
    df['ROI'] = (df['Revenue'] - df['Cost']) / (df['Cost'] + 1e-6)
    df['Profit_Margin'] = (df['Revenue'] - df['Cost']) / (df['Revenue'] + 1e-6)
    df['Revenue_per_Click'] = df['Revenue'] / (df['Clicks'] + 1e-6)
    df['Revenue_per_Conversion'] = df['Revenue'] / (df['Conversions'] + 1e-6)

    df.fillna(0, inplace=True)
    return df
