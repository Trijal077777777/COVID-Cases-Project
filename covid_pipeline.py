import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("covid_19_data.csv")
print("shape:", df.shape)
print(df.dtypes)
print(df.head())
print(df.info())
print(df.describe(include="all"))


def cleancolumns(df_in):
    df_local = df_in.copy()
    df_local.columns = (
        df_local.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r'[^a-z0-9_]', '', regex=True)
    )
    return df_local

df = cleancolumns(df)
print("columns:", df.columns.tolist())


if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
    print("date parse failures:", df['date'].isna().sum())
else:
    print("WARNING: no date column found")


num_cols = ['confirmed', 'deaths', 'recovered', 'population']
for c in num_cols:
    if c in df.columns:
        
        df[c] = df[c].astype(str).str.replace(',', '').replace({'nan': np.nan, 'None': np.nan})
        df[c] = pd.to_numeric(df[c], errors='coerce')
print("dtypes after numeric coercion:\n", df.dtypes)


full_dups = df.duplicated()
print("full duplicated rows:", full_dups.sum())
if full_dups.any():
    df = df[~full_dups].copy()

subset_keys = ['date', 'country'] + (['state'] if 'state' in df.columns else [])
key_dups = df.duplicated(subset=subset_keys)
print("key duplicated rows:", key_dups.sum())

if key_dups.any():
    df = df.groupby(subset_keys, as_index=False).agg({
        'confirmed': 'max' if 'confirmed' in df.columns else 'first',
        'deaths': 'max' if 'deaths' in df.columns else 'first',
        'recovered': 'max' if 'recovered' in df.columns else 'first',
        'population': 'max' if 'population' in df.columns else 'first',
    }).reset_index(drop=True)
    print("shape after agg:", df.shape)

df = df.dropna(subset=['date', 'country']).copy()

group_keys = ['country'] + (['state'] if 'state' in df.columns else []) + ['date']
df = df.sort_values(by=group_keys).reset_index(drop=True)


cumulative_cols = [c for c in ['confirmed', 'deaths', 'recovered'] if c in df.columns]
if cumulative_cols:
    grp_keys = ['country'] + (['state'] if 'state' in df.columns else [])
    df[cumulative_cols] = df.groupby(grp_keys)[cumulative_cols].ffill().fillna(0)


if 'country' in df.columns:
    df['country'] = df['country'].astype(str).str.strip()
    mapping = {
        'Usa': 'United States',
        'Us': 'United States',
        'U.s.': 'United States',
        'Uk': 'United Kingdom',
    }
    df['country'] = df['country'].replace(mapping)


for c in cumulative_cols:
    decreases = df.groupby(['country'])[c].diff() < 0
    print(f"{c} decreases found:", decreases.sum())
    df[c] = df.groupby(['country'])[c].cummax()


if 'confirmed' in df.columns:
    df['daily_confirmed'] = df.groupby('country')['confirmed'].diff().fillna(0)
    df['daily_confirmed'] = df['daily_confirmed'].clip(lower=0)
    df['daily_confirmed_7d'] = (
        df.groupby('country')['daily_confirmed']
          .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )


if 'population' in df.columns and 'confirmed' in df.columns:
    df['cases_per_100k'] = (df['confirmed'] / df['population']) * 100000


country = 'United States' if 'United States' in df['country'].unique() else df['country'].iloc[0]
sub = df[df['country'] == country].set_index('date').sort_index()

plt.figure(figsize=(10, 5))
plt.plot(sub.index, sub['daily_confirmed'], label='Daily confirmed')
plt.plot(sub.index, sub['daily_confirmed_7d'], label='7-day avg')
plt.title(f'Daily confirmed cases - {country}')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()


df.to_csv('covid_cleaned.csv', index=False)


def clean_covid(df_raw):
    df_work = cleancolumns(df_raw)
    if 'date' in df_work.columns:
        df_work['date'] = pd.to_datetime(df_work['date'], errors='coerce', dayfirst=False)

    return df_work
