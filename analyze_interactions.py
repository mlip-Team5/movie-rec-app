import pandas as pd
import matplotlib.pyplot as plt

def analyze_interactions(csv_file='user_movie_interactions.csv'):
    df = pd.read_csv(csv_file)
    print('Basic info:')
    print(df.info())
    print('\nSample rows:')
    print(df.head())
    print('\nDescriptive statistics:')
    print(df['score'].describe())
    print('\nNumber of unique users:', df['userid'].nunique())
    print('Number of unique movies:', df['movieid'].nunique())
    print('Total interactions:', len(df))
    print('\nTop 10 most-watched movies:')
    print(df.groupby('movieid')['score'].sum().sort_values(ascending=False).head(10))
    print('\nTop 10 most active users:')
    print(df.groupby('userid')['score'].sum().sort_values(ascending=False).head(10))
    # Plot score distribution
    plt.figure(figsize=(8,4))
    df['score'].hist(bins=50)
    plt.title('Distribution of Watch Counts per User-Movie Pair')
    plt.xlabel('Watch Count (score)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('score_distribution.png')
    print('\nScore distribution plot saved as score_distribution.png')

if __name__ == '__main__':
    analyze_interactions()
