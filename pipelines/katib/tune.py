import argparse
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--max_depth', type=int, default=3)
parser.add_argument('--subsample', type=float, default=1.0)
parser.add_argument('--start_date', type=str, default='2025-03-01')
parser.add_argument('--random_state', type=int, default=1234)

if __name__ == "__main__":
    args = parser.parse_args()
    df = pd.read_csv(f'/dataset/seoul-{args.start_date}.csv')  # 전처리된 데이터셋 경로    
    
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=args.random_state
    )
    
    X = df[['x1', 'x2', 'x3', 'x4']]
    y = df['y']
    
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    mae = -scores.mean()
    print("MAE=%f" % mae)
