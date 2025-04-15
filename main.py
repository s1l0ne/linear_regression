import pandas as pd


file_test_data = 'data/house_prices/test.csv'
file_train_data = 'data/house_prices/train.csv'


if __name__ == '__main__':
    train_data = pd.read_csv(file_train_data)
    test_data = pd.read_csv(file_test_data)

    print(train_data.head())

    features = ['GrLivArea', 'GarageCars', 'FullBath']
    target = 'SalePrice'

    train_data = train_data[features + [target]]
    test_data = test_data[['Id'] + features]

    print(train_data.head())
    print(test_data.head())
