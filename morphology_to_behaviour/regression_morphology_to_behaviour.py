import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    behavior_df = pd.read_excel('Data/behaviour.xlsx')
    morphology_df = pd.read_excel('Data/morphology.xlsx')
    df = pd.merge(behavior_df, morphology_df, on='Fish ID')

    P_columns = [
        'Drum Speed Clockwise', 'Drum Speed Counterclockwise',
        'Sum_V_Drums', 'Symmetry_V_Drums',
        'Speed in channel', 'Lateralization Normal',
        'Lateralization Outliers', 'Lateralization All',
        'UK_Standard Deviation'
    ]
    M_columns = [
        'LLD', 'RLD', 'LVD', 'RVD', 'LLD+LVD', 'RLD+RVD', 'All',
        'R/All', 'L/All', 'LVD+RVD',
        'LLD+RLD', 'V/All', 'Lat/All', 'bc_LLD', 'bc_RLD', 'bc_LVD',
        'bc_RVD', 'bc_LLD+LVD',
        'bc_RLD+RVD', 'bc_All', 'bc_R/All', 'bc_L/All', 'bc_LVD+RVD',
        'bc_LLD+RLD', 'bc_V/All',
        'bc_Lat/All'
    ]

    correlation_weights = calculate_correlation_weights(df, M_columns,
                                                        P_columns)
    correlation_weights.to_excel(
        'Data/Results/correlation_weights.xlsx')

    X = df[M_columns]
    y = df[P_columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    models = train_regression_models(X_train, y_train, P_columns)
    evaluate_models(models, X_test, y_test, P_columns)

    predict_for_new_data(models, M_columns)


def calculate_correlation_weights(df, X_columns, Y_columns):
    correlations = df[X_columns + Y_columns].corr()
    correlation_weights = correlations.loc[X_columns, Y_columns]
    return correlation_weights


def train_regression_models(X_train, y_train, P_columns):
    models = {}

    for p_col in P_columns:
        y = y_train[p_col]
        model = LinearRegression()
        model.fit(X_train, y)
        models[p_col] = model
        print(f"Coefficients for {p_col}: {model.coef_}")

    return models


def evaluate_models(models, X_test, y_test, P_columns):
    for p_col in P_columns:
        model = models[p_col]
        y_true = y_test[p_col]
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"\nEvaluation for {p_col}:")
        print(f"Mean Squared Error: {mse}")
        print(f"R^2 Score: {r2}")


def predict_for_new_data(models, M_columns):
    new_data_df = pd.read_excel('predict_from_morphology.xlsx', header=0)

    if new_data_df.shape[0] < 1:
        print("Error: Not enough data in predict_from_morphology.xlsx.")
        return

    new_data = new_data_df.iloc[0].to_dict()
    input_df = pd.DataFrame([new_data], columns=M_columns)

    print("\nPredictions for new data:")
    for p_col, model in models.items():
        prediction = model.predict(input_df[M_columns])
        print(f"Predicted {p_col}: {prediction[0]}")



# Запуск основной программы
if __name__ == "__main__":
    main()
