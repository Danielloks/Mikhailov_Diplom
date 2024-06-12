import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    behavior_df = pd.read_excel('Data/Results/behavior_data_updated.xlsx')
    morphology_df = pd.read_excel('Data/Results/morphology_data.xlsx')
    df = pd.merge(behavior_df, morphology_df, on='Fish ID')
    df.to_excel('Data/Results/combined_data.xlsx')

    P_columns = [
        'Speed', 'Lateralization Normal', 'Lateralization Outliers', 'Lateralization All',
        'UK_Standard Deviation', 'Drum Speed Clockwise', 'Drum Speed Counterclockwise'
    ]
    M_columns = ['LLD', 'bc_LLD', 'RLD', 'bc_RLD', 'LVD', 'bc_LVD', 'RVD', 'bc_RVD']

    correlation_weights = calculate_correlation_weights(df, P_columns, M_columns)
    correlation_weights.to_excel('Data/Results/correlation_weights.xlsx')

    models = train_regression_models(df, P_columns, M_columns)
    save_regression_coefficients(models, P_columns, 'Data/Results/regression_coefficients.xlsx')
    predict_for_new_data(models, P_columns)

def calculate_correlation_weights(df, P_columns, M_columns):
    correlations = df[P_columns + M_columns].corr()
    correlation_weights = correlations.loc[P_columns, M_columns]
    return correlation_weights

def train_regression_models(df, P_columns, M_columns):
    models = {}
    for m_col in M_columns:
        X = df[P_columns]
        y = df[m_col]

        model = LinearRegression()
        model.fit(X, y)
        models[m_col] = model
        print(f"Coefficients for {m_col}: {model.coef_}")
    return models

def predict_for_new_data(models, P_columns):
    new_data_df = pd.read_excel('predict_from_behaviour.xlsx')

    if new_data_df.shape[0] < 1:
        print("Error: Not enough data in predict_from_behaviour.xlsx.")
        return

    new_data = new_data_df.iloc[0].to_dict()
    input_df = pd.DataFrame([new_data], columns=P_columns)

    print("\nPredictions for new data:")
    for m_col, model in models.items():
        prediction = model.predict(input_df[P_columns])
        print(f"Predicted {m_col}: {prediction[0]}")

def save_regression_coefficients(models, P_columns, output_file):
    coefficients_data = []
    for m_col, model in models.items():
        coefficients_data.append([m_col] + list(model.coef_))

    columns = ['M_Column'] + P_columns
    coefficients_df = pd.DataFrame(coefficients_data, columns=columns)
    coefficients_df.to_excel(output_file, index=False)

if __name__ == "__main__":
    main()
