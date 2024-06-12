import pandas as pd
import pickle
from lor_analise import Fish, analyse_all_data


def read_drum_data(file_path):
    df = pd.read_excel(file_path)
    drum_data = {}
    for _, row in df.iterrows():
        fish_id = row.iloc[0]
        drum_speed_clockwise = row.iloc[1]
        drum_speed_counterclockwise = row.iloc[2]
        drum_data[fish_id] = (drum_speed_clockwise, drum_speed_counterclockwise)
    return drum_data


def read_morphology_data(file_path):
    df = pd.read_excel(file_path)
    morphology_data = {}
    for _, row in df.iterrows():
        fish_id = row.iloc[0]
        morphology_data[fish_id] = {
            'LLD': row.iloc[1], 'bc_LLD': row.iloc[2],
            'RLD': row.iloc[3], 'bc_RLD': row.iloc[4],
            'LVD': row.iloc[5], 'bc_LVD': row.iloc[6],
            'RVD': row.iloc[7], 'bc_RVD': row.iloc[8],
        }
    return morphology_data


def update_fish_data_with_drum_and_morphology(fish_data, drum_data,
                                              morphology_data):
    updated_fish_data = {}
    for fish_id, fish in fish_data.items():
        if fish_id in drum_data and fish_id in morphology_data:
            drum_speed_clockwise, drum_speed_counterclockwise = \
            drum_data[fish_id]
            fish.set_drum_speed(drum_speed_clockwise,
                                drum_speed_counterclockwise)

            morphology = morphology_data[fish_id]
            fish.set_morphology(morphology)

            updated_fish_data[fish_id] = fish
        else:
            print(f"No complete data found for fish ID: {fish_id}")
    return updated_fish_data


def save_combined_data(fish_data, output_behavior_file,
                       output_morphology_file):
    behavior_data = []
    morphology_data = []
    for fish_id, fish in fish_data.items():
        behavior_data.append([
            fish_id, fish.speed,
            fish.lateralization_normal, fish.lateralization_outliers,
            fish.lateralization_all,
            fish.uk_std_dev,
            fish.drum_speed_clockwise,
            fish.drum_speed_counterclockwise
        ])
        morphology = fish.morphology
        morphology_data.append([fish_id] + list(morphology.values()))

    behavior_df = pd.DataFrame(behavior_data, columns=[
        'Fish ID', 'Speed',
        'Lateralization Normal', 'Lateralization Outliers',
        'Lateralization All',
        'UK_Standard Deviation',
        'Drum Speed Clockwise', 'Drum Speed Counterclockwise'
    ])
    morphology_df = pd.DataFrame(morphology_data, columns=[
        'Fish ID', 'LLD', 'bc_LLD','RLD', 'bc_RLD',
        'LVD', 'bc_LVD','RVD', 'bc_RVD'
    ])

    behavior_df.to_excel(output_behavior_file, index=False)
    morphology_df.to_excel(output_morphology_file, index=False)


def main():
    fish_data_file = 'Data/fish_data.pkl'

    with open(fish_data_file, 'rb') as f:
        fish_data = pickle.load(f)

    drum_data_file = 'Data/Drum.xlsx'
    morphology_data_file = 'Data/Morfology.xlsx'

    try:
        drum_data = read_drum_data(drum_data_file)
    except Exception as e:
        print(f"Error reading drum data: {e}")
        return

    try:
        morphology_data = read_morphology_data(morphology_data_file)
    except Exception as e:
        print(f"Error reading morphology data: {e}")
        return

    updated_fish_data = update_fish_data_with_drum_and_morphology(
        fish_data, drum_data, morphology_data)

    save_combined_data(updated_fish_data,
                       'Data/Results/behavior_data_updated.xlsx',
                       'Data/Results/morphology_data.xlsx')

    for fish_id, fish in updated_fish_data.items():
        print(f"\n{fish}")
        print(f"Experiment Time: {fish.exp_time} seconds")
        print(f"Speed: {fish.speed:.4f} turns/second")
        print(f"Обычные значения: {fish.normal_values}")
        print(
            f"Латерализация (нормальные значения): {fish.lateralization_normal:.2f}")
        print(f"Выбросы: {fish.outliers}")
        print(
            f"Латерализация (выбросы): {fish.lateralization_outliers:.2f}")
        print(
            f"Все значения (нормальные + выбросы): {fish.all_values}")
        print(
            f"Латерализация (все значения): {fish.lateralization_all:.2f}")
        print(f"UK_SNR (УК Процент выбросов): {fish.uk_snr:.2f}")
        print(
            f"UK_Standard Deviation (Стандартное отклонение): {fish.uk_std_dev:.2f}")
        print(
            f"UK-Interquartile Range (Межквартильный размах, IQR): {fish.uk_iqr:.2f}")
        print(f"Drum Speed Clockwise: {fish.drum_speed_clockwise}")
        print(
            f"Drum Speed Counterclockwise: {fish.drum_speed_counterclockwise}")
        print(f"Morphology: {fish.morphology}")


if __name__ == "__main__":
    main()
