from behaviour_to_morphology.lor_analise import *
from behaviour_to_morphology.parsing import *

def save_behavior_data(fish_data, output_behavior_file):
    behavior_data = []
    for fish_id, fish in fish_data.items():
        behavior_data.append([
            fish_id, fish.exp_time, fish.speed,
            fish.lateralization_normal, fish.lateralization_outliers,
            fish.lateralization_all,
            fish.uk_snr, fish.uk_std_dev, fish.uk_iqr,
            fish.drum_speed_clockwise,
            fish.drum_speed_counterclockwise
        ])


    behavior_df = pd.DataFrame(behavior_data, columns=[
        'Fish ID', 'Experiment Time', 'Speed',
        'Lateralization Normal', 'Lateralization Outliers',
        'Lateralization All',
        'UK_SNR', 'UK_Standard Deviation', 'UK_IQR',
        'Drum Speed Clockwise', 'Drum Speed Counterclockwise'
    ])
    behavior_df.to_excel(output_behavior_file, index=False)



def update_fish_data_with_drum(fish_data, drum_data):
    updated_fish_data = {}
    for fish_id, fish in fish_data.items():
        if fish_id in drum_data:
            drum_speed_clockwise, drum_speed_counterclockwise = \
            drum_data[fish_id]
            fish.set_drum_speed(drum_speed_clockwise,
                                drum_speed_counterclockwise)
            updated_fish_data[fish_id] = fish
        else:
            print(f"No complete data found for fish ID: {fish_id}")
    return updated_fish_data

def main():
    data_folder = 'Seq'
    output_folder = 'Graphics'
    results = analyse_all_data(data_folder)
    save_graphics(results, output_folder)

    drum_data_file = 'Data/Drum.xlsx'

    try:
        drum_data = read_drum_data(drum_data_file)
    except Exception as e:
        print(f"Error reading drum data: {e}")
        return

    updated_fish_data = update_fish_data_with_drum(results, drum_data)

    save_behavior_data(updated_fish_data,
                       'Data/Results/behavior_data_updated.xlsx')



if __name__ == "__main__":
    main()
