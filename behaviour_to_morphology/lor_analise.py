import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import iqr
import pickle

class Fish:
    def __init__(self, fish_id, exp_time):
        self.fish_id = fish_id
        self.exp_time = exp_time
        self.fm_xp = None
        self.fm_yp = None
        self.lor_seq = None
        self.normal_values = None
        self.lateralization_normal = None
        self.outliers = None
        self.lateralization_outliers = None
        self.all_values = None
        self.lateralization_all = None
        self.speed = None
        self.uk_snr = None
        self.uk_std_dev = None
        self.uk_iqr = None
        self.drum_speed_clockwise = None
        self.drum_speed_counterclockwise = None
        self.morphology = None

    def set_fm_data(self, fm_xp, fm_yp, lor_seq):
        self.fm_xp = fm_xp
        self.fm_yp = fm_yp
        self.lor_seq = lor_seq

    def set_turns_data(self, normal_values, outliers, all_values,
                       lateralization_normal, lateralization_outliers,
                       lateralization_all):
        self.normal_values = normal_values
        self.lateralization_normal = lateralization_normal
        self.outliers = outliers
        self.lateralization_outliers = lateralization_outliers
        self.all_values = all_values
        self.lateralization_all = lateralization_all

    def set_speed(self, speed):
        self.speed = speed

    def calculate_uk_metrics(self):
        self.uk_snr = calculate_snr(self.outliers, self.all_values)
        self.uk_std_dev = calculate_standard_deviation(
            self.all_values)
        self.uk_iqr = calculate_iqr(self.all_values)

    def set_drum_speed(self, drum_speed_clockwise,
                       drum_speed_counterclockwise):
        self.drum_speed_clockwise = drum_speed_clockwise
        self.drum_speed_counterclockwise = drum_speed_counterclockwise

    def set_morphology(self, morphology):
        self.morphology = morphology

    def __str__(self):
        return f"Fish ID: {self.fish_id}, Experiment Time: {self.exp_time} seconds, Speed: {self.speed:.2f} turns/second"


def calculate_snr(outliers, all_values):
    sum_outliers = sum(abs(value) for value in outliers)
    total_values = sum(abs(value) for value in all_values)
    snr = (
                      sum_outliers / total_values) * 100 if total_values != 0 else 0
    return snr


def calculate_standard_deviation(all_values):
    std_dev = np.std(all_values)
    return std_dev


def calculate_iqr(all_values):
    q1 = np.percentile(all_values, 25)
    q3 = np.percentile(all_values, 75)
    iqr_value = q3 - q1
    return iqr_value


def read_excel_data(file_path):
    df = pd.read_excel(file_path)
    lor_sequences = df.values.tolist()
    experiment_time = len(lor_sequences) * 5
    combined_sequences = ''
    for seq in lor_sequences:
        for element in seq:
            if isinstance(element, str) and element in ['L', 'R',
                                                        '|']:
                combined_sequences += element
    return combined_sequences, experiment_time


def get_lor_func(lor_seq):
    lor_yp = [0]
    lor_xp = [0]
    x = 1
    for lyteral in lor_seq:
        if lyteral == 'L':
            lor_yp.append(-1)
            lor_xp.append(x)
        elif lyteral == 'R':
            lor_yp.append(1)
            lor_xp.append(x)
        x += 1
    lor_yp.append(0)
    lor_xp.append(x)
    return lor_xp, lor_yp


def sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def get_fm(yp):
    fc = 0
    x = 0
    fm_yp = []
    fm_xp = []
    for i, y in enumerate(yp):
        direction = sign(y)
        next_direction = sign(yp[i + 1]) if i + 1 < len(yp) else 0
        fc += 1
        if next_direction * direction <= 0:
            fm_xp.append(x)
            fm_yp.append(fc * direction)
            x += 1
            fc = 0
    return fm_xp, fm_yp


def analyse_lor_sequence(seq):
    xp, yp = get_lor_func(seq)
    fm_xp, fm_yp = get_fm(yp)
    return fm_xp, fm_yp


def analyse_all_data(data_folder):
    results = {}
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.xlsx') and not file_name.startswith(
                '~$'):
            file_path = os.path.join(data_folder, file_name)
            seq, exp_time = read_excel_data(file_path)
            move_count = len(seq)
            speed = move_count / exp_time if exp_time != 0 else 0
            fish_id = os.path.splitext(file_name)[0]
            fish = Fish(fish_id, exp_time)
            fish.set_speed(speed)
            fm_xp, fm_yp = analyse_lor_sequence(seq)
            fish.set_fm_data(fm_xp, fm_yp, seq)

            normal_values, outliers = calculate_outliers_and_normal(
                fm_yp)
            lateralization_normal = calculate_lateralization(
                normal_values)
            lateralization_outliers = calculate_lateralization(
                outliers)
            all_values = normal_values + outliers
            lateralization_all = calculate_lateralization(all_values)

            fish.set_turns_data(normal_values, outliers, all_values,
                                lateralization_normal,
                                lateralization_outliers,
                                lateralization_all)
            fish.calculate_uk_metrics()

            results[fish_id] = fish

    return results


def save_graphics(results, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for fish_id, fish in results.items():
        plt.figure(figsize=(10, 5))
        plt.plot(fish.fm_xp, fish.fm_yp, marker='o')
        plt.title(f'Fish ID: {fish_id}')
        plt.xlabel('Turn Number')
        plt.ylabel('Turn Count')
        plt.grid(True)

        uk_snr = fish.uk_snr
        uk_std_dev = fish.uk_std_dev
        uk_iqr = fish.uk_iqr

        plt.text(0.05, 0.95,
                 f'SNR: {uk_snr:.2f}\nSTD: {uk_std_dev:.2f}\nIQR: {uk_iqr:.2f}',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.5))

        plt.savefig(
            os.path.join(output_folder, f'{fish_id}_fm_lor.png'))
        plt.close()


def calculate_outliers_and_normal(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr_value = iqr(values)
    outliers = [y for y in values if (y < q1 - 1.5 * iqr_value) or (
                y > q3 + 1.5 * iqr_value)]
    normal_values = [y for y in values if
                     (q1 - 1.5 * iqr_value) <= y <= (
                                 q3 + 1.5 * iqr_value)]
    return normal_values, outliers


def calculate_lateralization(values):
    sum_left = sum(abs(y) for y in values if y < 0)
    sum_right = sum(y for y in values if y > 0)
    sum_all = sum(abs(y) for y in values)
    lateralization = (
                                 sum_right - sum_left) / sum_all if sum_all != 0 else 0
    return lateralization


def save_fish_data(results, output_file):
    data = []
    for fish_id, fish in results.items():
        data.append([
            fish_id, fish.exp_time, fish.speed,
            fish.lateralization_normal, fish.lateralization_outliers,
            fish.lateralization_all,
            fish.uk_snr, fish.uk_std_dev, fish.uk_iqr
        ])
    df = pd.DataFrame(data, columns=[
        'Fish ID', 'Experiment Time', 'Speed',
        'Lateralization Normal', 'Lateralization Outliers',
        'Lateralization All',
        'UK_SNR', 'UK_Standard Deviation', 'UK_IQR'
    ])
    df.to_excel(output_file, index=False)


def main():
    data_folder = 'Data/P-seq'
    output_folder = 'Graphics'
    results = analyse_all_data(data_folder)

    save_graphics(results, output_folder)
    save_fish_data(results,
                   'Data/behavior_data.xlsx')

    with open('Data/fish_data.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
