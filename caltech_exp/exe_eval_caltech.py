from DLBio.helpers import search_rgx
from os.path import join
import matplotlib.pyplot as plt

BASE_FOLDER = 'exp_data'
RGX = r'caltech(_[01 ]+|)'
IMAGE_FOLDER = 'caltech_exp'


def run():
    folders_ = search_rgx(RGX, BASE_FOLDER)
    assert folders_
    logs_ = {}
    for folder in folders_:
        logs_[folder] = load_log(folder)
        keys_ = logs_[folder].keys()

    for key in keys_:
        plt.figure()
        for name, log in logs_.items():
            plt.plot(log[key], label=name)

        plt.grid()
        plt.legend()
        plt.tight_layout()

        plt.savefig(join(IMAGE_FOLDER, f'{key}.png'))
        plt.close()


def load_log(folder):
    out = {}
    with open(join(BASE_FOLDER, folder, 'log.txt'), 'r') as file:
        for line in file.readlines():
            for c in ['{', '}', '\n', '"']:
                line = line.replace(c, '')
            for entry in line.split(','):
                key, value = entry.split(':')
                try:
                    out[key].append(float(value))
                except:
                    out[key] = [float(value)]

    return out


if __name__ == '__main__':
    run()
