import json
import random


def crop(file, ratio=0.1):
    with open(file, 'r') as f:
        samples = f.readlines()
        if samples[-1] == '':
            samples = samples[:-1]
        edge = int(len(samples)*ratio+0.5)
        reservoir = samples[:edge]
        for i in range(edge, len(samples)):
            point = random.randint(0, i)
            if point < edge:
                reservoir[point] = samples[i]
    return reservoir


if __name__ == "__main__":
    file='train_data.json'
    ratio = 0.1
    samples = crop(file, ratio)
    open(file.split('.')[0]+'-'+str(ratio)+'.json', 'w').writelines(samples)
