import re
import matplotlib.pyplot as plt


def find_loss(file):
    with open(file, 'r') as f:
        text = f.read()
        losses = re.findall(r'loss: ([0-9]+.[0-9]+) - val_loss: ([0-9]+.[0-9]+)', text)
        print(losses)
        return losses


def plot_loss(losses):
    axis_x = [i+1 for i in range(len(losses))]
    train_loss = [float(x[0]) for x in losses]
    cv_loss = [float(x[1]) for x in losses]
    plt.plot(axis_x, train_loss, color='blue')
    plt.plot(axis_x, cv_loss, color='red')
    plt.show()


if __name__ == "__main__":
    losses = find_loss('/home/htt/Desktop/NYT-Multi/nohup.out')
    plot_loss(losses)
