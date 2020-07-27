import matplotlib.pyplot as plt
import csv
import argparse

parser = argparse.ArgumentParser("Plot loss file")
parser.add_argument('path', type=str, help='Path to the loss file')
parser.add_argument('--chart-title', dest='chart_title', type=str, help='The title, the chart should depict', default='Losses')


if __name__ == "__main__":
    args, _ = parser.parse_known_args()

    x=[]
    yg = []
    yd = []

    with open(args.path, 'r') as csvfile:
        plots = csv.reader(csvfile,delimiter=",")
        for count, row in enumerate(plots):
            x.append(count)
            yg.append(float(row[1]))
            yd.append(float(row[0]))

    plt.plot(x,yg,linewidth=0.5, color='red', label='generator')
    plt.plot(x,yd,linewidth=0.5, color='blue', label='discriminator')

    plt.legend(loc='best')

    plt.title(args.chart_title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.show()


