import matplotlib.pyplot as plt
import csv

x=[]
yg = []
yd = []

with open('F:/GAN/src/fake_samples/improvedPGAN/losses5.txt', 'r') as csvfile:
    plots = csv.reader(csvfile,delimiter=",")
    for count, row in enumerate(plots):
        x.append(count)
        yg.append(float(row[1]))
        yd.append(float(row[0]))

plt.plot(x,yg,linewidth=0.5, color='red', label='generator')
plt.plot(x,yd,linewidth=0.5, color='blue', label='discriminator')

plt.legend(loc='best')

plt.title('Losses - 4x4')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.show()


