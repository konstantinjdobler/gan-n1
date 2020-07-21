import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data
import torch
import json


attribs = ['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows',
'Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin',
'Pointy_Nose','Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling','Straight_Hair','Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace',
'Wearing_Necktie','Young']

with open('./list_attr_celeba.txt', 'r') as f:
    data = f.read()
    data = data.strip().split('\n')
attrs = [( line.split()[0], list(map(float, line.split()[1:]))) for line in data[2:]]

def boi(condition_list):
    returni = {}
    for condition in range(len(condition_list)):
        returni[attribs[condition]] = condition_list[condition]
    return returni

jsonified ={}
for at in attrs:
    jsonified[at[0]] = boi(at[1])

with open('list_attr_celeba.json', 'w') as outfile:
    json.dump(jsonified, outfile, separators=(',', ':'))


