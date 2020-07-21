import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.utils.data as data
import torch

import os

import json

import numpy as np

from copy import deepcopy
from PIL import Image

class ImageFeatureFolder(dset.ImageFolder):
    def __init__(self, image_root, attribute_file, transform):
        super(ImageFeatureFolder, self).__init__(
            root=image_root, transform=transform)

        with open(attribute_file, 'r') as f:
            data = f.read()
        data = data.strip().split('\n')
        self.attrs = torch.FloatTensor(
            [list(map(float, line.split()[1:])) for line in data[2:]])

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)

        return img, self.attrs[index]


def pil_loader(path):
    imgExt = os.path.splitext(path)[1]
    if imgExt == ".npy":
        img = np.load(path)[0]
        return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)

    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def getClassStats(inputDict, className):

    outStats = {}

    for item in inputDict:

        val = item[className]
        if val not in outStats:
            outStats[val] = 0

        outStats[val] += 1

    return outStats


def buildDictStats(inputDict, classList):

    locStats = {"total": len(inputDict)}

    for cat in classList:

        locStats[cat] = getClassStats(inputDict, cat)

    return locStats


def buildKeyOrder(shiftAttrib,
                  shiftAttribVal,
                  stats=None):
    r"""
    If the dataset is labelled, give the order in which the attributes are given
    Args:
        - shiftAttrib (dict): order of each category in the category vector
        - shiftAttribVal (dict): list (ordered) of each possible labels for each
                                category of the category vector
        - stats (dict): if not None, number of representant of each label for
                        each category. Will update the output dictionary with a
                        "weights" index telling how each labels should be
                        balanced in the classification loss.
    Returns:
        A dictionary output[key] = { "order" : int , "values" : list of string}
    """

    MAX_VAL_EQUALIZATION = 10

    output = {}
    for key in shiftAttrib:
        output[key] = {}
        output[key]["order"] = shiftAttrib[key]
        output[key]["values"] = [None for i in range(len(shiftAttribVal[key]))]
        for cat, shift in shiftAttribVal[key].items():
            output[key]["values"][shift] = cat

    if stats is not None:
        for key in output:

            n = sum([x for key, x in stats[key].items()])

            output[key]["weights"] = {}
            for item, value in stats[key].items():
                output[key]["weights"][item] = min(
                    MAX_VAL_EQUALIZATION, n / float(value + 1.0))

    return output


class AttribDataset(data.Dataset):

    def __init__(self,
                 pathdb,
                 attribDictPath=None,
                 specificAttrib=None,
                 transform=None,
                 mimicImageFolder=False,
                 ignoreAttribs=False,
                 getEqualizer=False,
                 pathMask=None):
        r"""
        Args:
            - root (string): path to the directory containing the images
            - attribDictPath (string): path to a json file containing the images'
                                       specific attributes
            - specificAttrib (list of string): if not None, specify which attributes
                                                be selected
            - transform (torchvision.transforms): transformation to apply to the
                                                  loaded images.
            - mimicImageFolder (bool): set to True if the dataset is stored in the
                                      torchvision.datasets.ImageFolder format
            - ignoreAttribs (bool): set to True if you just want to use the attrib
                                    dict as a filter on images' name
        """

        self.totAttribSize = 0
        self.hasAttrib = attribDictPath is not None or mimicImageFolder
        self.pathdb = pathdb
        self.transform = transform
        self.shiftAttrib = None
        self.stats = None
        self.pathMask = None

        if attribDictPath:
            if ignoreAttribs:
                self.attribDict = None

                with open(attribDictPath, 'rb') as file:
                    tmpDict = json.load(file)
                self.listImg = [imgName for imgName in os.listdir(pathdb)
                                if (os.path.splitext(imgName)[1] in [".jpg",
                                                                     ".png", ".npy"] and imgName in tmpDict)]
            else:
                self.loadAttribDict(attribDictPath, pathdb, specificAttrib)

        elif mimicImageFolder:
            self.loadImageFolder(pathdb)
        else:
            self.attribDict = None
            self.listImg = [imgName for imgName in os.listdir(pathdb)
                            if os.path.splitext(imgName)[1] in [".jpg", ".png",
                                                                ".npy"]]

        if pathMask is not None:
            print("Path mask found " + pathMask)
            self.pathMask = pathMask
            self.listImg = [imgName for imgName in self.listImg
                            if os.path.isfile(os.path.join(pathMask,
                                                           os.path.splitext(imgName)[0] + "_mask.jpg"))]

        if len(self.listImg) == 0:
            raise AttributeError("Empty dataset")

        self.buildStatsOnDict()

        print("%d images found" % len(self))

    def __len__(self):
        return len(self.listImg)

    def hasMask(self):
        return self.pathMask is not None

    def buildStatsOnDict(self):

        if self.attribDict is None:
            return

        self.stats = {}
        for item in self.attribDict:

            for category, value in self.attribDict[item].items():

                if category not in self.stats:
                    self.stats[category] = {}

                if value not in self.stats[category]:
                    self.stats[category][value] = 0

                self.stats[category][value] += 1

    def loadAttribDict(self,
                       dictPath,
                       dbDir,
                       specificAttrib):
        r"""
        Load a dictionnary describing the attributes of each image in the
        dataset and save the list of all the possible attributes and their
        acceptable values.
        Args:
            - dictPath (string): path to a json file describing the dictionnary.
                                 If None, no attribute will be loaded
            - dbDir (string): path to the directory containing the dataset
            - specificAttrib (list of string): if not None, specify which
                                               attributes should be selected
        """

        self.attribDict = {}
        attribList = {}

        with open(dictPath, 'rb') as file:
            tmpDict = json.load(file)

            for fileName, attrib in tmpDict.items():

                if not os.path.isfile(os.path.join(dbDir, fileName)):
                    continue

                if specificAttrib is None:
                    self.attribDict[fileName] = deepcopy(attrib)
                else:
                    self.attribDict[fileName] = {
                        k: attrib[k] for k in specificAttrib}

                for attribName, attribVal in self.attribDict[fileName].items():
                    if attribName not in attribList:
                        attribList[attribName] = set()

                    attribList[attribName].add(attribVal)

        # Filter the attrib list
        self.totAttribSize = 0

        self.shiftAttrib = {}
        self.shiftAttribVal = {}

        for attribName, attribVals in attribList.items():

            if len(attribVals) == 1:
                continue

            self.shiftAttrib[attribName] = self.totAttribSize
            self.totAttribSize += 1

            self.shiftAttribVal[attribName] = {
                name: c for c, name in enumerate(attribVals)}

        # Img list
        self.listImg = list(self.attribDict.keys())

    def loadImageFolder(self, pathdb):
        r"""
        Load a dataset saved in the torchvision.datasets.ImageFolder format.
        Arguments:
            - pathdb: path to the directory containing the dataset
        """

        listDir = [dirName for dirName in os.listdir(pathdb)
                   if os.path.isdir(os.path.join(pathdb, dirName))]

        imgExt = [".jpg", ".png", ".JPEG"]

        self.attribDict = {}

        self.totAttribSize = 1
        self.shiftAttrib = {"Main": 0}
        self.shiftAttribVal = {"Main": {}}

        for index, dirName in enumerate(listDir):

            dirPath = os.path.join(pathdb, dirName)
            self.shiftAttribVal["Main"][dirName] = index

            for img in os.listdir(dirPath):

                if os.path.splitext(img)[1] in imgExt:
                    fullName = os.path.join(dirName, img)
                    self.attribDict[fullName] = {"Main": dirName}

        # Img list
        self.listImg = list(self.attribDict.keys())

    def __getitem__(self, idx):

        imgName = self.listImg[idx]
        imgPath = os.path.join(self.pathdb, imgName)
        img = pil_loader(imgPath)

        if self.transform is not None:
            img = self.transform(img)

        # Build the attribute tensor
        attr = [0 for i in range(self.totAttribSize)]

        if self.hasAttrib:
            attribVals = self.attribDict[imgName]
            for key, val in attribVals.items():
                baseShift = self.shiftAttrib[key]
                attr[baseShift] = self.shiftAttribVal[key][val]
        else:
            attr = [0]

        if self.pathMask is not None:
            mask_path = os.path.join(
                self.pathMask, os.path.splitext(imgName)[0] + "_mask.jpg")
            mask = pil_loader(mask_path)
            mask = transforms.Grayscale(1)(mask)
            mask = self.transform(mask)

            return img, torch.tensor(attr, dtype=torch.long), mask

        return img, torch.tensor(attr, dtype=torch.long)

    def getName(self, idx):

        return self.listImg[idx]

    def getTextDescriptor(self, idx):
        r"""
        Get the text descriptor of the idx th image in the dataset
        """
        imgName = self.listImg[idx]

        if not self.hasAttrib:
            return {}

        return self.attribDict[imgName]

    def getKeyOrders(self, equlizationWeights=False):
        r"""
        If the dataset is labelled, give the order in which the attributes are
        given
        Returns:
            A dictionary output[key] = { "order" : int , "values" : list of
            string}
        """

        if self.attribDict is None:
            return None

        if equlizationWeights:

            if self.stats is None:
                raise ValueError("The weight equalization can only be \
                                 performed on labelled datasets")

            return buildKeyOrder(self.shiftAttrib,
                                 self.shiftAttribVal,
                                 stats=self.stats)
        return buildKeyOrder(self.shiftAttrib,
                             self.shiftAttribVal,
                             stats=None)