import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


#train and test data directory
data_dir='/home/supreeta/Documents/HiWi/Objekterkennung/Basedata/Training'
test_data_dir='/home/supreeta/Documents/HiWi/Objekterkennung/Basedata/Testing'


#data_dir = "../input/intel-image-classification/seg_train/seg_train/"
#test_data_dir = "../input/intel-image-classification/seg_test/seg_test"


#load the train and test data
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((600,800)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((600,800)),transforms.ToTensor()
]))

img, label = dataset[0]
print(img.shape,label)

print("Follwing classes are there : \n",dataset.classes)

def display_img(img,label):
    print(f"Label : {dataset.classes[label]}")
    plt.imshow(img.permute(1,2,0))

#display the first image in the dataset
display_img(*dataset[3])
plt.show()