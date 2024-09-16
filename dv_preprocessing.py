import os
import struct
import sys
from PIL import Image, ImageDraw
import imageio
import numpy as np
import numbers
from skimage.filters.rank import entropy
from skimage.morphology import disk
import math
from tkinter import Tk
from tkinter.filedialog import askdirectory


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def strftimefloat(secs):
    total_secs, millisecs = int(secs), int((secs - int(secs))*1000)
    hours, remainder = divmod(total_secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return "%02d:%02d:%02d.%03d"%(hours,minutes,seconds,millisecs)

def save_im(offset, factor, im): 
    im = Image.eval(im, lambda x:x*factor)
    #im = im.convert('L')
    return im
    
def dv_loader(filename):
    f = open(filename, 'rb')
    data = f.read()
    f.close
    
    headerSize = struct.unpack_from('<I', data, 92)[0]
    
    
    # endian-ness test
    if not struct.unpack_from("<H", data, 96)[0] == 0xc0a0:
        print("unsupported endian-ness")
        exit(1)
        
    imageWidth=struct.unpack_from("<I", data, 0)[0]
    imageHeight=struct.unpack_from("<I", data, 4)[0]
    numImages=struct.unpack_from("<I", data, 8)[0]
    pixelType=struct.unpack_from("<I", data, 12)[0]
    
    pixelWidth=struct.unpack_from("<f", data, 40)[0]
    pixelHeight=struct.unpack_from("<f", data, 44)[0]
    pixelDepth=struct.unpack_from("<f", data, 48)[0]
    dvInterleave=struct.unpack_from("n", data, 182)[0] # 0 is not interleaved, 1 is interleaved
    NWL=struct.unpack_from("<H", data, 196)[0]
    
    numChannels=NWL
    imageDataOffset=1024+headerSize
    
    if pixelType < 0 or pixelType > 7: #pixelType != 6
        print("unsupported pixel type")
        exit(1)
        
    headerNumInts=struct.unpack_from("<H", data, 128)[0]
    headerNumFloats=struct.unpack_from("<H", data, 130)[0]
    sectionSize = 4*(headerNumFloats+headerNumInts)
    sections = headerSize/sectionSize
    if (sections < numImages):
        print("number of sections is less than the number of images")
        exit(1)
    sections = numImages
    elapsed_times = [[struct.unpack_from("<f", data, i*sectionSize+k*4)[0] for k in range(int(sectionSize/4))][25] for i in range(sections)]
    
    elapsed_times = [strftimefloat(s) for s in elapsed_times]
    
    offset = imageDataOffset
    size = imageWidth*imageHeight*2
    totalSize = imageWidth*imageHeight*2*numImages+imageDataOffset
    stack = []
    for i in range(int(numImages)):
        im = Image.frombuffer("I;16", [imageWidth,imageHeight], data[offset:offset+size],'raw','I;16',0,1)
        stack.append(save_im(offset, 1, im))
        offset+=size
    channelStack = []
    impc=int(numImages/numChannels)
    if dvInterleave==1:
        for i in range(numChannels):
            channelStack.append(np.stack(stack[i::numChannels],axis=0))
    elif dvInterleave==2:
        for i in range(numChannels):
            channelStack.append(np.stack(stack[impc*i:impc+impc*i],axis=0))
    else:
        for i in range(numChannels):
            channelStack.append(np.stack(stack[i*impc:(i+1)*impc-1],axis=0))
    return channelStack

def normalize_stack(currentStack,ADJUST_FLAG):
    normStack = []
    if ADJUST_FLAG == 1:
        cf=[0.6,0.2,0.04]
    else:
        cf=[1,1,1]
    for i in range(len(currentStack)):
        currentChannel = np.array(currentStack[i])
        currentChannel = cf[i]*currentChannel*(math.pow(2,8)-1)/math.pow(2,15)
        currentChannel = currentChannel.astype(dtype=np.uint8)
        normStack.append(np.stack(currentChannel,axis=0))
    return normStack

def sum_stack(currentStack):
    sumStack = []
    for i in range(len(currentStack)):
        currentChannel = np.array(currentStack[i])
        currentChannel = np.sum(currentChannel,0)/len(currentChannel)
        currentChannel = currentChannel.astype(dtype=np.uint16)
        sumStack.append(np.stack(currentChannel,axis=0))
    return sumStack
        
    
def crop_rectangle(currentStack, x, y, length_x, length_y):
    if not isinstance(x + y + length_x + length_y,numbers.Integral):
        print("Rectangle dimensions invalid!")
        exit(1)
    croppedStack = np.zeros((length_x,length_y,len(currentStack)),dtype=np.uint8)
    for i in range(len(currentStack)):
        currentChannel = currentStack[i]
        currentChannel = currentChannel[x:x+length_x,y:y+length_y]
        croppedStack[:,:,i]=currentChannel
        ImageFinal = Image.fromarray(croppedStack)
    return croppedStack

def calc_entropy(image_final, dim):
    image = np.sum(image_final,2)/dim
    image = image.astype(dtype=np.uint8)
    ent = entropy(image, disk(10))
    ent_num = np.mean(ent)
    return ent_num
    
def image_cut(currentStack, num, filename, dir_above, dir_below, runID):
    x = currentStack.shape[0]
    y = currentStack.shape[1]
    dim = currentStack.shape[2]
    x_inc = int(x/num)
    y_inc = int(y/num)
    x1, x2 = 0, x_inc
    y1, y2 = 0, y_inc
    for j in range(0,num**2):
        cut_image = np.zeros((x2,x2,dim),dtype=np.uint8)
        cut_image = currentStack[x1:x2,y1:y2,:]
        if x2==x:
            x1, x2 = 0, x_inc
            y1 += y_inc
            y2 += y_inc
        else:
            x1 += x_inc
            x2 += x_inc
            
        image_final = Image.fromarray(cut_image)
        ent_num = calc_entropy(image_final, dim)
        if ent_num > 0.5:
            image_final.save(dir_above + '\\' + filename + runID + str(j) + '.tif')
        else:
            image_final.save(dir_below + '\\' + filename + runID + str(j) + '.tif')
    return
    
    

input_path = askdirectory(title='Input folder')
dlist = listdir_fullpath(os.path.abspath(input_path))
dir_above = os.path.abspath(input_path[:-5] + 'above')
dir_below = os.path.abspath(input_path[:-5] + 'below')
index = os.listdir(input_path) 
runID = 'runDQ20240510'

for i in range(0,np.shape(dlist)[0]):
    currentStack=dv_loader(dlist[i])
    currentStack=sum_stack(currentStack[:-1])
    currentStack=normalize_stack(currentStack,0)
    currentStack=crop_rectangle(currentStack, 120, 120, 1800, 1800)
    image_cut(currentStack, 1, index[i][:-3], dir_above, dir_below, runID)


