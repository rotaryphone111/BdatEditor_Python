# coding: utf-8
from test4 import calc_data_types
import numpy as np
b = np.fromfile('test1.bdat.part', dtype=np.uint8)
c = b.view(DataBuffer)
from Common.DataBuffer import DataBuffer
c = b.view(DataBuffer)
ItemCount = b.view(np.uint16)[8]
a = []
while i < ItemCount:
    a.append(d[int(176*i):int((i+1)*176)])
    i+= 1
    
i = 0
while i < ItemCount:
    a.append(d[int(176*i):int((i+1)*176)])
    i+= 1
    
d = c[itemOffset:itemOffset+itemlength].tobytes()
itemOffset = ItemTableOffset
ItemTableOffset + 21472
ItemTableOffset = b.view(np.uint16)[7]
itemOffset = ItemTableOffset
itemlength = 176 *269
d = c[itemOffset:itemOffset+itemlength].tobytes()
while i < ItemCount:
    a.append(d[int(176*i):int((i+1)*176)])
    i+= 1
    
a
dt = calc_data_types(c, memberTableOffset, memberCount)
memberTableOffset = b.view(np.uint16)[16]
memberCount = b.view(np.uint16)[17]
dt = calc_data_types(c, memberTableOffset, memberCount)
names = dt[0]
offsets = dt[2]
types = dt[1]
p = []
