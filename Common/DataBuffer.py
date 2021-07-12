'''
Created on May 13, 2021

@author: kenji
'''

import numpy as np


class DataBuffer(np.ndarray):
    '''
    classdocs
    '''
    
    def __new__(cls, input_array, start = 0, length=False):
        obj = np.asarray(input_array).view(cls)
        obj.Position = start
        if not length:
            obj.Length = input_array.nbytes - start
        else:
            obj.Length = length 
        return obj

    def __array_finalize__(self, obj):
        self.Position = 0
    
    def ReadValue(self, index = 0, Dtype=np.uint8, use_position = False):
        if not use_position:
            return self[index: int(index + Dtype(1).itemsize)].view(Dtype)[0]
        else:
            value = self[self.Position + index: int(self.Position + index + Dtype(1).itemsize)].view(Dtype)[0]
            self.Position += Dtype(1).itemsize
            return value

    def ReadUInt8(self, index):
        result = self.view(np.uint8)[index]
        return result

    def ReadInt8(self, index):
        result = self.view(np.int8)[index]
        return result
    
    def ReadUInt16(self, index):
        result = self[index: index + 2].view(np.uint16)[0]
        return result

    def ReadInt16(self, index):
        result = self[index: index + 2].view(np.uint16)[0]
        return result

    def ReadUInt32(self, index):
        result = self[index: index + 4].view(np.uint32)[0]
        return result

    def ReadInt32(self, index):
        result = self[index: index + 4].view(np.int32)[0]
        return result

    def ReadUInt64(self, index):
        result = self[index: index + 8].view(np.uint64)[0]
        return result

    def ReadInt64(self, index):
        result = self[index: index + 8].view(np.int64)[0]
        return result
    
    def ReadSingle(self, index):
        result = self.view(np.float32)[int(index/4)]
        return result
    
    def ReadUTF8(self, index, length, use_position=False):
        if not use_position:
            result = self[index: index + length].tobytes().decode()
        else:
            result = self[self.Position: self.Position + length].tobytes().decode()
            self.Position += length
        return result
    
    def ReadUTF8Z(self, index, use_position = False):
        if not use_position:
            result = self[index:].tobytes().split(sep=b'\x00')[0].decode()
        else:
            result = self[self.Position + index:].tobytes().split(sep=b'\x00')[0].decode()
        return result

    def ReadBytes(self, index, length):
        return self[index:length].tobytes()
    
    def AppendValue(self, value, dtype):
        val_array = np.array([value], dtype=dtype).view(np.uint8)
        return np.append(self, val_array).view(DataBuffer)
    
    def AppendUInt8(self, value):
        val_array = np.array(value, dtype=np.uint8)
        return np.append(self, val_array).view(DataBuffer)
        
    def AppendInt8(self, value):
        val_array = np.array(value, dtype=np.int8)
        return np.append(self.view(np.int8), val_array).view(np.uint8).view(DataBuffer)

    def AppendUInt16(self, value):
        val_array = np.array(value, dtype=np.uint16)
        return np.append(self.view(np.uint16), val_array).view(np.uint8).view(DataBuffer)
        
    def AppendInt16(self, value):
        val_array = np.array(value, dtype=np.int16)
        return np.append(self.view(np.int16), val_array).view(np.uint8).view(DataBuffer)

    def AppendUInt32(self, value):
        val_array = np.array(value, dtype=np.uint32)
        return np.append(self.view(np.uint32), val_array).view(np.uint8).view(DataBuffer)
        
    def AppendInt32(self, value):
        val_array = np.array(value, dtype=np.int32)
        return np.append(self.view(np.int32), val_array).view(np.uint8).view(DataBuffer)
    
    def AppendBytes(self, value):
        val_array = np.frombuffer(value, dtype=np.uint8)
        return np.append(self, val_array).view(DataBuffer)

    def WriteUInt8(self, value, index):
        self[index] = value
        return self
    
    def WriteInt8(self, value, index):    
        self.view(np.int8)[index] = value
        return self

    def WriteUInt16(self, value, index):
        self[index: index + 2].view(np.uint16)[0] = value
        return self
    
    def WriteInt16(self, value, index):
        self[index: index + 2].view(np.int16)[0] = value
        return self
    
    def WriteUInt32(self, value, index):
        self[index: index + 4].view(np.uint32)[0] = value
        return self
    
    def WriteInt32(self, value, index):
        self[index: index + 4].view(np.int32)[0] = value
        return self
        
    def WriteFloat32(self, value, index):
        self[index: index + 4].view(np.float32)[0] = value
        return self
    
    def WriteBytes(self, index, step, value, use_position=False):
        val_array = np.frombuffer(value, dtype=np.uint8)
        if use_position:
            self[index + self.Position: index + self.Position + step] = val_array
        else:
            self[index: index + step] = val_array
        return None
    
    def WriteValue(self, index, value, dtype, use_position=False):
        val_array = np.array([value], dtype=dtype).view(np.uint8)
        if use_position:
            if dtype == np.uint8 or dtype == np.int8:
                self[index + self.Position] = val_array
            elif dtype == np.uint16 or dtype == np.int16:
                self[index + self.Position: index + self.Position + 2] = val_array
            elif dtype == np.uint32 or dtype == np.int32 or dtype == np.float32:
                self[index + self.Position: index + self.Position + 4] = val_array
        else:
            if dtype == np.uint8 or dtype == np.int8:
                self[index] = val_array
            elif dtype == np.uint16 or dtype == np.int16:
                self[index: index + 2] = val_array
            elif dtype == np.uint32 or dtype == np.int32 or dtype == np.float32:
                self[index: index + 4] = val_array
        return None


    
