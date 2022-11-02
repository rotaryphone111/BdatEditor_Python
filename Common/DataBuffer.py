'''
Created on May 13, 2021

@author: kenji
'''

import numpy as np


class DataBuffer(np.ndarray):
    '''
    classdocs
    '''
    
    def __new__(cls, input_array, game='XC2', start=0, length=False):
        obj = np.asarray(input_array).view(cls)
        obj.Position = start
        obj.Game=game

        if not length:
            obj.Length = input_array.nbytes - start
        else:
            obj.Length = length 
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.Position = 0
        self.Game = getattr(obj, 'Game', 'XC2')
    
    def ReadValue(self, index = 0, Dtype=np.uint8, use_position=False):
        if not use_position:
            if self.Game == 'XC1' or self.Game == 'XBX':
                if self.Game == 'XBX' and Dtype == np.float32:
                    return self[index: int(index + np.uint32(1).itemsize)].view(np.uint32).byteswap()[0]/4096.0
                return self[index: int(index + Dtype(1).itemsize)].view(Dtype).byteswap()[0]
            elif self.Game == 'XC2' or self.Game == 'XC1DE':
                return self[index: int(index + Dtype(1).itemsize)].view(Dtype)[0]
        else:
            if self.Game == 'XC1' or self.Game == 'XBX':
                value = self[self.Position + index: int(self.Position + index + Dtype(1).itemsize)].view(Dtype).byteswap()[0]
                if self.Game == 'XBX' and Dtype == np.float32:
                    value = self[self.Position + index: int(self.Position + index + np.uint32(1).itemsize)].view(np.uint32).byteswap()[0]/4096.0
            elif self.Game == 'XC2' or self.Game == 'XC1DE':
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
        if self.Game == 'XC1' or self.Game == 'XBX':
            result = self[index: index + 2].view(np.uint16).byteswap()[0]
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            result = self[index: index + 2].view(np.uint16)[0]
        return result

    def ReadInt16(self, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            result = self[index: index + 2].view(np.int16).byteswap()[0]
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            result = self[index: index + 2].view(np.int16)[0]
        return result

    def ReadUInt32(self, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            result = self[index: index + 4].view(np.uint32).byteswap()[0]
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            result = self[index: index + 4].view(np.uint32)[0]
        return result

    def ReadInt32(self, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            result = self[index: index + 4].view(np.int32).byteswap()[0]
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            result = self[index: index + 4].view(np.int32)[0]
        return result

    def ReadUInt64(self, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            result = self[index: index + 8].view(np.uint64).byteswap()[0]
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            result = self[index: index + 8].view(np.uint64)[0]
        return result

    def ReadInt64(self, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            result = self[index: index + 8].view(np.int64).byteswap()[0]
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            result = self[index: index + 8].view(np.int64)[0]
        return result
    
    def ReadSingle(self, index):
        if self.Game == 'XC1':
            result = self.view(np.float32).byteswap()[int(index/4)]
        elif self.Game == 'XBX':
            result = self.view(np.uint32).byteswap()[int(index/4)]/4096.0
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
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
        if self.Game == 'XC1' or self.Game == 'XBX':
            if self.Game == 'XBX' and dtype == np.float32:
                val_array = (np.array([value*4096], dtype=np.uint32)).byteswap().view(np.uint8)
            else:
                val_array = np.array([value], dtype=dtype).byteswap().view(np.uint8)
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array([value], dtype=dtype).view(np.uint8)
        return DataBuffer(np.append(self, val_array), self.Game)
    
    def AppendUInt8(self, value):
        val_array = np.array(value, dtype=np.uint8)
        return DataBuffer(np.append(self, val_array), self.Game)
        
    def AppendInt8(self, value):
        val_array = np.array(value, dtype=np.int8)
        return DataBuffer(np.append(self.view(np.int8), val_array).view(np.uint8), self.Game)

    def AppendUInt16(self, value):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.uint16).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.uint16)
        return DataBuffer(np.append(self.view(np.uint16), val_array).view(np.uint8), self.Game)
        
    def AppendInt16(self, value):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.int16).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.int16)
        return DataBuffer(np.append(self.view(np.int16), val_array).view(np.uint8), self.Game)

    def AppendUInt32(self, value):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.uint32).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.uint32)
        return DataBuffer(np.append(self.view(np.uint32), val_array).view(np.uint8), self.Game)
        
    def AppendInt32(self, value):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.int32).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.int32)
        return DataBuffer(np.append(self.view(np.int32), val_array).view(np.uint8), self.Game)
    
    def AppendBytes(self, value):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.frombuffer(value, dtype=np.uint8).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.frombuffer(value, dtype=np.uint8)
        return DataBuffer(np.append(self, val_array), self.Game)

    def WriteUInt8(self, value, index):
        self[index] = value
        return self
    
    def WriteInt8(self, value, index):    
        self.view(np.int8)[index] = value
        return self

    def WriteUInt16(self, value, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.uint16).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.uint16)
        self[index: index + 2].view(np.uint16)[0] = val_array
        return self
    
    def WriteInt16(self, value, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.int16).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.int16)
        self[index: index + 2].view(np.int16)[0] = val_array
        return self
    
    def WriteUInt32(self, value, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.uint32).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.uint32)
        self[index: index + 4].view(np.uint32)[0] = val_array
        return self
    
    def WriteInt32(self, value, index):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.array(value, dtype=np.int32).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.int32)
        self[index: index + 4].view(np.int32)[0] = val_array
        return self
        
    def WriteFloat32(self, value, index):
        if self.Game == 'XC1':
            val_array = np.array(value, dtype=np.float32).byteswap()
        elif self.Game == 'XBX':
            val_array = np.array((value*4096), dtype=np.uint32).byteswap()
            self[index: index + 4].view(np.uint32)[0] = val_array
            return self
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.array(value, dtype=np.float32)
        self[index: index + 4].view(np.float32)[0] = val_array
        return self
    
    def WriteBytes(self, index, step, value, use_position=False):
        if self.Game == 'XC1' or self.Game == 'XBX':
            val_array = np.frombuffer(value, dtype=np.uint8).byteswap()
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
            val_array = np.frombuffer(value, dtype=np.uint8)

        if use_position:
            self[index + self.Position: index + self.Position + step] = val_array
        else:
            self[index: index + step] = val_array
        return None
    
    def WriteValue(self, index, value, dtype, use_position=False):
        if self.Game == 'XC1' or self.Game == 'XBX':
            if self.Game == 'XBX' and dtype == np.float32:
                val_array = np.array([value*4096], dtype=np.uint32).byteswap().view(np.uint8)
            else:
                val_array = np.array([value], dtype=dtype).byteswap().view(np.uint8)
        elif self.Game == 'XC2' or self.Game == 'XC1DE':
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


    
