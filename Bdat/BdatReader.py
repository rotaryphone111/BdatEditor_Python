# coding: utf-8

import numpy as np
import pandas as pd
from Common.DataBuffer import DataBuffer


def read_bdat_file(file):
    bdat = open_bdat_file(file)
    tables = split_bdat_file(bdat)
    return construct_table_dicts(tables)

def open_bdat_file(file):
    return np.fromfile(file, dtype=np.uint8).view(DataBuffer)


def split_bdat_file(file):
    table_count = file.ReadInt32(0)
    tables = []
    i = 0
    while i < table_count:
        offset = file.ReadInt32(8 + (4 * i))
        if i < table_count - 1:
            nextOffset = file.ReadInt32(8 + (4 * (i+1)))
        else:
            nextOffset = file.nbytes
        tables.append(file.tobytes()[offset:nextOffset])
        i += 1
    return tables


def construct_table_dicts(tables):
    table_dicts = {}

    for table in tables:
        table_dict = {}
        data = np.frombuffer(table, dtype=np.uint8).view(DataBuffer)

        NamesOffset = data.ReadUInt16(6)
        name = data.ReadUTF8Z(NamesOffset)
        
        table_dict['raw'] = data
        table_dict['edited'] = False
        table_dicts[name] = table_dict

    return(table_dicts)

def read_raw_data(table_dict):
    data = table_dict['raw']

    EncryptionFlag = data.ReadUInt16(4) 
    NamesOffset = data.ReadUInt16(6)
    ItemSize = data.ReadUInt16(8)
    HashTableOffset = data.ReadUInt16(10)
    HashTableLength = data.ReadUInt16(12)
    ItemTableOffset = data.ReadUInt16(14)
    ItemCount = np.int64(data.ReadUInt16(16))
    BaseId = data.ReadUInt16(18)
    Field14 = data.ReadUInt16(20)
    Checksum = data.ReadUInt16(22)
    MemberTableOffset = data.ReadUInt16(32)
    MemberCount = data.ReadUInt16(34)

    if MemberTableOffset > ItemTableOffset:
        print('problem')
        
    members = calc_data_types(data, MemberTableOffset, MemberCount)
    names = members[0]
    types = members[1]
    offsets = members[2]
    string_members = members[3]
    flag_members = members[4]
    array_members = members[5]
    dup_names = members[6]
    
    ItemsLength = int(ItemSize * ItemCount)
    item_slice = data[ItemTableOffset:ItemTableOffset + ItemsLength]
    item_slices = []

    str_arrays = {}
    arrays = {}

    i = 0
    while i < ItemCount:
        item_slices.append(item_slice[int(ItemSize * i) : int(ItemSize * (i+1))])
        i += 1


    table_values = []
    for chunk in item_slices:
        i = 0
        l = {}

        while i < len(types):
            if names[i] in array_members.keys():
                j = 0
                array_count = array_members[names[i]]

                new_strings = {}
                new_keys = {}

                while j < array_count:
                    l[names[i] + '_' + str(j)] = chunk.ReadValue((offsets[i] + int(j * types[i](1).itemsize)), types[i])
                    if names[i] in string_members:
                        new_strings[names[i] + '_' + str(j)] = np.int32
                    else:
                        new_keys[names[i] + '_' + str(j)] = types[i]
                    j += 1

                if names[i] in string_members:
                    str_arrays[names[i]] = new_strings

                else:
                    arrays[names[i]] = new_keys

            else:
                l[names[i]] = chunk.ReadValue(offsets[i], types[i])
            
            i += 1

        for flag_name in flag_members.keys():
            flag_member_pos = flag_members[flag_name]['VarPos']
            flag_mask = flag_members[flag_name]['mask']
            flag_value = (chunk.ReadUInt8(flag_member_pos) & flag_mask != 0)
            flag_members[flag_name]['VarName'] = names[flag_members[flag_name]['VarIndex']]
            l[flag_name] = flag_value

        table_values.append(l)
    
    
    typedict = {}
    i = 0
    while i < len(types):
        if names[i] not in array_members.keys():
            typedict[names[i]] = types[i]
        i += 1
    
    for flag in flag_members.keys():
        typedict[flag] = bool
    
    for array in array_members.keys():
        if array in string_members:
            for st in str_arrays[array]:
                typedict[st] = np.int32
        
        else:
            for arr in arrays[array].keys():
                typedict[arr] = arrays[array][arr]
    
    for typed in typedict.keys():
        if typed not in table_values[0].keys():
            print(typed)
   
    if MemberCount > 0:
        items = pd.DataFrame(table_values, columns=table_values[0].keys()).astype(dtype=typedict)
        str_names = []
        for str_name in string_members:
            if str_name in str_arrays.keys():
                str_names.extend(str_arrays[str_name])
            else:
                str_names.append(str_name)

        items[str_names] = items[str_names].apply(get_strings_from_columns, data_in=data, axis=1) 
    else:
        items = pd.DataFrame()
        
    header = data[36:ItemTableOffset].tobytes()

    table_dict['encryption_flag'] = EncryptionFlag
    table_dict['header'] = header
    table_dict['hash_offset'] = HashTableOffset
    table_dict['hash_length'] = HashTableLength
    table_dict['base_id'] = BaseId
    table_dict['field_14'] = Field14
    table_dict['checksum'] = Checksum
    table_dict['data'] = items
    table_dict['item_size'] = ItemSize
    table_dict['item_count'] = ItemCount
    table_dict['item_offset'] = ItemTableOffset
    table_dict['item_names'] = names
    table_dict['member_count'] = MemberCount
    table_dict['member_offset'] = MemberTableOffset
    table_dict['names_offset'] = NamesOffset
    table_dict['strings'] = string_members
    table_dict['string_arrays'] = str_arrays
    table_dict['arrays'] = arrays
    table_dict['flags'] = flag_members
    table_dict['dup_names'] = dup_names

    return table_dict


def get_strings_from_columns(column, data_in): #, strs_len, strs_offset):
    return column.apply(get_strings_from_offsets, data=data_in) #, strings_len=strs_len, strings_offset=strs_offset)


def get_strings_from_offsets(offset, data): #, strings_len, strings_offset):
    # end = strings_offset + strings_len
    # str_sec = data[offset:end]
    # return(data.ReadUTF8Z(str_sec))
    return(data.ReadUTF8Z(offset))


def calc_data_types(table, memberTableOffset, memberCount):
    names = []
    dup_names = []
    num_dup = 0
    valtypes = []
    offsets = []
    string_members = []
    flag_members = {}
    array_members = {}
    i = 0

    while i < memberCount: 
        memberOffset = memberTableOffset + (i * 6) 
        infoOffset = table.ReadUInt16(memberOffset) 
        nameOffset = table.ReadUInt16(memberOffset + 4) 
        name = table.ReadUTF8Z(nameOffset)
        if name in names:
            dup_names.append(name)
            names.append(name + '_' + str(num_dup))
            num_dup += 1
        else:
            names.append(name)
        membertype = table[infoOffset]

        if membertype == 3:
            FlagVarOffset = table.ReadUInt16(infoOffset + 6)
            FlagVarInfo = table.ReadUInt16(FlagVarOffset)
            FlagVarPos = table.ReadUInt16(FlagVarInfo + 2)
            FlagVarIndex = int((FlagVarOffset - memberTableOffset) / 6)

            flag_members[name] = {'index': table[infoOffset + 1],
                                  'mask': table.ReadUInt32(infoOffset + 2),
                                  'VarPos': FlagVarPos,
                                  'VarIndex': FlagVarIndex}
            names.pop()

        else:
            valtype = table[infoOffset + 1]
            memberPos = table.ReadUInt16(infoOffset + 2)

            if valtype == 0:
                valtypes.append(None)
            elif valtype == 1:
                valtypes.append(np.uint8)
            elif valtype == 2:
                valtypes.append(np.uint16)
            elif valtype == 3:
                valtypes.append(np.uint32)
            elif valtype == 4:
                valtypes.append(np.int8)
            elif valtype == 5:
                valtypes.append(np.int16)
            elif valtype == 6:
                valtypes.append(np.int32)
            elif valtype == 7:
                valtypes.append(np.int32)
                string_members.append(name)
            elif valtype == 8:
                valtypes.append(np.float32)

            offsets.append(memberPos)

        if membertype == 2:
            array_count = table.ReadUInt16(infoOffset + 4)
            array_members[name] = array_count

        i += 1

    return (names, valtypes, offsets, string_members, flag_members, array_members, dup_names)

# def decrypt_bdat_file(file):
#     table_count = file.ReadInt32(0)
#
#     i = 0
#
#     while i < table_count:
#         offset = file.ReadInt32(8 + 4 * i)
#         table = file[offset:]










    
    
