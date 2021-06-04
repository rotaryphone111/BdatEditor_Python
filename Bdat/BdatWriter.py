'''
Created on May 18, 2021

@author: kenji
'''

import pandas as pd
import numpy as np
from Common.DataBuffer import DataBuffer


def write_bdat(bdat, file):
    buffer = np.array([], dtype=np.uint8).view(DataBuffer)

    table_count = len(bdat.keys())
    buffer = buffer.AppendValue(table_count, np.int32)
    
    
    file_len = 8 + (4 * table_count)
    table_offset = 8 + (4 * table_count)
    offsets = []
    
    tables = []
    
    for key in bdat.keys():
        if bdat[key]['edited']:
            table_buffer = np.array([66, 68, 65, 84], dtype=np.uint8).view(DataBuffer) #BDAT 
            
            header = np.frombuffer(bdat[key]['header'], dtype=np.uint8)
            names_offset = bdat[key]['names_offset']
            hash_table_offset = bdat[key]['hash_offset']
            hash_table_length = bdat[key]['hash_length']
            item_table_offset = bdat[key]['item_offset']
            encryption_flag = bdat[key]['encryption_flag']
            base_id = bdat[key]['base_id']
            field_14 = bdat[key]['field_14']
            item_size = bdat[key]['item_size']
            checksum = bdat[key]['checksum']
            member_offset = bdat[key]['member_offset']
            member_count = bdat[key]['member_count']

            data = bdat[key]['data'].copy()
            item_count = data.shape[0]


            table_buffer = table_buffer.AppendValue(encryption_flag, np.uint16)
            table_buffer = table_buffer.AppendValue(names_offset, np.uint16)
            table_buffer = table_buffer.AppendValue(item_size, np.uint16)
            table_buffer = table_buffer.AppendValue(hash_table_offset, np.uint16)
            table_buffer = table_buffer.AppendValue(hash_table_length, np.uint16)
            table_buffer = table_buffer.AppendValue(item_table_offset, np.uint16)
            table_buffer = table_buffer.AppendValue(item_count, np.uint16)
            table_buffer = table_buffer.AppendValue(base_id, np.uint16)
            table_buffer = table_buffer.AppendValue(field_14, np.uint16)
             
            
            item_offset = item_table_offset
            items_length = int(item_count * item_size)

            strings_offset = np.int32(item_offset + items_length)
            str_offset = strings_offset.copy()
            
            string_table = np.array([], dtype=np.uint8).view(DataBuffer)
            item_table = np.zeros((item_count * item_size,), dtype=np.uint8).view(DataBuffer)
            
            names = bdat[key]['item_names'].copy()
            strings = bdat[key]['strings'].copy()
            

            for array in bdat[key]['arrays'].keys():
                cols = []
                for new_arr in bdat[key]['arrays'][array].keys():
                    cols.append(new_arr)

                reduced_array = combine_arrays(data[cols])
                reduced_array.name = array
                data.drop(cols, axis=1, inplace=True)
                data = pd.concat((data, reduced_array), axis=1)
            
            for array in bdat[key]['string_arrays'].keys():
                cols = []
                for new_arr in bdat[key]['string_arrays'][array].keys():
                    cols.append(new_arr)

                strs_tup = convert_table_strings(data[cols], str_offset, string_table)
                data[cols] = strs_tup[0]
                str_offset = strs_tup[1]
                string_table = strs_tup[2]

                reduced_array = combine_arrays(data[cols])
                strings.remove(array)
                
                reduced_array.name = array
                data.drop(cols, axis=1, inplace=True)
                data = pd.concat((data, reduced_array), axis=1)
                
            if bdat[key]['flags'] != {}:
                cols = []
                masks = {}
                flag_member_name = ''
                for flag in bdat[key]['flags'].keys():
                    flag_member_name = bdat[key]['flags'][flag]['VarName']
                    cols.append(flag)
                    masks[flag] = bdat[key]['flags'][flag]['mask']

                cols.append(flag_member_name)
                
                type = data[flag_member_name].dtype
                data[flag_member_name] = write_flag_col(data[cols], masks, flag_member_name).astype(type)
                cols.remove(flag_member_name)
                data.drop(cols, axis=1, inplace=True)
                
            data = data.reindex(columns=names)

            
            if strings != []:
                strs_tup = convert_table_strings(data[strings], str_offset, string_table)
                data[strings] = strs_tup[0]
                str_offset = strs_tup[1]
                string_table = strs_tup[2]
            
            
            strings_length = str_offset - strings_offset
            table_buffer = table_buffer.AppendValue(checksum, np.uint16)
            table_buffer = table_buffer.AppendValue(strings_offset, np.uint32)
            table_buffer = table_buffer.AppendValue(strings_length, np.uint32)
            table_buffer = table_buffer.AppendValue(member_offset, np.uint16)
            table_buffer = table_buffer.AppendValue(member_count, np.uint16)

            item_table = write_table_items(data, item_table, item_size)

            
            table_buffer = np.append(table_buffer, header).view(DataBuffer)
            table_buffer = np.append(table_buffer, item_table).view(DataBuffer)
            table_buffer = np.append(table_buffer, string_table).view(DataBuffer)
            if len(table_buffer) % 4 != 0:
                while len(table_buffer) % 4 != 0:
                    table_buffer = table_buffer.AppendUInt8(0)

            bdat[key]['raw'] = table_buffer
            bdat[key]['edited'] = False
        
        else:
            table_buffer = bdat[key]['raw']

        tables.append(table_buffer)
        offsets.append(table_offset)
        file_len += table_buffer.nbytes
        table_offset += table_buffer.nbytes
    
    buffer = buffer.AppendValue(file_len, np.int32)
    for table_offset in offsets:
        buffer = buffer.AppendValue(table_offset, np.int32)
    
    tables_1 = np.concatenate(tables).view(np.uint8) #this is faster than iteration by about half a second
    
    buffer = np.append(buffer, tables_1)
    buffer.tofile(file)
    

def convert_table_strings(table, strings_offset, str_table):
    length = table.shape[0]
    i = 0
    offset_list = {}
    offset = strings_offset
    table1 = table.copy()
    used_strings = {}

    for col in table.columns:
        offset_list[col] = []
        
    while i < length:
        for col in table.columns:
            string = table[col].iloc[i]
            str_len = len(table[col].iloc[i].encode('utf-8'))
            if string not in used_strings.keys():
                used_strings[string] = offset
                offset = strings_offset + str_len + 1
                strings_offset += str_len + 1
                str_table = str_table.AppendBytes(table[col].iloc[i].encode('utf-8'))
                str_table = str_table.AppendBytes(b'\x00')

            offset_list[col].append(used_strings[string])
            
        i += 1
    
    for key in offset_list.keys():
        offsets = pd.Series(offset_list[key], dtype=np.int32)
        offsets.name = key
        table1[key] = offsets

    return(table1, strings_offset, str_table)
     
            
def combine_arrays(table):
    length = table.shape[0]
    a = []
    i = 0 

    while i < length: 
        b = np.array([], dtype=np.uint8).view(DataBuffer)
        
        for col in table.columns:
            if table[col].dtype != object:
                b = b.AppendValue(table[col].iloc[i], dtype=table[col].dtype) 
            
            else:
                print(table[col].iloc[i])

        a.append(b.tobytes())
        i += 1
    return(pd.Series(a))


def write_table_items(table, item_table, item_size):
    length = table.shape[0]

    i = 0 
    j = 0
    while i < length:
        for col in table.columns:
            if table[col].dtype != object:
                item_table.WriteValue(j, table[col].iloc[i], dtype=table[col].dtype)
                j += table[col].dtype.itemsize
            
            elif isinstance(table[col].iloc[i], bytearray) or isinstance(table[col].iloc[i], bytes):
                step = len(table[col].iloc[i])
                item_table.WriteBytes(j, step, table[col].iloc[i])
                j += step
            
            else:
                print(table[col].iloc[i])
        
        if j < (i + 1) * item_size:
            j = (i + 1) * item_size

        i += 1

    return(item_table)


def write_flag_col(flag_cols, masks, flag_name):
    flag_col = flag_cols.apply(write_flags, axis=1, mask_list=masks, name=flag_name)
    return flag_col


def write_flags(flags, mask_list, name):
    flag = flags[name]
    for flag_name in mask_list.keys():
        if flags[flag_name]:
            flag |= mask_list[flag_name]
        else:
            flag ^= (flag & mask_list[flag_name])
    return flag



