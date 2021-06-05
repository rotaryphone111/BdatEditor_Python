import PySimpleGUIQt as sg
import pandas as pd
import json
import os.path

from Bdat import BdatReader
from Bdat.BdatWriter import write_bdat

    

def FileOpenMenu():
    sg.theme('LightGrey1')
    layout = [[sg.Text('Filename')], 
              [sg.Input(), sg.FileBrowse()],
              [sg.OK(), sg.Cancel()]]
    event, values = sg.Window('Enter Bdat File', layout).read(close=True)
    
    file = values[0]
    return(file)

def BdatEditor(file=None):
    sg.theme('LightGrey1')
    if not file:
        file = FileOpenMenu()
    if file == '':
        return 
    backup_file = os.path.dirname(file) + '/.' + os.path.basename(file) + '.histfile'
    if os.path.isfile(backup_file):
        with open(backup_file, 'r') as b:
            backup_dict = json.load(b)
    else:
        backup_dict = {}

    table_dicts = BdatReader.read_bdat_file(file)
    table_names = list(table_dicts.keys())
    name_max_len = len(max(table_names, key=len))
    width, height = sg.Window.get_screen_dimensions(sg.Window)
    # size_table = (int(width - (name_max_len * 10) - 14), height - 25)
    # size_px = (((name_max_len * 10) + 14), height)
    height -= 27
    height = int(height/37) - 2
    size = (name_max_len, height)

    layout = [[sg.Button('Open File')],
              [sg.Listbox(values=table_names, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, key='name_list', size=size)],
              [sg.Button('Read Table'), sg.Button('Exit')]]
    window = sg.Window('Bdat Editor', layout)

    while True:
        try:
            event, values = window.read()
            print(event, values)
            if event == sg.WIN_CLOSED or event == 'Exit':
                break

            if event == 'Read Table':
                if values['name_list'] == []:
                    continue
                key = values['name_list'][0]

                try:
                    backup_dict_table = backup_dict[key]
                    undo_stack = backup_dict_table['undo'] 
                    redo_stack = backup_dict_table['redo']
                except NameError:
                    undo_stack = []
                    redo_stack = []

                table = BdatReader.read_raw_data(table_dicts[key])
                t = table['data'].copy()
                d = table['data'].dtypes.copy()
                t_str = t.apply(stringify_column)

                id_num = []
                for i in range(table['base_id'], table['base_id'] + table['item_count']):
                    id_num.append(i)
                table_values = []
                for i in t_str.index:
                    table_values.append([str(id_num[i])] + t_str.iloc[i].tolist())
                    
                table_values_orig = table_values.copy()

                cols = []
                col_widths = []
                cols.append('orig_row_id')
                col_widths.append(12)
                for col in t.columns:
                    if d[col] != object:
                        name = str(col) + '\n' + '(' + str(d[col]) + ')'
                    else:
                        name = str(col) + '\n' + '(string)'
                    cols.append(name)
                    col_widths.append(len(name))
                
                
                # col_layout = [[sg.Button('Open File', size=(size[0], 1))], 
                #               [sg.Listbox(values=table_names, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, key='name_list', size=(size[0], size[1] - int((2*25)/25)))],
                #               [sg.Button('Read Table', size=(int(size[0]/2), 1)), sg.Button('Save Table', size=(int(size[0]/2), 1))]]
                # table_layout =  [[sg.Table(table_values, headings=cols, display_row_numbers=True, enable_events=True, key='table')],
                #                 [sg.Button('Add Row'), sg.Button('Remove Rows'), sg.Button('Exit')]]

                layout = [[sg.Button('Open File'), sg.Button('Undo Change'), sg.Button('Redo Change'), sg.Button('Restore Initial Table')],
                          [sg.Text(key, justification='right')],
                          [sg.Listbox(values=table_names, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, key='name_list', size=size),
                           sg.Table(table_values, headings=cols, auto_size_columns=False, bind_return_key=True, col_widths=col_widths, display_row_numbers=True, enable_events=True, key='table')],
                          [sg.Button('Read Table'), sg.Button('Save Table'), sg.Button('Add Row'), sg.Button('Remove Rows'),
                           sg.Button('Exit')]]

                # layout = [[sg.Column(col_layout),
                #            sg.Table(table_values, headings=list(t.columns), display_row_numbers=True, enable_events=True, key='table')],
                #           [sg.Button('Add Row'), sg.Button('Remove Rows'),
                #             sg.Button('Exit')]]
                # layout = [[sg.Column(col_layout, size=size_px), sg.Column(table_layout, size=size_table)]]
                window.close()
                window = sg.Window('Bdat Editor', layout, size=sg.Window.get_screen_dimensions(sg.Window))
            
            if event == 'Open File':
                file = FileOpenMenu()
                if file == '':
                    continue
                BdatEditor(file)
                window.close()
                return

            if event == 'Save Table':
                undo_stack.append(table_values.copy())
                window.FindElement('table').Update(values=window.FindElement('table').get())
                table_values = window.FindElement('table').Values
                table_values_new = []
                for value in table_values:
                    value.pop(0)
                    table_values_new.append(pd.Series(value, index=t.columns))
                df = pd.DataFrame(table_values_new, columns=t.columns)
                for col in d.index:
                    if d[col] != object and d[col] != bool:
                        df[col] = pd.to_numeric(df[col])
                    elif d[col] == bool:
                        invalid_values = df[col].str.match('true|false', case=False) == False
                        if invalid_values.any():
                            invalid_values = df[col].where(invalid_values).dropna()
                            raise ValueError('invalid values: ' + str(invalid_values.tolist()))
                        df[col] = df[col].str.match('true', case=False)

                df = df.astype(d)
                
                table['data'] = df
                table['edited'] = True
                table_dicts[key] = table
                write_bdat(table_dicts, file)

                backup_dict_table = {}
                backup_dict_table['undo'] = undo_stack
                backup_dict_table['redo'] = redo_stack
                backup_dict[key] = backup_dict_table
                with open(backup_file, 'w') as b:
                    json.dump(backup_dict, b)


            if event == 'Add Row':
                table_values = window.FindElement('table').Values
                undo_stack.append(table_values.copy())
                window.FindElement('table').Update(values=table_values)
                new_row = []
                for col in cols:
                    new_row.append('')
                table_values.append(new_row)
                window.FindElement('table').Update(values=table_values)
                table['item_count'] += 1


            if event == 'Remove Rows':
                if values['table'] != []:
                    table['item_count'] -= len(values['table'])
                    table_values = window.FindElement('table').Values
                    window.FindElement('table').Update(values=table_values)
                    undo_stack.append(table_values.copy())
                    for i in sorted(values['table'], reverse=True):
                        table_values.pop(i)
                    window.FindElement('table').Update(values=table_values)
            
            if event == 'Undo Change':
                if len(undo_stack) == 0:
                    continue
                else:
                    table_values = window.FindElement('table').get()
                    redo_stack.append(table_values.copy())
                    table_values_undo = undo_stack.pop()
                    item_diff = len(table_values) - len(table_values_undo)
                    table['item_count'] -= item_diff
                    window.FindElement('table').Update(values=table_values_undo)
            
            if event == 'Redo Change':
                if len(redo_stack) == 0:
                    continue
                else:
                    table_values = window.FindElement('table').get()
                    undo_stack.append(table_values.copy())
                    table_values_redo = redo_stack.pop()
                    item_diff = len(table_values) - len(table_values_redo)
                    table['item_count'] -= item_diff
                    window.FindElement('table').Update(values=table_values_redo)
            
            if event == 'Restore Initial Table':
                table_values = window.FindElement('table').get()
                undo_stack.append(table_values)
                window.FindElement('table').Update(values=table_values_orig)
                
            if event == 'table':
                if table_values == window.FindElement('table').get():
                    continue
                else:
                    undo_stack.append(table_values.copy())
                    table_values = window.FindElement('table').get()
                    window.FindElement('table').Update(values=table_values)
            

        except ValueError as v:
            sg.popup('Invalid Value Error: ' + str(v))

        except IndexError as I:
            if table_values == []:
                id_num = []
                for i in range(table['base_id'], table['base_id'] + table['item_count']):
                    id_num.append(i)
                table_values = []
                for i in id_num:
                    table_values.append([str(i)])

                cols = ['orig_row_id']
                col_widths = [12]
                noted_key = key + ' (note: empty table)'

                layout = [[sg.Button('Open File')],
                          [sg.Text(noted_key, justification='right')],
                          [sg.Listbox(values=table_names, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, key='name_list', size=size),
                           sg.Table(table_values, headings=cols, auto_size_columns=False, col_widths=col_widths, display_row_numbers=True, enable_events=True, key='table')],
                          [sg.Button('Read Table'), sg.Button('Exit')]]
                window.close()
                window = sg.Window('Bdat Editor', layout, size=sg.Window.get_screen_dimensions(sg.Window))
            else: 
                sg.popup(I)
                break
                
    window.close()

def stringify_column(col):
    return col.apply(str)


