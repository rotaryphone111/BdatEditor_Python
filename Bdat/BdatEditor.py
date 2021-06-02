import PySimpleGUIQt as sg
from Bdat import BdatReader
from Bdat.BdatWriter import write_bdat
import pandas as pd

    

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
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            if event == 'Read Table':
                if values['name_list'] == []:
                    continue
                key = values['name_list'][0]
                table = BdatReader.read_raw_data(table_dicts[key])
                t = table['data'].copy()
                d = table['data'].dtypes.copy()
                t_str = t.apply(stringify_column)

                table_values = []
                for i in t_str.index:
                    table_values.append(t_str.iloc[i].tolist())
                    
                cols = []
                col_widths = []
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

                layout = [[sg.Button('Open File')],
                          [sg.Listbox(values=table_names, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, key='name_list', size=size),
                           sg.Table(table_values, headings=cols, auto_size_columns=False, col_widths=col_widths, display_row_numbers=True, enable_events=True, key='table')],
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
                # layout1 = [[sg.Text('Filename')], 
                #           [sg.Input(), sg.FileBrowse()],
                #           [sg.OK(), sg.Cancel()]]
                # event, values = sg.Window('Enter Bdat File', layout1).read(close=True)
                # file = values[0]
                file = FileOpenMenu()
                if file == '':
                    continue
                BdatEditor(file)
                window.close()
                return

            if event == 'Save Table':
                window.FindElement('table').Update(values= window.FindElement('table').get())
                table_values = window.FindElement('table').Values
                table_values_new = []
                for value in table_values:
                    table_values_new.append(pd.Series(value, index=t.columns))
                df = pd.DataFrame(table_values_new, columns=t.columns)
                if isinstance(df, pd.Series):
                    if d != object and d != bool:
                        df = pd.to_numeric(df).astype(d)
                    elif d == bool:
                        df = (df == 'True')
                else:
                    for col in d.index:
                        if d[col] != object and d[col] != bool:
                            df[col] = pd.to_numeric(df[col])
                        elif d[col] == bool:
                            df[col] = (df[col] == 'True')

                    df = df.astype(d)
                
                table['data'] = df
                table['edited'] = True
                table_dicts[key] = table
                write_bdat(table_dicts, file)


            if event == 'Add Row':
                window.FindElement('table').Update(values= window.FindElement('table').get())
                new_row = []
                for col in t.columns:
                    new_row.append('')
                table_values = window.FindElement('table').Values
                table_values.append(new_row)
                window.FindElement('table').Update(values=table_values)
                table['item_count'] += 1


            if event == 'Remove Rows':
                if values['table'] != []:
                    table['item_count'] -= len(values['table'])
                    window.FindElement('table').Update(values= window.FindElement('table').get())
                    table_values = window.FindElement('table').Values
                    for i in sorted(values['table'], reverse=True):
                        table_values.pop(i)
                    window.FindElement('table').Update(values=table_values)

        except ValueError:
            sg.popup('Invalid Value')

        # except:
        #     break


    window.close()

def stringify_column(col):
    return col.apply(str)


