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
    # height -= sg.DEFAULT_BUTTON_ELEMENT_SIZE[1]
    height = int(height/30) - 1
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
                if isinstance(t, pd.Series):
                    d = t.dtype
                else:
                    d = table['data'].dtypes.copy()
                table_values = []
                for i in t.index:
                    table_values.append(t.iloc[i].apply(str).tolist())

                layout = [[sg.Button('Open File')],
                            [sg.Listbox(values=table_names, select_mode=sg.LISTBOX_SELECT_MODE_SINGLE, key='name_list', size=size),
                           sg.Table(table_values, headings=list(t.columns), display_row_numbers=True, enable_events=True, key='table')],
                          [sg.Button('Add Row'), sg.Button('Remove Rows'), sg.Button('Read Table'), sg.Button('Save Table'),
                            sg.Button('Exit')]]
                window.close()
                window = sg.Window('Bdat Editor', layout, size=sg.Window.get_screen_dimensions(sg.Window))
            
            if event == 'Open File':
                layout1 = [[sg.Text('Filename')], 
                          [sg.Input(), sg.FileBrowse()],
                          [sg.OK(), sg.Cancel()]]
                event, values = sg.Window('Enter Bdat File', layout1).read(close=True)
                file = values[0]
                BdatEditor(file)
                window.close()
                return

            if event == 'Save Table':
                i = 0
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


        except:
            break


    window.close()

