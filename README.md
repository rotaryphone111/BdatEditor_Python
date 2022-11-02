# BdatEditor-Python
Bdat Editor Python Rewrite

This program currently has only been tested for XC2. In theory it should work for XC1DE, XCX, and XC1 but no testing has been done for this.
As should be evident by the release numbers, this package is currently in alpha and should therefore be used at your own risk.


INSTALLATION:
Run either:

    python setup.py (OSX/Linux)
    py setup.py(Windows)

or 

    pip install . (OSX/Linux)
    python -m pip install . (Windows)


CAVEATS:

- New rows *must* be filled completely right now. If they are not the bdat writer will throw an error.

- Xenoblade X handles floats extremely strangely. If precision is a priority make sure your floating point 
  numbers work out to integer values when multiplied by 4096.


CHANGELOG:

11/02/2022: Support has been added for Xenoblade X and Xenoblade Chronicles 1, along with some bugfixes.

09/21/2021: I have cythonized the IO code to increase read and write speeds. 

08/24/2021: As per a request, I have added a copy-paste function. Currently you can only paste to the end of the table, to avoid interfering with IDs.

07/12/2021: Due to cross platform issues, I have added a tick box to display control characters. Saving while this is active is somewhat finnicky. Generally use at your own risk right now. Backup your stuff.

06/10/2021: Fixed an oopsies inherited from old BdatEditors to do with flags

06/09/2021: Fixed issues with the executable

06/09/2021: Crash bugfix

06/07/2021: Finally added setup scripts

06/04/2021: Added full undo and redo features
