## Display

### Initial Setup:
Creating the virtual environment:
- Open a new cmd terminal in VS code (if not default, click dropdown arrow next to '+' on the terminal top right then choose command prompt)
- In the terminal, ```pip install virtual env```
- Create an environment: ```virtual env venv```
- Launch environment: ```venv\Scripts\activate``` You should see (venv) at the beginning of the command line if you've successfully entered the environment.

Setup/install necessary libraries:
- Once the virtual environment is launched, run:
    - ```pip install openai speechrecognition gtts playsound torch torchvision matplotlib PyAudio Pillow```
- Make sure your python interpreter is correct by doing ```CTRL + SHIFT + P```, then "Python: Select Interpreter", and choosing "Python 3.11.4" or similar

### Every time:
Launching the display is hella easy:
- Just run the ```app.py``` file in ```src```
- The display is made using tkinter library