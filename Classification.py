import cv2

from keras.models import load_model
import keras
import numpy as np
import pandas as pd
import pyautogui
import subprocess

model = load_model(r"C:\Users\USER\Documents\HandGestureRecognitionSystemProject\model\resnetmodel.hdf5")

vic = cv2.VideoCapture(0)
vic.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
vic.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

labels = pd.read_csv(r"C:\Users\USER\Documents\Dataset\jester-v1-labels.csv", header = None)

print(labels)

from keyboard import Keyboard

buffer = []
cls = []
predicted_value = 0
final_label = ""
i = 1

while(vic.isOpened()):
    ret, frame = vic.read()
    if ret:
        image = cv2.resize(frame, (96, 64))
        image = image / 255.0
        buffer.append(image)
        if(i % 16 == 0):
            buffer = np.expand_dims(buffer, 0)
            predicted_value = np.argmax(model.predict(buffer))
            cls = labels.iloc[predicted_value]
            print(cls)
            print(cls.iloc[0])
            
            if(predicted_value == 0):
                final_label = "Swiping Left"
                pyautogui.hotkey('alt', 'tab')  # Switch between windows
            elif(predicted_value == 1):
                final_label = "Swiping Right"
                pyautogui.hotkey('win', 'tab')  # Open Task View
            elif(predicted_value == 2):
                final_label = "Swiping Down"
                pyautogui.press('volumedown')  # Decrease volume
            elif(predicted_value == 3):
                final_label = "Swiping Up"
                pyautogui.press('volumeup')  # Increase volume
            elif(predicted_value == 4):
                final_label = "Pushing hand away"
                pyautogui.hotkey('win', 'd')  # Show desktop
            elif(predicted_value == 5):
                final_label = "Pulling hand in"
                pyautogui.hotkey('win', 'm')  # Minimize all windows
            elif(predicted_value == 6):
                final_label = "Sliding two fingers left"
                pyautogui.hotkey('ctrl', 'shift', 'tab')  # Previous tab
            elif(predicted_value == 7):
                final_label = "Sliding two fingers right"
                pyautogui.hotkey('ctrl', 'tab')  # Next tab
            elif(predicted_value == 8):
                final_label = "Sliding two fingers down"
                pyautogui.scroll(-100)  # Scroll down
            elif(predicted_value == 9):
                final_label = "Sliding two fingers up"
                pyautogui.scroll(100)  # Scroll up
            elif(predicted_value == 10):
                final_label = "Pushing two fingers away"
                pyautogui.hotkey('win', 'up')  # Maximize window
            elif(predicted_value == 11):
                final_label = "Pulling two fingers in"
                pyautogui.hotkey('win', 'down')  # Minimize window
            elif(predicted_value == 12):
                final_label = "Rolling hand forward"
                pyautogui.hotkey('ctrl', '+')  # Zoom in
            elif(predicted_value == 13):
                final_label = "Rolling hand backward"
                pyautogui.hotkey('ctrl', '-')  # Zoom out
            elif(predicted_value == 14):
                final_label = "Turning hand clockwise"
                pyautogui.hotkey('alt', 'f4')  # Close window
            elif(predicted_value == 15):
                final_label = "Turning hand counterclockwise"
                pyautogui.press('esc')  # Escape key
            elif(predicted_value == 16):
                final_label = "Zooming in with full hand"
                subprocess.run(["powershell", "(New-Object -com Shell.Application).Windows() | foreach-object {$_.fullscreen=$true}"])  # Fullscreen
            elif(predicted_value == 17):
                final_label = "Zooming out with full hand"
                subprocess.run(["powershell", "(New-Object -com Shell.Application).Windows() | foreach-object {$_.fullscreen=$false}"])  # Exit fullscreen
            elif(predicted_value == 18):
                final_label = "Zooming in with two fingers"
                pyautogui.hotkey('win', '+')  # Windows magnifier zoom in
            elif(predicted_value == 19):
                final_label = "Zooming out with two fingers"
                pyautogui.hotkey('win', '-')  # Windows magnifier zoom out
            elif(predicted_value == 20):
                final_label = "Thumb up"
                pyautogui.press('playpause')  # Play/Pause media
            elif(predicted_value == 21):
                final_label = "Thumb down"
                pyautogui.press('volumemute')  # Mute volume
            elif(predicted_value == 22):
                final_label = "Shaking hand"
                pyautogui.hotkey('win', 'l')  # Lock screen
            elif(predicted_value == 23):
                final_label = "Stop sign"
                #subprocess.run(["shutdown", "/s", "/t", "0"])  # Shutdown PC
            elif(predicted_value == 24):
                final_label = "Drumming fingers"
                pyautogui.hotkey('win', 'prtscr')  # Take screenshot
            elif(predicted_value == 25):
                final_label = "No gesture"
            else:
                final_label = "Doing other things"
            
            cv2.imshow('frame', frame)
            buffer = []
        
        i += 1
        text = "activity: {}".format(final_label)
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 0), 5)
        cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vic.release()
cv2.destroyAllWindows()


