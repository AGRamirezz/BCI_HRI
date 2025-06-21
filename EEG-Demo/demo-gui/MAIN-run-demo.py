''' This code holds all the main functions to run the EEG-Robot demo. The key & magic to this code working with html and javascript is the eel function.
This exposes python functions to javascript, allowing really nice and easy integration & opens up the possibilities for making the GUI much more visually
appealing.

Note: this whole application runs on Python 3.11.0. Make sure the virtual environment is activated!

Last modified on: 9.12.24
Modified by: Ada Kanapskyte'''

from pylsl import StreamInlet, resolve_stream
import mne
from mne_lsl.stream import StreamLSL as Stream
from matplotlib import pyplot as plt
import time
import pandas as pd
import numpy as np
import socket
from matplotlib.animation import FuncAnimation
import eel
import mpld3
from keras.models import load_model


# Initialize Eel and specify web folder
eel.init('web')

# Set up EEG stream via LSL
stream = Stream(bufsize=2).connect()
stream.drop_channels(["X1", "X2", "X3", "TRG", "ACCT", "ACCX", "ACCY", "ACCZ"])
stream.filter(l_freq=2,h_freq=35)

# Set up communication with robot 
HOST = '192.168.1.99'
PORT = 5001
client_socket = None

# Change this variable to false whenever you don't want to test with the robot in the loop!
socket_on=True

# Create a figure for plotting
fig, ax = plt.subplots()

@eel.expose
def init_socket():
    global client_socket
    if client_socket is None:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print("Socket connected!") 

@eel.expose
def start_animation():
    global ani # Declare ani as global to access it later
    ani = FuncAnimation(fig, animate_plot, interval=1000)
    plt.show()


# Function to update the plot
def animate_plot(i):
    ''' This function runs the real-time EEG data visualization. Only 4 channels are displayed to reduce clutter.'''
    eeg_sample, ts = stream.get_data()
    init_ts = ts[0]
    ch1 = eeg_sample[0]/1000
    ch2 = eeg_sample[1]/1000
    ch3 = eeg_sample[2]/1000
    ch4 = eeg_sample[3]/1000
    ch5 = eeg_sample[4]/1000
    ch6 = eeg_sample[5]/1000
    ch7 = eeg_sample[6]/1000

    ax.cla()  # Clear the previous plot
    ax.plot(ts-init_ts, ch1, label='Ch1')
    ax.plot(ts-init_ts, ch2, label='Ch2')
    ax.plot(ts-init_ts, ch3, label='Ch3')
    ax.plot(ts-init_ts, ch4, label='Ch4')

    ax.legend(loc='upper left')
    ax.set_title('Brain Signal Live Stream')
    ax.set_ylabel('EEG Signal')
    ax.set_xlabel('Time')


@eel.expose
def predict_EEG():
    ''' This function is responsible for streaming chunks of real-time EEG data and then passing it into the keras model. 
    socket_on variable is something you can change at the top of this script to run the code without the robot in the loop.'''
    global client_socket
    cog_model = load_model('C:\\Users\\adolfo.ramirez\\Documents\\eeg_demo\\cnn_all_class1.keras')
    num_chunks = 0
    eeg_chunks = []
    final_prediction = ""
    if socket_on == True:
        while num_chunks <5:
            print('Running eeg')
            eeg_sample, ts = stream.get_data()
            time.sleep(1.25)
            #print(len(eeg_sample))
            eeg_chunks.append(eeg_sample[:,0:300])
            num_chunks +=1

        np_eeg = np.array(eeg_chunks)
        #print(np_test.shape)
        np_eeg_reshaped = np_eeg.reshape(num_chunks,300,20,1)
        #print(np_test_reshaped.shape)
        
        eeg_predictions = cog_model.predict(np_eeg_reshaped)
        print("eeg predictions:", eeg_predictions)
        avg_prediction = np.mean(eeg_predictions, axis=0)
        print("average prediction:", avg_prediction)

        max_pred_idx = np.argmax(avg_prediction)
        max_pred_val = avg_prediction[max_pred_idx]

        if max_pred_idx == 0:
            final_prediction = "left"
            img = 'img/arrow_pictures/pred_left.png'
        elif max_pred_idx == 1:
            final_prediction = "right"
            img = 'img/arrow_pictures/pred_right.png'
        elif max_pred_idx == 2:
            final_prediction = "up"
            img = 'img/arrow_pictures/pred_up.png'
        else:
            final_prediction = "down"
            img = 'img/arrow_pictures/pred_down.png'

        if client_socket is not None:
            print(final_prediction)
            message = final_prediction
            print(f"Sending: '{message}'")
            response = send_message(client_socket,message)
            print(f"Received: '{response}'")
        else:
            print('Error.')

        max_pred_val_rnd = np.round(max_pred_val,2)
        max_pred_val_to_string = str(max_pred_val_rnd)
        print(final_prediction)
        print(max_pred_val_to_string)
        
        return max_pred_val_to_string, img
    
    else:
        while num_chunks <5:
            print('Running eeg')
            eeg_sample, ts = stream.get_data()
            time.sleep(1.25)
            print(len(eeg_sample))
            eeg_chunks.append(eeg_sample[:,0:300])
            num_chunks +=1

        np_eeg = np.array(eeg_chunks)
        #print(np_test.shape)
        np_eeg_reshaped = np_eeg.reshape(num_chunks,300,20,1)
        #print(np_test_reshaped.shape)
        
        eeg_predictions = cog_model.predict(np_eeg_reshaped)
        print("eeg predictions:", eeg_predictions)
        avg_prediction = np.mean(eeg_predictions, axis=0)
        print("average prediction:", avg_prediction)

        max_pred_idx = np.argmax(avg_prediction)
        max_pred_val = avg_prediction[max_pred_idx]

        if max_pred_idx == 0:
            final_prediction = "left"
            img = 'img/arrow_pictures/pred_left.png'
        elif max_pred_idx == 1:
            final_prediction = "right"
            img = 'img/arrow_pictures/pred_right.png'
        elif max_pred_idx == 2:
            final_prediction = "up"
            img = 'img/arrow_pictures/pred_up.png'
        else:
            final_prediction = "down"
            img = 'img/arrow_pictures/pred_down.png'
        
        max_pred_val_rnd = np.round(max_pred_val,2)
        max_pred_val_to_string = str(max_pred_val_rnd)
        print(final_prediction)
        print(max_pred_val_to_string)
        return max_pred_val_to_string, img
    

def send_message(client_socket, message):
    ''' Thsi function sends and receive messages between the client (EEG computer) and the server (robot computer)'''
    client_socket.sendall(message.encode('utf-8'))
    response = client_socket.recv(1024).decode('utf-8')
    return response

@eel.expose
def close_socket():
    '''This function is important! Closing the socket appropriately will help minimize errors with the TCP connection.'''
    global client_socket
    if client_socket:
        client_socket.close()
        print('Socket closed.')

# Start Eel
eel.start('demo-main.html', size=(1000, 1000))

