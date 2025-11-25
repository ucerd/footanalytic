import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import threading
from tkinter import *
from PIL import ImageTk,Image
import serial.tools.list_ports
import csv
import functools
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time as app_time
import datetime
import pyautogui
from numpy import genfromtxt
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os
import time

app_running = True
m_time_string = "Time" #intialize string for time and date




m_cmap = ['coolwarm', 'coolwarm_r', 'Spectral', 'Spectral_r', 'YlGnBu','mako', 'BuPu','rocket', 'YlOrBr']
m_dd_hm_str=m_cmap[0]
###initialize global variables and parameter
m_app_type="Foot Weight Distribution";
row=16;
column=8;
plane=5;
m_time=0;
###Read Buffer
r_foot_buf_read=np.zeros((plane,row,column)) ### Right Foot buffer reads 10 values in a second
l_foot_buf_read=np.zeros((plane,row,column)) ### Left Foot buffer reads 10 values in a second



###Input Output Serial Communication Port Configuration

#for testing without hardware
#ports=serial.tools.list_ports.comports()
#serialObj = serial.Serial()
#serialObj.port = '/dev/ttyACM0'
#serialObj.baudrate = 115200
#serialObj.open()

#####
#Function receive data from input port (serial):
def heat_map_right_foot(app ,serialObj):
    app.update()  # Use to update the screen
    l_row = 0
    l_plane = 0
    global r_foot_buf_display

    while app_running:
        app.update()  # Use to update the screen
        
        raw_rcv = serialObj.readline()  # Reading line of 8 elements
        raw_rcv_tmp = raw_rcv.decode('utf').rstrip('\n').rstrip('\r').rstrip('').split(',')
       
        if raw_rcv_tmp == ['$$$$']:
            l_row = 0
        if np.shape(raw_rcv_tmp) == (8,):
            r_foot_buf_read[l_plane][l_row] = raw_rcv_tmp
            l_row += 1
            if l_row == 16:
                l_plane += 1
                l_row = 0
            if l_plane == 5:
                r_foot_buf_initial = np.average(r_foot_buf_read, axis=0)
                l_plane = 0
                r_foot_buf_avg = np.average(r_foot_buf_read, axis=0)
                r_foot_buf_display = r_foot_buf_avg.astype(int)
                ax1.clear()
                sns.heatmap(r_foot_buf_display, ax=ax1, linewidth=0.1, cmap='coolwarm', cbar=False, annot=True, fmt="d")
                hm_canvas.draw()

######
###Input Output Serial Communication Port Configuration
ports=serial.tools.list_ports.comports()
#for testing without hardware
#serialObj1 = serial.Serial()
#serialObj1.port = '/dev/ttyACM1'
#serialObj1.baudrate = 115200
#serialObj1.open()

def heat_map_left_foot(app ,serialObj1):
    app.update()  # Use to update the screen
    l_row = 0
    l_plane = 0
    global l_foot_buf_display

    while app_running:
        app.update()  # Use to update the screen
        
        raw_rcv = serialObj1.readline()  # Reading line of 8 elements
        raw_rcv_tmp = raw_rcv.decode('utf').rstrip('\n').rstrip('\r').rstrip('').split(',')
       
        if raw_rcv_tmp == ['$$$$']:
            l_row = 0
        if np.shape(raw_rcv_tmp) == (8,):
            l_foot_buf_read[l_plane][l_row] = raw_rcv_tmp
            l_row += 1
            if l_row == 16:
                l_plane += 1
                l_row = 0
            if l_plane == 5:
                l_foot_buf_initial = np.average(l_foot_buf_read, axis=0)
                l_plane = 0
                l_foot_buf_avg = np.average(l_foot_buf_read, axis=0)
                l_foot_buf_display = l_foot_buf_avg.astype(int)
                ax2.clear()
                sns.heatmap(l_foot_buf_display, ax=ax2, linewidth=0.1, cmap='coolwarm', cbar=False, annot=True, fmt="d")
                hm_canvas.draw()

# Create threads for both functions
thread_right_foot = threading.Thread(target=heat_map_right_foot, args=())#serialObj,))
thread_left_foot = threading.Thread(target=heat_map_left_foot, args=())#serialObj1,))

# Start both threads
thread_right_foot.start()
thread_left_foot.start()

# Wait for both threads to finish
thread_right_foot.join()
thread_left_foot.join()             
#function that takes data statically from Csv File

def read_csv(filepath):
    data = []
    with open(filepath, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data.append(row)
    return data


def btn_static():
    app.update() #use to update the screen
    global m_user_analysis_str
    global m_choise_hm
    m_user_analysis_str=""
    file = filedialog.askopenfile(title="Please Select Right Foot CSV File",mode='r', filetypes=[('Right Foot', '*.csv')])
    if file:
        filepath = os.path.abspath(file.name)
        l_foot_buf_display = pd.read_csv(filepath, header=None).values.astype(int)
        
    file = filedialog.askopenfile(title="Please Select Left Foot CSV File", mode='r', filetypes=[('Left Foot', '*.csv')])
    if file:


        filepath = os.path.abspath(file.name)
        r_foot_buf_display = genfromtxt(filepath, delimiter=',' ,dtype=int)
        r_foot_buf_display = np.rot90(np.rot90((r_foot_buf_display)))
        right_data = read_csv(filepath)
        print(right_data)
        r_foot_buf_display = pd.read_csv(filepath, header=None).values.astype(int)


    ax1.cla()
    ax2.cla()
    sns.heatmap(r_foot_buf_display, ax=ax1, linewidth = .1 , cbar=False, annot = True, fmt="d")
    sns.heatmap(l_foot_buf_display, ax=ax2, linewidth = .1 , cbar=False, annot = True, fmt="d")
    sns.heatmap(r_foot_buf_display, ax=ax1, linewidth = .1 ,cmap = m_dd_hm_str, cbar=False, annot = True, fmt="d")
    sns.heatmap(l_foot_buf_display, ax=ax2, linewidth = .1 ,cmap = m_dd_hm_str, cbar=False, annot = True, fmt="d")
    hm_canvas.draw()
    ax1.set_title('Heatmap with ' + m_dd_hm_str + ' Colormap')  

    app.update()
    l_mean = np.mean(l_foot_buf_display)
    l_median = np.median(l_foot_buf_display)
    l_variance = np.var(l_foot_buf_display)
    l_std = np.std(l_foot_buf_display)
    l_q1, l_q3 = np.percentile(l_foot_buf_display, [25, 75])
    l_iqr = l_q3 - l_q1
    l_outliers = l_foot_buf_display[(l_foot_buf_display < (l_q1 - 1.5 * l_iqr)) | (l_foot_buf_display > (l_q3 + 1.5 * l_iqr))]
    l_outliers = list(set(l_outliers))

    r_mean = np.mean(r_foot_buf_display)
    r_median = np.median(r_foot_buf_display)
    r_variance = np.var(r_foot_buf_display)
    r_std = np.std(r_foot_buf_display)
    r_q1, r_q3 = np.percentile(r_foot_buf_display, [25, 75])
    r_iqr = r_q3 - r_q1
    r_outliers = r_foot_buf_display[(r_foot_buf_display < (r_q1 - 1.5 * r_iqr)) | (r_foot_buf_display > (r_q3 + 1.5 * r_iqr))]
    r_outliers = list(set(r_outliers))
    Label(app, text="Left Foot:", bg='#002060', fg="lightgreen", font=("Arial", 20, 'bold')).place(x=50, y=150)
    Label(app, text=f"Average  {l_mean:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=190)
    Label(app, text=f"Median  {l_median:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=210)
    Label(app, text=f"Variance  {l_variance:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=230)
    Label(app, text=f"StD  {l_std:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=250)
    Label(app, text=f"Q1  {l_q1:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=270)
    Label(app, text=f"Q3  {l_q3:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=290)
    Label(app, text=f"IQR  {l_iqr:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=310)
   # Label(app, text=f"Outliers  {', '.join(map(str, l_outliers))}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=330)

    Label(app, text="Right Foot:", bg='#002060', fg="lightgreen", font=("Arial", 20, 'bold')).place(x=50, y=350)
    Label(app, text=f"Average  {r_mean:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=390)
    Label(app, text=f"Median  {r_median:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=410)
    Label(app, text=f"Variance  {r_variance:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=430)
    Label(app, text=f"StD  {r_std:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=450)
    Label(app, text=f"Q1  {r_q1:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=470)
    Label(app, text=f"Q3  {r_q3:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=490)
    Label(app, text=f"IQR  {r_iqr:.2f}", bg='#002060', fg="lightgreen", font=("Arial", 14, 'bold')).place(x=50, y=510)
  
    ax1.cla()
    ax2.cla()
    sns.heatmap(r_foot_buf_display, ax=ax1, linewidth = .1 , cbar=False, annot = True, fmt="d")
    sns.heatmap(l_foot_buf_display, ax=ax2, linewidth = .1 , cbar=False, annot = True, fmt="d")
    sns.heatmap(r_foot_buf_display, ax=ax1, linewidth = .1 ,cmap = m_dd_hm_str, cbar=False, annot = True, fmt="d")
    sns.heatmap(l_foot_buf_display, ax=ax2, linewidth = .1 ,cmap = m_dd_hm_str, cbar=False, annot = True, fmt="d")
    hm_canvas.draw()
    ax1.set_title('Heatmap with ' + m_dd_hm_str + ' Colormap')  
   
    ####################################################################


#----------------Clocktime and date update function after every 1 seconds-----
def clock_time():
    app.update()
    while app_running:
        app.update()
        time=datetime.datetime.now()
        time = (time.strftime("%d  %b, %Y           %I:%M:%S  %p"))
        m_time_string.set(time)
        app_time.sleep(.5)
#----------------Save File Button -----------------
def save_file():
    m_name = m_patient_name.get()
    m_age = m_patient_age.get()
    m_weight = m_patient_weight.get()
    time = datetime.datetime.now()
    m_time = time.strftime("%H:%M:%S")
    
    # Ensure the patient name is non-empty and sanitize it for use in filenames
    if m_name:
        sanitized_name = m_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        right_filename = f'/home/haris/Desktop/FYP_Demo/Repository/Source_Code/patient_data_{sanitized_name}_{m_time}_Right_foot.csv'
        left_filename = f'/home/haris/Desktop/FYP_Demo/Repository/Source_Code/patient_data_{sanitized_name}_{m_time}_Left_foot.csv'
        
        np.savetxt(right_filename, r_foot_buf_display, delimiter=',')
        np.savetxt(left_filename, l_foot_buf_display, delimiter=',')
        
        print(f"Data saved for patient {m_name}. Right foot data: {right_filename}, Left foot data: {left_filename}")
    else:
        print("Patient name is empty. Please enter the patient name.")

   

#-----------------HEAT MAP FUNCTION FOR SCREEN-------------------


def toggle():
    app.update()
    if btn_tg_fw_wa.config('relief')[-1] == 'sunken':
        btn_tg_fw_wa.config(relief="raised")
        print("Foot Weight Distribution")
        m_app_type='Foot Weight Distribution';

    else:
        btn_tg_fw_wa.config(relief="sunken")
        print("Walk Analysis")
        m_app_type='Foot Weight Distribution';


def exit_window():
    try:
        global app, hm_canvas
        hm_canvas.get_tk_widget().forget()
        app.destroy()
    except Exception as e:
        pass
       
def drop_down_menu_heatmap(m_choise_hm):
    global m_dd_hm_str
    m_dd_hm_str=variable.get()

   


#############GUI Work#####################



#-----------------MAIN CODE FOR BEGIN SCREEEN AND APP-------------
app = Tk()
app.title('Step Scan')    
app.geometry("1280x700")
app.configure(bg="black")
bg= ImageTk.PhotoImage(file="foot_design_pic_new.jpg")
canvas = Canvas(app, width=1280, height=700)
canvas.pack(fill=BOTH, expand=True)
canvas.create_image(0,0,image=bg, anchor=NW)

#-----------------MAIN SCREEN CANVAS INITIALIZER-----------------

fig = Figure();
ax1=fig.add_subplot(121)   #IMAGE SIZE, FIRST ROW, FIRST COLOUMN AND ONE FIGURE(111)
ax2=fig.add_subplot(122)   #IMAGE SIZE, FIRST ROW, FIRST COLOUMN AND ONE FIGURE(111)


fig.set_facecolor("#002060")# fACE COLOUR
#Creating Heat Map Foot Canvas
hm_canvas = FigureCanvasTkAgg(fig, master=app)  # A tk.DrawingArea.
hm_canvas.get_tk_widget().place(x = 500, y=50, width = 800, height = 520)# DIMESIONS OF DRAWING AREA
hm_canvas.draw()# THEN DRAW


import tkinter as tk

#-----------------Main screen button for heat map----------------
#btn_heat_mapr = Button(text='HEAT MAP Right', command=lambda: heat_map_right_foot(app, serialObj), bg='#70AD47', fg='white', font=10)
my_img7 = tk.PhotoImage(file = "./Heatmap_right.png") 
btn_heat_mapr=tk.Button(image=my_img7,command=lambda: heat_map_right_foot(app, serialObj),bd=0,bg='#ffffff',activebackground='#ffffff')

#btn_heat_mapl = Button(text='HEAT MAP Left', command=lambda: heat_map_left_foot(app, serialObj1), bg='#70AD47', fg='white', font=10)
my_img6 = tk.PhotoImage(file = "./Heatmap_left.png") 
btn_heat_mapl=tk.Button(image=my_img6,command=lambda: heat_map_left_foot(app, serialObj1),bd=0,bg='#ffffff',activebackground='#ffffff')

#btn_save = Button(text='          SAVE CSV         ', command=save_file, bg='#70AD47',fg='white',font= 10)
my_img = tk.PhotoImage(file = "./SAVE.png") 
btn_save=tk.Button(image=my_img,command=save_file,bd=0,bg='#ffffff',activebackground='#ffffff')

#btn_static = Button(text='Static Analysis', command=btn_static, bg='#70AD47',fg='white',  font=10)
my_img4 = tk.PhotoImage(file = "./Static_Analysis.png") 
btn_static=tk.Button(image=my_img4,command=btn_static,bd=0,bg='#ffffff',activebackground='#ffffff')
#btn_dynamic= Button(text='Dynamic Analysis', command=save_file, bg='#70AD47',fg='white',font= 10)
my_img5 = tk.PhotoImage(file = "./Dynamic_Analysis.png") 
btn_dynamic=tk.Button(image=my_img5,command=save_file,bd=0,bg='#ffffff',activebackground='#ffffff')
my_img1 = tk.PhotoImage(file = "./PATIENT_nAME.png") 
Patient_Name=tk.Button(image=my_img1,bd=0,bg='#ffffff',activebackground='#ffffff')
#Patient_Name = Button(text='Patient Name:',  bg='#70AD47',fg='white',font= 5)
#Age = Button(text='       Age:        ',  bg='#70AD47',fg='white',font= 5)
my_img2 = tk.PhotoImage(file = "./AGE.png") 
Age=tk.Button(image=my_img2,bd=0,bg='#ffffff',activebackground='#ffffff')
#Weight = Button(text='     Weight:     ',  bg='#70AD47',fg='white',font= 5)
my_img3 = tk.PhotoImage(file = "./Weight.png") 
Weight=tk.Button(image=my_img3,bd=0,bg='#ffffff',activebackground='#ffffff')

canvas.create_window(130, 600, window=btn_static)
canvas.create_window(380, 600, window=btn_dynamic)
canvas.create_window(700, 650, window=btn_heat_mapl)
canvas.create_window(930, 650, window=btn_heat_mapr)
canvas.create_window(1150, 650, window=btn_save)
canvas.create_window(80, 30, window=Patient_Name)
canvas.create_window(80, 80, window=Age)
canvas.create_window(80, 130, window=Weight)

#### User Data Input Entery
m_patient_name = Entry(app, width=30)
m_patient_name.place(x=170, y=20)  # Adjust the y-coordinate to prevent overlap
m_patient_name.insert(0, "")

m_patient_age = Entry(app, width=30)
m_patient_age.place(x=170, y=70)  # Adjust the y-coordinate as needed
m_patient_age.insert(0, "")

m_patient_weight = Entry(app, width=30)
m_patient_weight.place(x=170, y=120)  # Adjust the y-coordinate as needed
m_patient_weight.insert(0, "")

###-----------------Main screen time and data slot alocation----------

#-----------------Main screen time and data slot alocation----------
app.after(500,clock_time)# functions run after 500 ms here
m_time_string= StringVar()#intialize string for time and date
#m_app_type=StringVar()

lbl = Label(app, textvariable=m_app_type, bg= '#4472C4',fg="white", font=("Arial",20)).place(x=10,y=15) # initialize



my_img9 = tk.PhotoImage(file = "./close1.png") 
btn_close=tk.Button(image=my_img9,command=exit_window,bd=0,bg='#ffffff',activebackground='#ffffff')
canvas.create_window(1100, 25, window=btn_close)

# setting variable for Integers
variable = StringVar()
variable.set(m_cmap[0])

# creating widget
dropdown = OptionMenu(
    app,
    variable,
    *m_cmap,
    command=drop_down_menu_heatmap
)


# positioning widget
dropdown.pack(expand=True)
dropdown.place(x=200,y=650)


app.protocol("WM_DELETE_WINDOW", exit_window)
app.mainloop()
