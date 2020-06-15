from tkinter import *
import sqlite3
import bcrypt
from PIL import ImageTk,Image
from tkinter import messagebox
from tkinter import filedialog
from enum import Enum
import os

class User_type(Enum):
    ADMIN = 1
    NORMAL = 2


def logging_in():
    user_get = user_entry.get()#Retrieve Username
    pass_get = pass_entry.get()#Retrieve Password\
    exist_user, user_permission = is_user(user_get,pass_get)
    if exist_user:
        if (user_permission == User_type.ADMIN.value):
            raise_frame(admin, "250x670" )
        else: 
            raise_frame(principal,"700x700")
    else: 
        messagebox.showerror("Login Failed!", "User or password is not correct")

def raise_frame(frame, size):
    root.geometry(size)
    frame.tkraise()

def is_user(user, password):
    passwordToCompare = str.encode(password)
    conn = None
    try:
        conn = sqlite3.connect("theDatabase.db")
    except Error as e:
        print(e)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * from USER
     """)
    rows = cursor.fetchall()
    for row in rows:
        userToCompare = row[1]
        hashed_pass = row[2]
        user_permission = row[3]
        if user == userToCompare and bcrypt.checkpw(passwordToCompare, hashed_pass):
            return True, user_permission
    return False, None

# showinfo, showwarning, showerror, askquestion, askokcancel, askyesno
def more_images():
    response = messagebox.askokcancel("Images destination!", "Your images will be add to the last active model")
    if response:
        files = filedialog.askopenfilenames(initialdir="resources", title="Select A File", filetypes=(("nii files", "*.nii.gz"),))
        if (len(files)):
            messagebox.showinfo("Images destination", "Your images were added correctly")

def only_numbers(char):
    return char.isdigit()

def new_model():
    top = Toplevel()
    top.resizable(False, False)
    top.title("Model Parameters")

    validation = root.register(only_numbers)

    model_name_label = Label(top, text="Name: ")
    model_name_label.grid(row=0, column=0, sticky=W, pady=5, padx=(15,5))

    model_name = Entry(top)
    model_name.grid(row=0, column=1, padx=(3,10))

    model_ysize_label = Label(top, text="Y size: ")
    model_ysize_label.grid(row=1, column=0, sticky=W, pady=5, padx=(15,5))

    model_ysize = Entry(top,  validate="key", validatecommand=(validation, '%S'))
    model_ysize.grid(row=1, column=1, padx=(3,10))

    model_zsize_label = Label(top, text="Z size: ")
    model_zsize_label.grid(row=2, column=0, sticky=W, pady=5, padx=(15,5))

    model_zsize = Entry(top, validate="key", validatecommand=(validation, '%S'))
    model_zsize.grid(row=2, column=1, padx=(3,10))

    
    button_add_model = Button(top, text="Sign In", width=10, borderwidth=3, command=lambda: add_model(top))
    button_add_model.grid(row=3, column=0, columnspan=2, pady=5)

def add_model(top):
    top.destroy()
    messagebox.showinfo("Model training", "Your new model will be trained in background")

def go_to_principal():
    raise_frame(principal,"700x700")

#Main:
root = Tk()
root.title("Brainy")
if "nt" == os.name:
    root.wm_iconbitmap(bitmap = "resources/brain_ico.ico")
else:
    root.wm_iconbitmap(bitmap = "@resources/brain_ico.xbm")
#root.resizable(False, False)
login = Frame(root)
admin = Frame(root)
principal = Frame(root)


for frame in (login, admin, principal):
    frame.grid(row=0, column=0, sticky='nsew')

## Login
icon = Image.open("resources/logotype.png")
icon= icon.resize((70,70), Image.ANTIALIAS)
icon_img = ImageTk.PhotoImage(icon)

title = Label(login, image=icon_img)
title.grid(row=0, column=1, columnspan=2, pady=3)

user_entry_label = Label(login, text="Username: ")
user_entry_label.grid(row=1, column=1, sticky=W, pady=5, padx=(15,5))

user_entry = Entry(login)
user_entry.grid(row=1, column=2, padx=(3,10))

pass_entry_label = Label(login, text="Password: ")#PASSWORD LABEL
pass_entry_label.grid(row=2, column=1, sticky=W,padx=(15,5), pady=5)

pass_entry = Entry(login, show="*")#PASSWORD ENTRY BOX
pass_entry.grid(row=2, column=2,  padx=(3,10))

sign_in_butt = Button(login, text="Sign In",command = logging_in, width=10, borderwidth=3)#SIGN IN BUTTON
sign_in_butt.grid(row=5, column=1, columnspan=2, pady=5)

## Admin

icon0_raw = Image.open("resources/plus.png")
icon0_raw = icon0_raw.resize((150,150), Image.ANTIALIAS)
icon0_img = ImageTk.PhotoImage(icon0_raw )
button0 = Button(admin, image=icon0_img, borderwidth=3, command=more_images)
button0.grid(row=0, column=0, padx=15, pady=15)
label_admin_0 = Label(admin, text="Add new images to actual model", font=("", 12))
label_admin_0.grid(row=1, column=0, padx=(5,0))

icon1_raw = Image.open("resources/neural_network.png")
icon1_raw = icon1_raw.resize((150,150), Image.ANTIALIAS)
icon1_img = ImageTk.PhotoImage(icon1_raw )
button1 = Button(admin, image=icon1_img, borderwidth=3, command=new_model)
button1.grid(row=2, column=0, padx=15, pady=15)
label_admin_1 = Label(admin, text="Add new deep learning model", font=("", 12))
label_admin_1.grid(row=3, column=0, padx=(5,0))

icon2_raw = Image.open("resources/right_arrow.png")
icon2_raw = icon2_raw.resize((150,150), Image.ANTIALIAS)
icon2_img = ImageTk.PhotoImage(icon2_raw)
button2 = Button(admin, image=icon2_img, borderwidth=3, command=go_to_principal)
button2.grid(row=4, column=0, padx=15, pady=15)
label_admin_2 = Label(admin, text="Go to visualization", font=("", 12))
label_admin_2.grid(row=5, column=0, padx=(5,0))

raise_frame(login, "235x180")
root.mainloop()#Keeps the window open/running

# The principal


