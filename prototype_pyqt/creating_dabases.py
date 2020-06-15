import sqlite3
from sqlite3 import Error
from enum import Enum
import bcrypt
import datetime
import os

class User_type(Enum):
    ADMIN = 1
    NORMAL = 2

def sql_connection():
    try:
        con = sqlite3.connect('theDatabase.db')
        return con
    except Error:
        print(Error)

def create_tables(con):
    cursorObj = con.cursor()
    cursorObj.execute("CREATE TABLE user(id integer PRIMARY KEY, username text, password text, type tinyint)")
    con.commit()

    cursorObj.execute("CREATE TABLE model(id integer PRIMARY KEY, name text UNIQUE, Y integer, Z integer, path text, creationDate timestamp, executionDate timestamp)")
    con.commit()

    cursorObj.execute("CREATE TABLE image(id integer PRIMARY KEY, name text, pathImage text, pathMask text)")
    con.commit()

def create_default_users(con):
    cursorObj = con.cursor()
    sqlite_insert = """INSERT INTO 'user'
                        ('username', 'password', 'type') 
                        VALUES (?, ?, ?);"""
    passwd = b'admin123'
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(passwd, salt)
    data_tuple_admin = ('admin', hashed, User_type.ADMIN.value)
    cursorObj.execute(sqlite_insert, data_tuple_admin)
    con.commit()

    passwd2 = b'guest123'
    hashed_common = bcrypt.hashpw(passwd2, salt)
    data_tuple_guest = ('guest', hashed_common, User_type.NORMAL.value)
    cursorObj.execute(sqlite_insert, data_tuple_guest)
    con.commit()

def create_default_models(con):
    cursorObj = con.cursor()
    sqlite_insert = """INSERT INTO 'model'
                        ('name', 'path', 'Y', 'Z','creationDate', 'executionDate') 
                        VALUES (?, ?, ?, ?, ?, ?);"""

    default_y = 200
    default_z = 200
    
    name_1 = 'normal_85_15'
    path_1 = '../core/models/Full_Flair_T1_5_Augmentation/_85_15.h5'
    absolute_path_1 = os.path.abspath(path_1)
    creationDate_1 = datetime.datetime.now()
    executionDate_1 = datetime.datetime.now()

    name_2 = 'mult_85_15'
    path_2 = '../core/models/multData/_complete__multData.h5'
    absolute_path_2 = os.path.abspath(path_2)
    creationDate_2 = datetime.datetime.now()
    executionDate_2 = datetime.datetime.now()

    name_3 = 'flair_85_15'
    path_3 = '../core/models/onlyFlair/_complete__onlyFlair.h5'
    absolute_path_3 = os.path.abspath(path_3)
    creationDate_3 = datetime.datetime.now()
    executionDate_3 = datetime.datetime.now()

    data_tuple_1 = (name_1, absolute_path_1, default_y, default_z, creationDate_1, executionDate_1)
    cursorObj.execute(sqlite_insert, data_tuple_1)
    normal_model_id = cursorObj.lastrowid
    con.commit()

    data_tuple_2 = (name_2, absolute_path_2,default_y, default_z, creationDate_2, executionDate_2)
    cursorObj.execute(sqlite_insert, data_tuple_2)
    mult_model_id = cursorObj.lastrowid
    con.commit()

    data_tuple_3 = (name_3, absolute_path_3, default_y, default_z, creationDate_3, executionDate_3)
    cursorObj.execute(sqlite_insert, data_tuple_3)
    normal_model_id = cursorObj.lastrowid
    con.commit()

def create_default_images(con):
    cursorObj = con.cursor()
    sqlite_insert = """INSERT INTO 'image'
                    ('name', 'pathImage', 'pathMask') 
                    VALUES (?, ?, ? );"""
    directory1 = '../generatedData/flair_pre/'
    masksDirectory = '../generatedData/masks/masks_preprocessed/'
    abs_directory1 = os.path.abspath(directory1)
    abs_masksDirectory = os.path.abspath(masksDirectory)
    directory1_array = []
    directory2_array = []
    for filename in os.listdir(abs_directory1):
        path = abs_directory1 + os.sep + filename
        mask_path = abs_masksDirectory + os.sep + filename
        name = filename.split('.')[0]
        the_tuple_1 = (name, path, mask_path)
        directory1_array.append(the_tuple_1)
    cursorObj.executemany(sqlite_insert, directory1_array)
    con.commit()

def delete_connection(con):
    con.close()

if os.path.isfile('theDatabase.db'):
    os.remove('theDatabase.db')
con = sql_connection()
create_tables(con)
create_default_users(con)
create_default_models(con)
create_default_images(con)
delete_connection(con)