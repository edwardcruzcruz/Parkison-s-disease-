import sqlite3
from sqlite3 import Error
from enum import Enum
import bcrypt
import datetime

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

    cursorObj.execute("CREATE TABLE model(id integer PRIMARY KEY, name text UNIQUE, path text, creationDate timestamp, executionDate timestamp)")
    con.commit()

    cursorObj.execute("CREATE TABLE image(id integer PRIMARY KEY, name text, path text, model integer, FOREIGN KEY(model) REFERENCES model(text)")
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
                        ('name', 'path', 'creationDate', 'executionDate') 
                        VALUES (?, ?, ?, ?);"""
    
    name_1 = 'normal_85_15'
    path_1 = 'D:/Programacion/Integradora/gitFolder/core/models/Full_Flair_T1_5_Augmentation/_85_15.h5'
    creationDate_1 = datetime.datetime.now()
    executionDate_1 = datetime.datetime.now()

    name_2 = 'mult_85_15'
    path_2 = 'D:/Programacion/Integradora/gitFolder/core/models/multData/_complete__multData.h5'
    creationDate_2 = datetime.datetime.now()
    executionDate_2 = datetime.datetime.now()

    data_tuple_1 = (name_1, path_1, creationDate_1, executionDate_1)
    cursorObj.execute(sqlite_insert, data_tuple_1)
    con.commit()

    data_tuple_2 = (name_2, path_2, creationDate_2, executionDate_2)
    cursorObj.execute(sqlite_insert, data_tuple_2)
    con.commit()

def delete_connection(con):
    con.close()


con = sql_connection()
create_tables(con)
create_default_users(con)
delete_connection(con)