import pandas
from subprocess import call
import os

class decorator:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\x1B[3m'
    NORMAL = '\033[0m'

def clear():
    x = input("Press Enter to Continue")
    os.system('cls' if os.name=='nt' else 'clear')

def add_contact():
    print(decorator.BOLD + decorator.UNDERLINE + "Enter the Contact Details." + decorator.NORMAL)
    name = input(decorator.ITALIC + "Enter Contact Name : " + decorator.NORMAL)
    try:
        v = df.loc[name]
    except KeyError:
        email = input(decorator.ITALIC + "Enter Contact Email Id : " + decorator.NORMAL)
        phone = input(decorator.ITALIC + "Enter Contact Phone No. : " + decorator.NORMAL)
        n = name + "\t" + email + "\t" + phone + "\n"
        f = open('data.txt', 'a')
        f.write(n)
        f.close()
        print("Contact Successfully Created!\n")
    else:
        print("Contact Name already Exists.")
        print("\n")
        add_contact()

def display_contact():
    if df.empty:
        print("Address Book is empty.\n")
        return
    print(df)

def search_contact():
    if df.empty:
        print("Address Book is Empty.\n")
    name = input(decorator.ITALIC + "Enter Contact Name : " + decorator.NORMAL)
    try:
        v = df.loc[name]
        email = df.at[name, "Contact_email"]
        phone = df.at[name, "Contact_phone"]
        print(decorator.BOLD + decorator.UNDERLINE + "Contact Details." + decorator.NORMAL + "\n")
        print(decorator.ITALIC + "Contact name : " + decorator.NORMAL + str(name))
        print(decorator.ITALIC + "Email Id : " + decorator.NORMAL + str(email))
        print(decorator.ITALIC + "Phone No. : " + decorator.NORMAL + str(phone) + "\n")
    except KeyError:
        print("No Contact with this Name found!\n")

def modify_contact():
    if df.empty:
        print("Address Book is Empty.\n")
    name = input(decorator.ITALIC + "Enter Contact Name to be Modified : " + decorator.NORMAL)
    try:
        v = df.loc[name]
        delete(name)
        name = input(decorator.ITALIC + "Enter Contact New Name : " + decorator.NORMAL)
        email = input(decorator.ITALIC + "Enter Contact New Email Id : " + decorator.NORMAL)
        phone = input(decorator.ITALIC + "Enter Contact New Phone No. : " + decorator.NORMAL)
        n = name + "\t" + email + "\t" + phone + "\n"
        f = open('data.txt', 'a')
        f.write(n)
        f.close()
        print("Contact Details Successfully Modified!\n")
    except KeyError:
        print("No Contact with this Name found!\n")

def delete(name):
    f = open('data.txt', 'r')
    v = []
    for i in f.readlines():
        v.append(i)
    f.close()
    for i in range(len(v)):
        x = v[i].split("\t")
        if x[0] == name:
            v.pop(i)
            break
    f = open('data.txt', 'w')
    for i in v:
        f.write(i)
    f.close()

def delete_contact():
    if df.empty:
        print("Address Book is Empty.\n")
    name = input(decorator.ITALIC + "Enter Contact Name to be Deleted : " + decorator.NORMAL)
    try:
        v = df.loc[name]
        delete(name)
        print("\nContact Successfully Deleted.")
    except KeyError:
        print("No Contact with this Name found!\n")


print(decorator.ITALIC + decorator.UNDERLINE + decorator.BOLD + "Welcome to Address Book." + decorator.NORMAL)
val = None
while val != 0:
    print("Enter 1 to Add a Contact.\tEnter 2 to Display Contacts.\nEnter 3 to Delete a Contact.\tEnter 4 to Modify "
          " a Contact.\nEnter 5 to Search a Contact.\tEnter 0 to Exit.\n")
    val = input("Enter a option : ")
    df = pandas.read_csv(
        'data.txt', sep="\t", index_col="Contact_name")

    if val == '1':
        add_contact()
        print("\n")
    elif val == '2':
        display_contact()
        print("\n")
    elif val == '3':
        delete_contact()
        print("\n")
    elif val == '4':
        modify_contact()
        print("\n")
    elif val == '5':
        search_contact()
        print("\n")
    elif val == '0':
        print("EXIT !")
        break
    else:
        print("Invalid option.")

    clear()

