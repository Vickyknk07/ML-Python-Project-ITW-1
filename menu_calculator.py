import os

class decorator:
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\x1B[3m'
    NORMAL = '\033[0m'


def clear():
    x = input("Press Enter to Continue")
    os.system('cls' if os.name=='nt' else 'clear')


Items = ['Chicken Strips', 'French Fries', 'Hamburger', 'Hotdog', 'Large Drink',
        'Medium Drink', 'Milk Shake', 'Salad', 'Small Drink']

Costs = [3.50, 2.50, 4.00, 3.50, 1.75, 1.50, 2.25, 3.75, 1.25]

print(decorator.BOLD + decorator.UNDERLINE + "MENU CALCULATOR" + decorator.NORMAL)
print("")
val = None
while True:
    print(decorator.BOLD + decorator.ITALIC + "MENU CARD" + decorator.NORMAL)
    print("")
    for i in range(len(Items)):
        print(decorator.ITALIC + str(i + 1) + ") " + Items[i] + " : $" + str(Costs[i]) + decorator.NORMAL)
    print("")
    print(decorator.ITALIC + "Enter Exit to Terminate." + decorator.NORMAL)
    print("")
    ord_val = input("Enter your Order : ")
    print("")
    if ord_val == 'Exit':
        break
    l = []
    flag = 0
    for i in ord_val:
        try:
            l.append(int(i) - 1)
        except ValueError:
            flag = 1
            break
    if flag == 1:
        print("Please Enter Integer values to Order. \n")
        clear()
        continue
    l.sort()
    dict = {}
    C = 0
    for i in l:
        if i >= len(Items) or i < 0:
            flag = 2
            break
        if i in dict:
            dict[i] += 1
        else:
            C += Costs[i]
            dict.update({i: 1})
    if flag == 2:
        print(decorator.ITALIC + "Item No. given is Unavailable.\n" + decorator.NORMAL)
        clear()
        continue
    print(decorator.BOLD + decorator.UNDERLINE + "ITEMS ORDERED" + decorator.NORMAL)
    for i in dict:
        print(decorator.ITALIC + str(Items[i]) + " : " + str(dict[i]) + decorator.NORMAL)
    print()
    print(decorator.BOLD + "Total Amount to be Paid : $" + str(C) + decorator.NORMAL)
    print()
    clear()