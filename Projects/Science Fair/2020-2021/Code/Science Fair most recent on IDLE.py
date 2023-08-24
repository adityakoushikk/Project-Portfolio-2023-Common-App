import random

a = str('YLQPRTFLL')
b = str('FIAGLIAIV')
c = str('TLDSKTQSL')
d = str('LTDEMIAQY')
e = str('RLFRKSNLK')
f = str('NLNESLIDL')
g = str('VLNDILSRL')
h = str('RLNEVAKNL')
i = str('TPINLVRDL')
j = str('ALNTLVKQL')
k = str('LITGRLQSL')
l = str('GTITSGWTF')
m = str('KEIDRLNEV')
n = str('NQKLIANQF')
o = str('RISNCVADY')
p = str('FQPTNGVGY')
q = str('TSNQVAVLY')
r = str('SPRRARSVA')
s = str('YEQYIKWPW')
t = str('VLYQDVNCT')

for count in range (1,10000000):
    for word in a.split():
        random_amount = int(random.randint(0, 9))

        words = {'V': '0', 'L': '1', 'Y': '2', 'Q': '3', 'D': '4', 'V': '5', 'N': '6', 'C': '7', 'T': '8'}
        words_list = ['V', 'L', 'Y', 'Q', 'D', 'V', 'N', 'C', 'T']
        numbers_list = ['0','1','2','3','4','5','6','7','8',]

        for count in range (0,random_amount):
            print("sdf")
            random_placement = random.randint(0,9)
            integer = int(random_placement)
            if random_placement not in numbers_list:
                while True:
                    if random_placement in numbers_list:
                        break
                    else:
                        randomPlacement = random.randint(0, 9)
            else:
                del numbers_list[random_placement]
                print(numbers_list)





            words_list[integer] = random.choice(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])






        new_word = str(words_list[0] + words_list[1] + words_list[3] + words_list[4] + words_list[5] + words_list[6] + words_list[7] + words_list[8])

        if new_word == a or new_word == b or new_word == c or new_word == d or new_word == e or new_word == f or new_word == g or new_word == h or new_word == i or new_word == j or new_word == k or new_word == l or new_word == m or new_word == n or new_word == o or new_word == p or new_word == q or new_word == s or new_word == t:
            print(count)
            print(new_word)
            break
