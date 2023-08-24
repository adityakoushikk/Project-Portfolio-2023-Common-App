import random
a = str('YLQPRTFLL')
b = str('FIAGLIAIV')
c = str('TLDSKTQSL')
e = str('LTDEMIAQY')

for count in range (1,10000000):
    for word in a.split():
        random_amount = int(random.randint(0,9))
        words = ['M','F','I','F','L','L','F','L','T']
        for count in range (0,random_amount):
            str(randomPlacement + count) = int(random.randint(0,9))
            words[randomPlacement + count] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])

        bruh = str(words[0] + words[1] + words[2] + words[3])
        if bruh == a or bruh == b or bruh == c or bruh == d or bruh == e or bruh == f or bruh == g or bruh == h or bruh == i or bruh ==j or bruh == k or bruh == l or bruh == m or bruh == n or bruh == o or bruh == p:
            print(count)
            print(bruh)
            break
