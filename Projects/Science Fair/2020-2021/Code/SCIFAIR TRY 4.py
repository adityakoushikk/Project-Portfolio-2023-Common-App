import random
a = str('FIAG')
b = str('PSKP')
c = str('TESN')
d = str('CMTS')
e = str('DSFK')
f = str('GCFC')
g = str('IITT')
h = str('LLFN')
i = str('LLQY')
j = str('NLNE')
k = str('RLDK')
l = str('RLNE')
m = str('RLQS')
n = str('RVDF')
o = str('SEPV')
p = str('SLID')

for count in range (1,10000000):
    for word in a.split():
        randomPlacement1 = int(random.randint(0,3))
        randomPlacement2 = int(random.randint(0,3))
        randomPlacement3 = int(random.randint(0,3))
        words = ['M','F','I','F']
        words[randomPlacement1] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
        words[randomPlacement2] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
        words[randomPlacement3] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
        bruh = str(words[0] + words[1] + words[2] + words[3])
        if bruh == a or bruh == b or bruh == c or bruh == d or bruh == e or bruh == f or bruh == g or bruh == h or bruh == i or bruh ==j or bruh == k or bruh == l or bruh == m or bruh == n or bruh == o or bruh == p:
            print(count)
            print(bruh)
            break

