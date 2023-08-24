import random
import matplotlib
import matplotlib.pyplot as plt
from numpy import mean

a = str('VLYQGVNCT')
rX = []
cY = []

for count in range (1,2):
    rX.append(count)
    for count in range (1,10000001):
        for word in a.split():
            randomPlacement1 = int(random.randint(0,8))
            randomPlacement2 = int(random.randint(0,8))
            randomPlacement3 = int(random.randint(0,8))
            randomPlacement4 = int(random.randint(0,8))
            randomPlacement5 = int(random.randint(0,8))
            randomPlacement6 = int(random.randint(0,8))
            randomPlacement7 = int(random.randint(0,8))
            words = ['V','L','N','D','I','L','S','R','L']
            words[randomPlacement1] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            words[randomPlacement2] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            words[randomPlacement3] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            words[randomPlacement4] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            words[randomPlacement5] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            words[randomPlacement6] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            words[randomPlacement7] = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            bruh = str(words[0] + words[1] + words[2] + words[3] + word[4] + words[5] + words[6] + words[7] + words[8])
        if bruh == a:
            cY.append(count)
            print(count)
            break
'''        
avg = mean(cY)
print(round(avg,2))





plt.plot(rX,cY)
plt.xlabel('Number of Runs')
plt.ylabel('Cycles required to change D614 to G per run')
plt.show()'''
