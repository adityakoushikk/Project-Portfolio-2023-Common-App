import random
for count in range (1,1000000):
    rand = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])+random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
    rad = str(rand)
    print(rad)
    #print(bigboi)
    for word in rad.split():
        if rad == 'FLIA':
            print(count)

    

    


            

