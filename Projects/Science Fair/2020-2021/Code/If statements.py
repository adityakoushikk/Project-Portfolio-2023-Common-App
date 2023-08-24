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



import random



for count in range (1,100000):
        for word in a.split():
            letterChange = 1
            letterChange1 = 2
            letterChange = int(random.randint(1,4))    
            if letterChange == 1:
                letterChange1 = int(random.choice(['2','3','4']))
            elif letterChange == 2:
                letterChange1 = int(random.choice(['1','3','4']))
            elif letterChange == 3:
                letterChange1 = int(random.choice(['1','2','4']))
            elif letterChange == 4:
                letterChange1 = int(random.choice(['1','2','3']))

            words = 'MFIF'
            if letterChange == 1 and letterChange1 == 2:
                words = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + "I" + "F"
            elif letterChange == 1 and letterChange1 == 3:
                words = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + "F" + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + "F"
            elif letterChange == 1 and letterChange1 == 4:
                words = random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + "F" + "I" + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            elif letterChange == 2 and letterChange1 == 3:
                words = "M" + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + "F"
            elif letterChange == 2 and letterChange1 == 4:
                 words = "M" + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + "I" + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])
            elif letterChange == 3 and letterChange1 == 4:
                words = "M" + "F" + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']) + random.choice(['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'])


        if words == a or words == b or words == c or words == d or words == e or words == f or words == g or words == h or words == i or words ==j or words == k or words == l or words == m or words == n or words == o or words ==p:
                print(count)
                break




    
        
                                                                                                                                                                                                                            

