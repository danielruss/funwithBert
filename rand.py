import random

random.seed(4);
print( [ random.randrange(0,10) for i in range(1,20) ] )
print( [ random.randrange(0,10) for i in range(1,20) ] )
print( [ random.randrange(0,10) for i in range(1,20) ],"\n" )

random.seed(4);
print( [ random.randrange(0,10) for i in range(1,20) ] )
print( [ random.randrange(0,10) for i in range(1,20) ] )
print( [ random.randrange(0,10) for i in range(1,20) ] )

characters = [ chr(i) for i in range(97,123)]
print( "\n",characters,"\n")

random.seed(4);
cc = characters.copy()

random.shuffle(cc);print( cc[0:20] )
random.shuffle(cc);print( cc[0:20] )
random.shuffle(cc);print( cc[0:20] )

random.seed(4);
cc = characters.copy()
print()
random.shuffle(cc);print( cc[0:20] )
random.shuffle(cc);print( cc[0:20] )
random.shuffle(cc);print( cc[0:20] )
