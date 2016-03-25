small_steps = [100*x for x in range(1, 10)] 
large_steps = [1000*x for x in range(1, 10)]
increments = []
increments.extend(small_steps)
increments.extend(large_steps)
for size in increments:
    print "echo \"--------n=" + str(size) + "\""
    if False: # testing mode with self diff
        print( "./gpu -n " + str(size) + " -o gpu-" +str(size) + ".txt" )
        print( "./serial -n " + str(size) + " -o serial-" +str(size) + ".txt" )
        print( "diff gpu-" + str(size) + ".txt serial-" +str(size) + ".txt | wc" )
    else:
        print( "./gpu -n " + str(size) )
        print( "./serial -n " + str(size) )
