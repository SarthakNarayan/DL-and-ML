import matplotlib.pyplot as plt

x = [1,2,3]
y = [1,2,3]
a =  [5,6,7]
b =  [5,6,7]
# will plot points depending on x and y coordinates
# color is used to specify the color of the lines
# both the inputs should be of the same length
plt.scatter(x,y , label = 'X and Y' , color = 'r')
plt.scatter(a,b , label = 'A and B' , color = 'c')
# x and y labels are for naming x and y axis
plt.xlabel('X points')
plt.ylabel('Y points')
# title is for giving a name to the graph
plt.title('1st matplotlib graph')
# labels that u give will only work if you show legends
plt.legend()
# after plotting and adding all the details you show
plt.show()
