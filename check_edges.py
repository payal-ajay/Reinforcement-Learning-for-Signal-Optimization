import sumolib

# Load your network file
net = sumolib.net.readNet("simple_intersection.net.xml")

print(" Available Edges in the Network:")
for edge in net.getEdges():
    print(edge.getID())

