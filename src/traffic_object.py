import depthai as dai

class TrafficObject:
    def __init__(self, detection, frame):
        self.x_max = 0
        self.y_max = 0
        self.y_min = 0
        self.x_min = 0
        self.x_depth = 0
        self.y_depth = 0
        self.z_depth = 0 
        self.confidence =0
        self.label = ""
    
    def getBoudingBoxCoordinates(self):
        return [self.x_max,self.y_max,self.x_min, self.y_min]
    def getDepthCoordinates(self):
        return [self.x_depth,self.y_depth,self.z_depth]


if __name__ == "__main__":
    print("Module Traffic Object Started")
    Person = TrafficObject()
    print("Bounding Box coordinates",Person.getBoudingBoxCoordinates())
    print("Depth coordinates",Person.getDepthCoordinates())