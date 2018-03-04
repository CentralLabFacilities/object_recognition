

class Object():
    def __init__(self,label,prob,xmin,xmax,ymin,ymax):
        self.label = label
        self.prob = prob
        self.bbox = BoundingBox(xmin,xmax,ymin,ymax)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other): # python version perhaps cmp
        return self.__dict__ == other.__dict__

class BoundingBox():
    def __init__(self,xmin,xmax,ymin,ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__