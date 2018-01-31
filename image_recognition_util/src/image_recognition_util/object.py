

class Object():
    def __init__(self,label,prob,xmin,xmax,ymin,ymax):
        self.label = label
        self.prob = prob
        self.bbox = BoundingBox(xmin,xmax,ymin,ymax)

class BoundingBox():
    def __init__(self,xmin,xmax,ymin,ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax