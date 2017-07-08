#!/usr/bin/env python

from object_tracking_msgs.srv import Recognize
from object_tracking_msgs.msg import Recognition, CategoryProbability, Hypothesis
from object_tracking_msgs.srv import Classify
import rospy


class segmenation_classification_bridge:

    def __init__(self):
        print("ros initialized!")
        self.service = rospy.Service('classify', Classify, self.classification_bridge)
        print("ros service Server startet.")
        self.classify = rospy.ServiceProxy('recognize', Recognize)
        print("created ros service proxy")


    def classification_bridge(self, req):
        print("got request")

        response = []
        for hypothese in req.objects:
            labels = []
            '''
            result = self.classify(hypothese)
            for r in result.recognitions:
                objects = sorted(r.categorical_distribution.probabilities, lambda x: x.probability)
                for p in objects:
                    #label = Hypothesis(label=p.label, reliability=p.probability)
                    labels.append(label)
                    
            '''
            label = Hypothesis(label="unknown", reliability=1)
            labels.append(label)
            response.append(labels)
        return {"hypotheses":response}

if __name__ == "__main__":
    rospy.init_node('segmentation_classification_bridge')
    seg = segmenation_classification_bridge()
    rospy.spin()
