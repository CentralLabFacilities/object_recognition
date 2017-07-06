#!/usr/bin/env python

from object_recognition_msgs.srv import Recognize
from object_recognition_msgs.msg import Recognition, CategoryProbability, Hypothesis
from object_recognition_msgs.srv import Classify
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
        label = Hypothesis(labels=["unknown"], reliability=[1])
        for hypothese in req.objects:
            '''
            result = self.classify(hypothese)
            for r in result.recognitions:
                text_array = []
                best = CategoryProbability(label="unknown", probability=r.categorical_distribution.unknown_probability)
                for p in r.categorical_distribution.probabilities:
                    text_array.append("%s: %.2f" % (p.label, p.probability))
                    if p.probability > best.probability:
                        best = p
                        
                '''

            response.append(label)
        return {"hypotheses":response}

if __name__ == "__main__":
    rospy.init_node('segmentation_classification_bridge')
    seg = segmenation_classification_bridge()
    rospy.spin()