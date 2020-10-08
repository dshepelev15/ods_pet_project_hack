import dlib


class LandmarksDetector:
    detector = None
    shape_predictor = None

    def __init__(self, predictor_model_path=None):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        if predictor_model_path is not None:
            LandmarksDetector.detector = dlib.get_frontal_face_detector()  # cnn_face_detection_model_v1 also can be used
            LandmarksDetector.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks
