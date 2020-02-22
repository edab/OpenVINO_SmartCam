import argparse
import time

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore

# The dataset can be downloaded from here:
# https://web.bii.a-star.edu.sg/archive/machine_learning/Projects/FluRecognition/videos/biisc.zip

# The models used are:
# - action-recognition-0001-decoder
# - action-recognition-0001-encoder
# - human-pose-estimation-0001
# and can be downloaded using:
# /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name action-recognition-0001-decoder --precisions FP32 -o models
# /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name action-recognition-0001-encoder --precisions FP32 -o models
# /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name human-pose-estimation-0001 --precisions FP32 -o models

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
INPUT_STREAM = "dataset/biisc/videos/S003_M_COUG_WLK_FCE.avi"
HUMAN_POSE_MODEL = "models/intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml"
HUMAN_POSE_WEIGHTS = "models/intel/human-pose-estimation-0001/FP32/human-pose-estimation-0001.bin"
ACTION_RECOGNITION_MODEL = "models/intel/action-recognition-0001-decoder/FP32/action-recognition-0001-decoder.xml"

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

def handle_output(model_type):
    '''
    Returns the related function to handle an output,
        based on the model_type being used.
    '''
    if model_type == "POSE":
        return handle_pose
    elif model_type == "ACTION":
        return handle_action
    else:
        return None

def handle_pose(output, frame, args):
    '''
    Handles the output of the Pose Estimation model.
    Returns ONLY the keypoint heatmaps, and not the Part Affinity Fields.
    '''

    # Get only heatmaps from model output
    heatmaps = output['Mconv7_stage2_L2']

    # Check if the output are correct
    assert(len(BODY_PARTS) <= heatmaps.shape[1])

    # Init structures
    points = []
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    for i in range(len(BODY_PARTS)):

        # Slice heatmap of corresponding body's partRWrist.
        heatMap = heatmaps[0, i, :, :]

        # TODO: keep track of multiple poses
        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / heatmaps.shape[3]
        y = (frameHeight * point[1]) / heatmaps.shape[2]

        # Add a point if it's confidence is higher than threshold.
        points.append((int(x), int(y)) if conf > args.t else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    # TODO: Generate an event when LWrist or RWrist come close to Nose

    return frame

def handle_action(output, input_shape, args):

    # TODO: Implement the inference for the ACTION_RECOGNITION_MODEL
    print("Error: not yet implemented")
    quit()

def preprocessing(frame, H, W):
    '''
    Given an input image, height and width:1
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start
    '''
    image = np.copy(frame)
    image = cv2.resize(image, (W, H))
    image = image.transpose((2,0,1)) # Change data layout from HWC to CHW
    image = image.reshape(1, 3, H, W)

    return image

def create_output_image(model_type, image, output):
    '''
    Using the model type, input # TODO: Implement the inference for the ACTION_RECOGNITION_MODEL
    print("Error: not yet implemented")
    quit()image, and processed output,
    creates an output image showing the result of inference.
    '''

    if model_type == "POSE":

        # Remove final part of output not used for heatmaps
        output = output[:-1]

        # Get only pose detections above 0.5 confidence, set to 255
        for c in range(len(output)):
            output[c] = np.where(output[c]>0.5, int(255), int(0))

        # Sum along the "class" axis
        output = np.sum(output, axis=0)

        # Get semantic mask
        pose_mask = get_mask(output)

        # Combine with original image
        output_image = image + pose_mask

        #output_image = cv2.addWeighted(image, 0.7, pose_mask, 0.3, 0)

        return pose_mask

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")

    # Create the descriptions for the commands
    i_desc = "The location of the input file (default: 'dataset/biisc/videos/S003_M_COUG_WLK_FCE.avi')"
    d_desc = "Target device: CPU, GPU, FPGA, MYRIAD, MULTI:CPU,GPU, HETERO:FPGA,CPU (default: 'CPU')"
    m_desc = "The model to use ['POSE' or 'ACTION'] (default: 'POSE')"
    t_desc = "Threshold value for pose parts heat map (default: 0.5)"

    # Create the arguments
    parser.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    parser.add_argument("-d", help=d_desc, default='CPU')
    parser.add_argument("-m", help=m_desc, default='POSE')
    parser.add_argument('-t', help=t_desc, default=0.5, type=float)

    args = parser.parse_args()

    return args

def load_network(args):

    if args.m == 'POSE':

        print('Loading Pose Estimation Model....')

        # Initialise the class
        Network = IENetwork(model=HUMAN_POSE_MODEL, weights=HUMAN_POSE_WEIGHTS)

        # Get Input Layer Informationoutput_image
        PoseEstimationInputLayer = next(iter(Network.inputs))
        print("Pose Estimation Input Layer: ", PoseEstimationInputLayer)
        print(Network.inputs)

        # Get Output Layer Information
        PoseEstimationOutputLayer = next(iter(Network.outputs))
        print("Pose Estimation Output Layer: ", PoseEstimationOutputLayer)

        # Get Input Shape of Model
        PoseEstimationInputShape = Network.inputs[PoseEstimationInputLayer].shape
        print("Pose Estimation Input Shape: ", PoseEstimationInputShape)

        # Get Output Shape of Model
        PoseEstimationOutputShape = Network.outputs[PoseEstimationOutputLayer].shape
        print("Pose Estimation Output Shape: ", PoseEstimationOutputShape)

        # Get Shape Values for Face Detection Network
        N, C, H, W = Network.inputs[PoseEstimationInputLayer].shape

        return N, C, H, W, Network

    elif args.m == 'ACTION':

        # TODO: Implement the the ACTION_RECOGNITION_MODEL loading
        print("Error: not yet implemented")
        quit()

        Network = IENetwork(model=ACTION_RECOGNITION_MODEL)

        return Network

    else:
        print("Unknow model '{}'".format(args.m))
        quit()

def get_mask(processed_output):
    '''
    Given an input image size and processed output for a semantic mask,
    returns a masks able to be combined with the original image.
    '''
# START_TEST
    # Create an empty array for other color channels of mask
    empty = np.zeros(processed_output.shape)

    # Stack to make a Green mask where text detected
    mask = np.dstack((empty, processed_output, empty))

    return mask

def infer_on_video(args, N, C, H, W, Network, ExecutableNetwork, cap):

    # Variables to Hold Inference Time Information
    total_ag_inference_time = 0
    inferred_face_count = 0

    # Process frames until the video ends, or process is exited
    while cap.isOpened():

        # Read the next frame
        has_frame, frame = cap.read()
        if not has_frame:
            break

        # Keypress
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        input_image = preprocessing(frame, H, W)

        # Perform inference on the frame
        fdetect_start = time.time()
        results = ExecutableNetwork.infer(inputs={'data': input_image})
        fdetect_end = time.time()
        inf_time = fdetect_end - fdetect_start
        fps = 1. / inf_time

        # Write Information on Image
        text = 'Sneeze-Cough estimation - FPS: {:.1f}, INF: {:.2f}'.format(round(fps, 2), round(inf_time, 4))
        cv2.putText(frame, text, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 125, 255), 1)

        # Add model layer
        process_func = handle_output(args.m)
        points = process_func(results, frame, args)

        # Show computed image
        cv2.imshow('Window', frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

def run_app(args):

    # Load IECore Object
    OpenVinoIE = IECore()
    print("Available Devices: ", OpenVinoIE.available_devices)

    # Load CPU Extensions if Necessary
    if args.d == 'CPU':
        print('Loading CPU extensions....')
        OpenVinoIE.add_extension(CPU_EXTENSION, 'CPU')

    # Load Network
    N, C, H, W, Network = load_network(args)

    # Load Executable Network
    ExecutableNetwork = OpenVinoIE.load_network(network=Network, device_name=args.d)

    # Generate a Named Window
    cv2.namedWindow('Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Window', 800, 600)

    # Get and open video capture
    cap = cv2.VideoCapture()
    cap.open(args.i)
    has_frame, frame = cap.read()

    # Get frame size
    fh = frame.shape[0]
    fw = frame.shape[1]
    print('Original Frame Shape: ', fw, fh)

    infer_on_video(args, N, C, H, W, Network, ExecutableNetwork, cap)

    # Release the capture and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    '''
    Start the OpenVINO Sneeze-Cough detector.
    '''
    args = get_args()

    run_app(args)
