import cv2

# Function to print default parameters of a tracker
def print_tracker_defaults(tracker_create,tracker_param,i):
    # Attempt to create tracker instance
    try:
        tracker = tracker_create()
        # Print tracker name
        print(f"Default parameters for {tracker.__class__.__name__}:")
        # Print attributes and values
        for attr in filter(lambda x: x.find('__') < 0, dir(tracker_param)):
            print(f"    {attr}: {getattr(tracker_param, attr)}")
        print()
    except Exception as e:
        print(tracker_name[i] + " not supported")
        print(f"Error: {e}")

    

# List of tracker creation functions
tracker_create = [
    cv2.TrackerCSRT.create,
    cv2.TrackerKCF.create,
    cv2.TrackerMIL.create,
    cv2.TrackerDaSiamRPN.create,
    cv2.TrackerGOTURN.create,
    cv2.TrackerNano.create,
]

tracker_param = [
    cv2.TrackerCSRT.Params(),
    cv2.TrackerKCF.Params(),
    cv2.TrackerMIL.Params(),
    cv2.TrackerDaSiamRPN.Params(),
    cv2.TrackerGOTURN.Params(),
    cv2.TrackerNano.Params(),
]
tracker_name =[
    "TrackerCSRT",
    "TrackerKCF",
    "TrackerMIL",
    "TrackerDaSiamRPN",
    "TrackerGOTURN",
    "TrackerNano",
]

params = {
    'admm_iterations': 1,
    'background_ratio': 10,    
}

for i in range(0,len(tracker_param)):
    print_tracker_defaults(tracker_create[i],tracker_param[i],i)
"""
param_handler = cv2.TrackerCSRT.Params()
params = {
    'admm_iterations': 1,
    'background_ratio': 10,    
}
for key, val in params.items():
    setattr(param_handler, key, val)
print_tracker_defaults(tracker_create[0],param_handler)


# List of tracker creation functions
tracker_create = [
    cv2.TrackerCSRT.create,
    cv2.TrackerKCF.create,
    cv2.TrackerDaSiamRPN.create,
    cv2.TrackerGOTURN.create,
    cv2.TrackerMIL.create,
    cv2.TrackerNano.create
]

tracker_param = [
    cv2.TrackerCSRT_Params(),
    cv2.TrackerKCF_Params(),
    cv2.TrackerDaSiamRPN_Params(),
    cv2.TrackerGOTURN_Params(),
    cv2.TrackerMIL_Params(),
    cv2.TrackerNano_Params()
]

"""