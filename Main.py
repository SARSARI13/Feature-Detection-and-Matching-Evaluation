import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2 as cv
import pykitti
import time
from prettytable import PrettyTable
from tqdm import tqdm

# Define the base path, date, and drive for the KITTI dataset
base = 'C:/Users/Yassine/Desktop/git/TP_2_3_4/KITTI_SAMPLE/RAW'
date = '2011_09_26'
drive = '0009'
dataset = pykitti.raw(base, date, drive, frames=range(0, 50, 1))

# Methods that act as both detectors and descriptors
# SIFT
def SIFT(img1, img2, approach):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(np.array(img1), None)
    kp2, des2 = sift.detectAndCompute(np.array(img2), None)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# KAZE
def KAZE(img1, img2, approach):
    kaze = cv.KAZE_create()
    kp1, des1 = kaze.detectAndCompute(img1, None)
    kp2, des2 = kaze.detectAndCompute(img2, None)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# ORB (fusion of FAST keypoint detector and BRIEF descriptor)
def ORB(img1, img2, approach):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(np.array(img1), None)
    kp2, des2 = orb.detectAndCompute(np.array(img2), None)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# BRISK
def BRISK(img1, img2, approach):
    brisk = cv.BRISK_create()
    kp1, des1 = brisk.detectAndCompute(np.array(img1), None)
    kp2, des2 = brisk.detectAndCompute(np.array(img2), None)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# AKAZE
def AKAZE(img1, img2, approach):
    akaze = cv.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# FAST with FREAK
def FAST_FREAK(img1, img2, approach):
    fast = cv.FastFeatureDetector_create()
    freak = cv.xfeatures2d.FREAK_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = freak.compute(img1, kp1)
    kp2, des2 = freak.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# FAST with BRIEF
def FAST_BRIEF(img1, img2, approach):
    fast = cv.FastFeatureDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# STAR with BRIEF
def STAR_BRIEF(img1, img2, approach):
    star = cv.xfeatures2d.StarDetector_create()
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    kp1 = star.detect(img1, None)
    kp2 = star.detect(img2, None)
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# STAR with FREAK
def STAR_FREAK(img1, img2, approach):
    star = cv.xfeatures2d.StarDetector_create()
    freak = cv.xfeatures2d.FREAK_create()
    kp1 = star.detect(img1, None)
    kp2 = star.detect(img2, None)
    kp1, des1 = freak.compute(img1, kp1)
    kp2, des2 = freak.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# STAR with LATCH
def STAR_LATCH(img1, img2, approach):
    star = cv.xfeatures2d.StarDetector_create()
    latch = cv.xfeatures2d.LATCH_create()
    kp1 = star.detect(img1, None)
    kp2 = star.detect(img2, None)
    kp1, des1 = latch.compute(img1, kp1)
    kp2, des2 = latch.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# STAR with LUCID
def STAR_LUCID(img1, img2, approach):
    star = cv.xfeatures2d.StarDetector_create()
    lucid = cv.xfeatures2d.LUCID_create()
    kp1 = star.detect(img1, None)
    kp2 = star.detect(img2, None)
    kp1, des1 = lucid.compute(img1, kp1)
    kp2, des2 = lucid.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# STAR with DAISY
def STAR_DAISY(img1, img2, approach):
    star = cv.xfeatures2d.StarDetector_create()
    daisy = cv.xfeatures2d.DAISY_create()
    kp1 = star.detect(img1, None)
    kp2 = star.detect(img2, None)
    kp1, des1 = daisy.compute(img1, kp1)
    kp2, des2 = daisy.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# FAST with DAISY
def FAST_DAISY(img1, img2, approach):
    fast = cv.FastFeatureDetector_create()
    daisy = cv.xfeatures2d.DAISY_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = daisy.compute(img1, kp1)
    kp2, des2 = daisy.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# FAST with LATCH
def FAST_LATCH(img1, img2, approach):
    fast = cv.FastFeatureDetector_create()
    latch = cv.xfeatures2d.LATCH_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = latch.compute(img1, kp1)
    kp2, des2 = latch.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# FAST with LUCID
def FAST_LUCID(img1, img2, approach):
    fast = cv.FastFeatureDetector_create()
    lucid = cv.xfeatures2d.LUCID_create()
    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    kp1, des1 = lucid.compute(img1, kp1)
    kp2, des2 = lucid.compute(img2, kp2)
    return match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach)

# Function to match keypoints and extract points
def match_keypoints_and_extract_points(kp1, des1, kp2, des2, approach):
    bf = cv.BFMatcher(approach, crossCheck=True)
    matches = bf.match(des1, des2)
    query_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    train_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    return np.array([query_pts, train_pts])

# Function to generate different scenarios
def generate_scenarios(I, scenario_type):
    I = np.array(I)
    I_F = []
    if scenario_type == 1:
        params = [(-30, 50, 20)]
    elif scenario_type == 2:
        params = [(1.3, 2.3, 0.2)]
    elif scenario_type == 3:
        params = [(10, 100, 10)]
    else:
        return []

    for p in params:
        for param in np.arange(*p):
            if scenario_type == 1:
                Im = np.array(np.int16(I)) + param
                Im[Im < 0] = 0
                Im[Im > 255] = 255
                Im = np.array(np.uint8(Im))
                I_F.append([I, cv.cvtColor(Im, cv.COLOR_RGB2BGR)])
            elif scenario_type == 2:
                M = cv.getRotationMatrix2D((I.shape[1]//2, I.shape[0]//2), 0, param)
                img = cv.warpAffine(I, M, (I.shape[1], I.shape[0]))
                I_F.append([I, img, M])
            elif scenario_type == 3:
                M = cv.getRotationMatrix2D((I.shape[1] // 2, I.shape[0] // 2), param, 1)
                angle = np.radians(param)
                a = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                img = cv.warpAffine(I, M, (I.shape[1], I.shape[0]))
                I_F.append([I, img, M, a])
    return np.array(I_F)

# Function to get camera images
def get_camera(Ncamera, n):
    if 0 <= Ncamera <= 3 and 0 <= n <= 49:
        return getattr(dataset, f"get_cam{Ncamera}")(n)
    else:
        print("ERROR")
        return None

# Evaluation function 1
def evaluation_1(X, Y):
    diff = np.abs(X - Y)
    good_match = np.sum((diff[:, 0] < 3) & (diff[:, 1] < 3))
    return 100 * good_match / X.shape[0]

# Evaluation function 2
def evaluation_2(X, Y, M):
    X_transformed = np.dot(X, M[:2, :2]) + M[:2, 2]
    diff = np.abs(X_transformed - Y)
    good_match = np.sum((diff[:, 0] < 3) & (diff[:, 1] < 3))
    return 100 * good_match / X.shape[0]

# Evaluation function 3
def evaluation_3(X, Y, M, angle):
    X_transformed = np.dot(X, angle) + M[:2, 2]
    diff = np.abs(X_transformed - Y)
    good_match = np.sum((diff[:, 0] < 3) & (diff[:, 1] < 3))
    return 100 * good_match / X.shape[0]

# Detection function for scenarios 1, 2, and 3
def detection_1_2_3(I, Methodes, approach, scenarios_n):
    I = np.array(I)
    M = []
    Images = generate_scenarios(I, scenarios_n)
    for img_pair in Images:
        S = Methodes(I, img_pair[1], approach)
        if scenarios_n == 1:
            M.append(evaluation_1(S[0], S[1]))
        elif scenarios_n == 2:
            M.append(evaluation_2(S[0], S[1], img_pair[2]))
        elif scenarios_n == 3:
            M.append(evaluation_3(S[0], S[1], img_pair[2], img_pair[3]))
    return M

# Display function for scenarios 1, 2, and 3
def display_1_2_3(Method, x, approach, variable, norm):
    cst = ["SIFT", "ORB", "BRISK", "AKAZE", "KAZE", "FAST_FREAK", "FAST_BRIEF", "FAST_LATCH",
           "FAST_DAISY", "STAR_BRIEF", "STAR_FREAK", "STAR_LATCH", "STAR_DAISY"]
    plt.figure(figsize=(12, 20))
    plt.xlabel("{}".format(variable))
    plt.ylabel("Percentage of correct points")
    for i in range(len(Method)):
        if i < 5:
            plt.plot(x, Method[i], label=cst[i])
        elif 5 <= i < 10:
            plt.plot(x, Method[i], label=cst[i], linestyle='--', marker='o')
        else:
            plt.plot(x, Method[i], label=cst[i], linestyle=':', marker='<')
    plt.title('Evaluation by {}'.format(norm))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    return 0

# Function to display results for scenarios 1, 2, and 3
def display_results_scenarios_1_2_3(n, idx, detection, approach, norm, scenarios_n):
    if n == 2 and 0 <= idx <= 49:
        I = dataset.get_cam2(idx)
    elif n == 3 and 0 <= idx <= 49:
        I = dataset.get_cam3(idx)
    else:
        return "ERROR"
    Method = [SIFT, ORB, BRISK, KAZE, AKAZE, FAST_BRIEF, FAST_FREAK, FAST_LATCH, FAST_DAISY,
              STAR_BRIEF, STAR_FREAK, STAR_LATCH, STAR_DAISY]
    M = []
    for i in tqdm(range(len(Method))):
        M.append(detection_1_2_3(I, Method[i], approach, scenarios_n))
    if scenarios_n == 1:
        intensity = np.arange(-30, 50, 20)
        display_1_2_3(M, intensity, approach, "intensity", norm)
    elif scenarios_n == 2:
        S = np.arange(1.3, 2.3, 0.2)
        display_1_2_3(M, S, approach, "Scale", norm)
    elif scenarios_n == 3:
        R = np.arange(10, 100, 10)
        display_1_2_3(M, R, approach, "Rotation", norm)
    return 0




display_results_scenarios_1_2_3(2,0,detection_1_2_3,cv.NORM_L2,"NORM_L2",1)  # Display results scenarios 1
display_results_scenarios_1_2_3(2,0,detection_1_2_3,cv.NORM_L2,"NORM_L2",2)  # Display results scenarios 2
display_results_scenarios_1_2_3(2,0,detection_1_2_3,cv.NORM_L2,"NORM_L2",3)  # Display results scenarios 3





# Definition of scenario 4
def scenario_4(idx):
    if 0 <= idx <= 49:
        current_img = np.array(dataset.get_cam2(idx))  
        next_img = np.array(dataset.get_cam3(idx))     
    else:
        print("ERROR")  
    return current_img, next_img  

# Definition of scenario 5
def scenario_5(Ncamera, idx):
    if (Ncamera == 2 or Ncamera == 3) and 0 <= idx <= 48:
        if Ncamera == 2:
            current_img = np.array(dataset.get_cam2(idx))         
            next_img = np.array(dataset.get_cam2(idx + 1))        
        else:
            current_img = np.array(dataset.get_cam3(idx))         
            next_img = np.array(dataset.get_cam3(idx + 1))        
    else:
        print("ERROR")  
    return current_img, next_img  

# Definition of scenario 6
def scenario_6(Ncamera, idx):
    if (Ncamera == 2 or Ncamera == 3) and 0 <= idx <= 48:
        if Ncamera == 2:
            current_img = np.array(dataset.get_cam2(idx))         
            next_img = np.array(dataset.get_cam3(idx + 1))        
        else:
            current_img = np.array(dataset.get_cam3(idx))         
            next_img = np.arraydataset.get_cam2(idx + 1)          
    else:
        print("ERROR")  
    return np.array([current_img, next_img])  

# Definition of scenario 7
def scenario_7(Ncamera, idx):
    if (Ncamera == 2 or Ncamera == 3) and 0 <= idx <= 47:
        if Ncamera == 2:
            current_img = np.array(dataset.get_cam2(idx))         
            next_img = np.array(dataset.get_cam3(idx + 2))        
        else:
            current_img = np.array(dataset.get_cam3(idx))        
            next_img = np.array(dataset.get_cam2(idx + 2))        
    else:
        print("ERROR")  
    return current_img, next_img  

# Function for evaluating scenarios 4, 5, 6, and 7
def Evaluation_4_5_6_7(X, Y):
    try:
        F, mask = cv.findFundamentalMat(X, Y, method=cv.FM_RANSAC + cv.FM_8POINT)
        if F is None or F.shape == (1, 1):
            
            raise Exception('No fundamental matrix found')
        c = np.count_nonzero(mask)  
    except:
        return 0
    return c, c * 100 / len(mask)  

# Function for detecting features and matching in scenarios 4, 5, 6, and 7
def detection_4_5_6_7(scenario, Ncamera, Methods, approach):
    s = 0
    m = 0
    t = 0
    if scenario == scenario_4:
        t1 = time.perf_counter()  
        for idx in range(50):
            I = scenario(idx)  
            t1 = time.perf_counter()  
            S = Methods(I[0], I[1], approach)  
            t2 = time.perf_counter()  
            t += t2 - t1  
            E = Evaluation_4_5_6_7(S[0], S[1])  
            s += E[0]  
            m += E[1]  
        return s / 50, m / 50, t / 50 
    elif scenario == scenario_5 or scenario == scenario_6:
        t1 = time.perf_counter() 
        for idx in range(49):
            I = scenario(Ncamera, idx)  
            t1 = time.perf_counter()  
            S = Methods(I[0], I[1], approach)  
            t2 = time.perf_counter()  
            t += t2 - t1  
            E = Evaluation_4_5_6_7(S[0], S[1])  
            s += E[0]  
            m += E[1]  
        return s / 49, m / 49, t / 49  
    elif scenario == scenario_7:
        t1 = time.perf_counter()  
        for idx in range(48):
            I = scenario(Ncamera, idx)  
            t1 = time.perf_counter()  
            S = Methods(I[0], I[1], approach)  
            t2 = time.perf_counter()  
            t += t2 - t1  
            E = Evaluation_4_5_6_7(S[0], S[1])  
            s += E[0]  
            m += E[1]  
        return s / 48, m / 48, t / 48  
    else:
        return "ERROR"  

# Function for displaying results of scenarios 4, 5, 6, and 7
# Function for displaying results of scenarios 4, 5, 6, and 7
def Display_4_5_6_7(times, Np, T_ap, approach, scenario, norm):
    Method = ["SIFT", "ORB", "BRISK", "AKAZE", "KAZE", "FAST_FREAK", "FAST_BRIEF", "FAST_LATCH",
               "FAST_DAISY", "STAR_BRIEF", "STAR_FREAK", "STAR_LATCH", "STAR_DAISY"]
    x = PrettyTable()  # Create a table for displaying results
    x.field_names = ["Method", "Execution time (s)", "Number of points per image", "Matching rate (%)"]

    if scenario == scenario_4:
        x.title = "Scenario 4 matching using " + norm
    elif scenario == scenario_5:
        x.title = "Scenario 5 matching using " + norm
    elif scenario == scenario_6:
        x.title = "Scenario 6 matching using " + norm
    else:
        x.title = "Scenario 7 matching using " + norm

    # Format and add rows to the table
    for i in range(len(Method)):
        time_formatted = "{:.2f}".format(times[i])  # Format execution time
        Np_formatted = str(int(Np[i]))  # Convert number of points to integer
        T_ap_formatted = "{:.2f}".format(T_ap[i])  # Format matching rate
        x.add_row([Method[i], time_formatted, Np_formatted, T_ap_formatted])

    print(x)  # Print the table
    return 0

# Function for obtaining results of scenarios 4, 5, 6, and 7
def Result_scenarios_4_5_6_7(Ncamera, scenario, approach, Norm):
    if Ncamera == 2 or Ncamera == 3:
        Methods = [SIFT, ORB, BRISK, KAZE, AKAZE, FAST_BRIEF, FAST_FREAK, FAST_LATCH, FAST_DAISY,
                   STAR_BRIEF, STAR_FREAK, STAR_LATCH, STAR_DAISY]
        Nb_pt = []
        T_ap = []
        times = []
        for i in tqdm(range(len(Methods))):
            x = detection_4_5_6_7(scenario, Ncamera, Methods[i], approach)
            Nb_pt.append(x[0])
            T_ap.append(x[1])
            times.append(x[2])
        Display_4_5_6_7(times, Nb_pt, T_ap, approach, scenario, Norm)
    else:
        return "ERROR"
    return 0




Result_scenarios_4_5_6_7(2,scenario_4,cv.NORM_L2,"NORM_L2")  # Display results scenarios 4
Result_scenarios_4_5_6_7(2,scenario_5,cv.NORM_L2,"NORM_L2")  # Display results scenarios 5
Result_scenarios_4_5_6_7(2,scenario_6,cv.NORM_L2,"NORM_L2")  # Display results scenarios 6
Result_scenarios_4_5_6_7(2,scenario_7,cv.NORM_L2,"NORM_L2")  # Display results scenarios 7
