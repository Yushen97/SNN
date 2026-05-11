import  os
import  csv
import  math
import  random
import  optuna
import  numpy as np
import  matplotlib.pyplot as plt
from    tqdm import tqdm
from    datetime import datetime
from    optuna.samplers import TPESampler
from    optuna.storages import JournalStorage
from    optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

seed_num=random.randint(0,100000)
random.seed(seed_num)

# Dataset parameter settings
Data_path   = "attributes.csv"
Label_path  = "label.csv"

Total_num   = 8000  # Max 8000                   
Train_num   = 6400  # Max 5600
Epoch_num   = 1000  # Ideal ~ 200
Data_min    = 12    # Minimum sampling range
Data_max    = 79    # Maximum sampling range
Data_Len    = Data_max-Data_min-1   # Data length, original max 200
Label_num   = 8     # Number of label types
Delta_time  = 10    # Dataset time difference, 10ms

# Network hyperparameter settings
Norm_col    = 511       # 0~Norm_col, prefer integer
Norm_factor = 1.2398    # Normalization factor
STDP_A      = 912.69    # LW amplitude parameter
STDP_TC     = 450.88    # LW time constant
Leakage_TC  = 6.76775e-05  # Leakage time constant, T_Dif*Leakage_TC
Vth         = 4.24      # Threshold voltage
LR          = 3.853     # Learning rate
Step        = 2         # Weight delta bounds
Ratio       = 3.2326    # Ratio of designated to non-designated rows
Shift       = 9.8986    # Time offset
Delta_TC    = 1E6       # Current-to-voltage conversion parameter on capacitor

Runtime = datetime.now().strftime("%Y%m%d_%H%M%S")
Parameters = f"{Norm_col} {Norm_factor} {STDP_A} {STDP_TC} {Leakage_TC} {Vth} {LR} {Step} {Ratio} {Shift} {Delta_TC}"

def main():     
    
    print("Algorithm started at", Runtime,  " Random seed:", seed_num)
    print(Parameters)
    
    #Training phase
    Training(Epoch_num)  
    print("Finish!")

def Data_Preparation():
    global dataset_list, dataset, label_list, Train_dataset, Train_label, Test_dataset, Test_label
    dataset_list  = []  # Raw data
    dataset       = []  # Time difference, normalized, randomized
    label_list    = []  # Initial labels
    Train_dataset = []  # Training set
    Train_label   = []  # Training labels
    Test_dataset  = []  # Test set
    Test_label    = []  # Test labels
    # Read dataset
    with open(Data_path, 'r', encoding='utf-8') as dataset_file:
        dataset_reader = csv.reader(dataset_file)
        next(dataset_reader)# Skip header row
        for row in dataset_reader:
            if row:  # Skip empty rows
                numeric_row = [float(x) for x in row[Data_min:Data_max]]
                dataset_list.append(numeric_row)
    # Read labels             
    with open(Label_path, 'r', encoding='utf-8') as label_file:
        label_reader = csv.reader(label_file)
        next(label_reader)  # Skip header row
        for row in label_reader:
            if row:  # Skip empty rows
                numeric_row = int(row[0])
                label_list.append(numeric_row)
    
    # Data conversion data_list --> dataset differential time signal
    dataset = [[sublist[i+1] - sublist[i] for i in range(len(sublist)-1)] for sublist in dataset_list]
    
    # Compute max and min values
    avg_max = sum(max(row) for row in dataset) / len(dataset)
    avg_max_F = avg_max*Norm_factor
    avg_min = sum(min(row) for row in dataset) / len(dataset)
    avg_min_F = avg_min*Norm_factor
    
    # Normalize dataset
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):           
            if(dataset[i][j]>=avg_max_F):
                dataset[i][j]=1
            elif(dataset[i][j]<=avg_min_F):
                dataset[i][j]=-1
            elif(dataset[i][j]>=0 and dataset[i][j]<avg_max_F):
                dataset[i][j]=dataset[i][j]/avg_max_F
            elif(dataset[i][j]<0 and dataset[i][j]>avg_min_F):
                dataset[i][j]=-dataset[i][j]/avg_min_F
            else:
                None
    
    # Map dataset to discrete regions
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            dataset[i][j]= round((dataset[i][j] + 1) / 2 * Norm_col)    
    
    # Random shuffle
    indices = list(range(len(dataset))) # Create index list
    random.shuffle(indices)             # Randomly shuffle indices
    
    # Split
    train_indices = indices[:Train_num]         # First Train_num as training set
    test_indices = indices[Train_num:Total_num] # Train_num to Total_num as test set
    
    # Create training and test sets
    Train_dataset = [dataset[i] for i in train_indices]
    Train_label = [label_list[i] for i in train_indices]
    
    Test_dataset = [dataset[i] for i in test_indices]
    Test_label = [label_list[i] for i in test_indices]
    
def ForPropagation(data, W0B1): # Forward propagation for any data, data is list type            
    global Result_Time
    Result_Time  = np.zeros((Label_num, Data_Len)) # Initialize output
    for t_index in range(Data_Len):   # Iterate time column by column
        for row in range(Label_num):  # Iterate space row by row               
            if t_index==0:
                Pre_value = 0
            else:                               
                # Get previous value and compute leakage result as initial accumulated value
                Pre_value = Leakage(Result_Time[row][t_index-1], Delta_time)   # Leakage depends on previous value and time difference
            if(W0B1==0):
                Result_Time[row][t_index] = Pre_value + TFT_output(Weight[row][data[t_index]]) * Delta_TC
            else:
                Result_Time[row][t_index] = Pre_value + TFT_output(BWeight[row][data[t_index]]) * Delta_TC
           
def Inference():    # Inference on test set
    Cor_Count=0     # Reset count
    Cor_PTCnt=0     # Reset performance counter
    Total_Map  = np.zeros((Label_num, Label_num), dtype=int)
    for sid in range(len(Test_label)):
        ForPropagation(Test_dataset[sid],0)
        Correct, Cor_col, Pre_row = Predict(Test_dataset[sid],Test_label[sid])
        Total_Map[Test_label[sid]][Pre_row]+=1
        Cor_Count += Correct
        Cor_PTCnt += 1.0*Correct*(Cor_col/Data_Len)
        IF_acc=100.0*Cor_Count/(Total_num-Train_num)
        if(Cor_Count==0):
            IF_pt=100
        else:
            IF_pt=100-100.0*Cor_PTCnt/Cor_Count
    # Compute high and medium accuracy from Total_Map
    IF_HighAcc, IF_MediAcc, IF_Precision, IF_Recall, IF_F1 = HM_Out(Total_Map)   
    return IF_acc, IF_pt, IF_HighAcc, IF_MediAcc, IF_Precision, IF_Recall, IF_F1

def Best_Inference(Map_print):
    #print("Best inference result: ")
    Total_Map  = np.zeros((Label_num, Label_num), dtype=int)
    Total_item = [0 for x in range(Label_num)]    # Count of each type
    Total_Cnt  = [0 for x in range(Label_num)]    # Track successful predictions for 0-7
    Total_Pt   = [0 for x in range(Label_num)]    # Track performance for 0-7
    Avg_Acc=0.0
    Avg_Pt=0.0
    for sid in range(Total_num-Train_num):
        ForPropagation(Test_dataset[sid],1)
        Correct, Cor_col, Pre_row = Predict(Test_dataset[sid],Test_label[sid])
        Total_item[Test_label[sid]]+=1
        Total_Cnt[Test_label[sid]]+=Correct
        Total_Pt[Test_label[sid]]+=1.0*Correct*(Cor_col/Data_Len)
        Total_Map[Test_label[sid]][Pre_row]+=1
    # Compute High, Medi, Precision, Recall, F1
    BIF_HighAcc, BIF_MediAcc, BIF_Precision, BIF_Recall, BIF_F1 = HM_Out(Total_Map)
    #print("Accuracy and Performance: ")
    for i in range(Label_num):
        Total_Cnt[i]= 100.0*Total_Cnt[i]/Total_item[i]
        if(Total_Map[i][i]==0):
            Total_Pt[i] = 100
        else:
            Total_Pt[i] = 100-100.0*Total_Pt[i]/Total_Map[i][i]
        Avg_Acc+=Total_Cnt[i]
        Avg_Pt+=Total_Pt[i]
        if(Map_print==1):
            print(i, "  {:.2f}   {:.2f}".format(Total_Cnt[i], Total_Pt[i]))
    if(Map_print==1):        
        print("Total Map: ")
        print(Total_Map, "\n\nVth   Avg_Acc  Avg_Pt HighAcc MediAcc Precision Recall F1")
    # Compute high and medium accuracy from Total_Map
    
    #BWeight_Print()     #print BWeight
    print("{:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}".format(Vth, Avg_Acc/Label_num, Avg_Pt/Label_num, BIF_HighAcc, BIF_MediAcc, BIF_Precision, BIF_Recall, BIF_F1)) 
    
def Training(epoch):
    # Read dataset
    Data_Preparation()

    # Start training phase
    global Weight, BWeight, DWeight, Result_Time, Train_Acc, Test_Acc, Train_PT, Test_PT, Vth
    #Weight = np.zeros((Label_num, (Norm_col+1)), dtype=int)
    Weight = np.random.randint(0, 100, size=(Label_num, (Norm_col+1)))  # Initialize weights
    DWeight= np.zeros((Label_num, (Norm_col+1)), dtype=int)             # Weight delta
    BWeight= np.zeros((Label_num, (Norm_col+1)), dtype=int)             # Save best weights
    Train_Acc = []      # Clear training accuracy list
    Test_Acc = []       # Clear test accuracy list
    Train_PT = []       # Clear training performance list
    Test_PT  = []       # Clear test performance list
    Test_HighAcc = []   # Clear training HighAcc list
    Test_MediAcc = []   # Clear training MediAcc list
    Test_Precision = [] # Clear training Precision list
    Test_Recall = []    # Clear training Recall list
    Test_F1 = []        # Clear training F1 list
    TR_Max_acc=0        # Clear training best
    IF_Max_acc=0        # Clear inference best
   
    for e in tqdm(range(epoch)): 
        DWeight.fill(0) # Clear Delta Weight        
        Cor_Count=0     # Reset count
        Cor_PTCnt=0     # Reset performance counter
        
        # Forward propagation to get Result_Time and weight delta matrix
        for sid in range(Train_num):
            ForPropagation(Train_dataset[sid],0)
            Correct, Cor_col, Pre_row = Predict(Train_dataset[sid],Train_label[sid])
            Cor_Count += Correct
            Cor_PTCnt += 1.0*Correct*(Cor_col/Data_Len)
            
            # Analyze Result_time and update DWeight        
            for row_result in range(Label_num):    # Iterate rows
                # Designated row
                if row_result == Train_label[sid]:
                    # Insert Post based on Cor_col to update all weights, Cor_col can be correct firing position or last non-firing position
                    for col_data in range(Data_Len):
                        DWeight[row_result][Train_dataset[sid][col_data]]+=STDP(((Cor_col-col_data)*Delta_time)+Shift)
                    
                # Non-designated row
                else:
                    # If non-designated row output exceeds threshold and occurs before Cor_col, insert Post before that Pre, apply reverse STDP to all earlier positively increasing Pre
                    for col_data in range(Cor_col):
                        if (Result_Time[row_result][col_data] >= Vth):
                            for pcol in range(col_data,0,-1):
                                if(Result_Time[row_result][pcol-1]<Result_Time[row_result][pcol]):
                                    DWeight[row_result][Train_dataset[sid][pcol]]+=Ratio*STDP(((pcol-col_data)*Delta_time)-Shift)
                                else:
                                    break
                            break 
        
        # Record training set accuracy and performance
        Train_AC=100.0*Cor_Count/Train_num
        Train_Acc.append(Train_AC)
        if(Cor_Count==0):
            Train_PT.append(100)
        else:
            Train_PT.append(100-100.0*Cor_PTCnt/Cor_Count)
        
        # Record test set accuracy and performance
        acc, pt, HighAcc, MediAcc, Precision, Recall, F1=Inference()
        Test_Acc.append(acc)
        Test_PT.append(pt)
        Test_HighAcc.append(HighAcc)
        Test_MediAcc.append(MediAcc)
        Test_Precision.append(Precision)
        Test_Recall.append(Recall)
        Test_F1.append(F1)

        # Record best weight matrix
        if(acc>IF_Max_acc):
            IF_Max_acc=acc
            TR_Max_acc=Train_AC
            BWeight=np.copy(Weight)
               
        # Update weight matrix    
        for row in range(Label_num):
            for col in range(len(DWeight[0])):
                Delta_value = max(-Step, min(DWeight[row][col]/(LR*Train_num), Step))
                Weight[row][col] += Delta_value               
                Weight[row][col] = max(0, min(Weight[row][col], 255))        
        # Print results as needed
        #DWeight_Print()
        #Weight_Print()
        
    #print(TR_Max_acc, IF_Max_acc)
    #return (IF_Max_acc)
    print("\nBest Performance, Vth=", Vth)
    for i in range(210):
        if(i>0):
            Vth=round((Vth-0.02),5)
            Best_Inference(0)
        else:
            Best_Inference(1) 
    Acc_Print(Train_Acc, Test_Acc, Train_PT, Test_PT, Test_HighAcc, Test_MediAcc, Test_Precision, Test_Recall, Test_F1)
    
def Predict(data,label):
    # Determine output based on Predict_Table results, return Correct, Cor_col, Pre_row
    Correct = 0
    Cor_col = 0
    Pre_row = -1    
            
    # Find Cor_col
    for col in range(Data_Len):
        if(Result_Time[label][col]>=Vth):
            Cor_col=col
            break
    else:
        Cor_col=Data_Len
        
    # Check correctness
    Found = 0
    Predict_Max_value=0
    for col in range(Data_Len):
        col_best_value = -float('inf')
        col_best_row = -1       
        for row in range(Label_num):
            if Result_Time[row][col] >= Vth and Result_Time[row][col] > col_best_value:
                col_best_value = Result_Time[row][col]
                col_best_row = row
        if col_best_row != -1:
            Found = 1
            Pre_row = col_best_row
            break
    if not Found:
        for col in range(Data_Len):
            for row in range(Label_num):
                if Result_Time[row][col] > Predict_Max_value:
                    Predict_Max_value = Result_Time[row][col]
                    Pre_row = row   
    if Pre_row == label:
        Correct = 1
  
    # Output results
    return Correct, Cor_col, Pre_row

def TFT_output(Weight_value): 
    # Normalization, Vbg input range and Vtg weight range [1.7-4.25], Bias=3.7
    # TFT_output operates with Pulse each time, initial Vbgs=0.55V, cumulative weight change range [-1.45V, 1.1V]
    #Current = max(TFT_IV[int(Weight_value / (255 / 52))], 0)
    Current = max(TFT_IV_5bit[int(Weight_value/8)], 0)  #5-bit   
    #Current = max(TFT_IV_8bit[int(Weight_value)], 0)
    return Current
    
def STDP(T_Dif):
    if(T_Dif>0):
        return int(STDP_A*math.exp(-T_Dif/STDP_TC))
    if(T_Dif<0):
        return int(-STDP_A*math.exp(T_Dif/STDP_TC))
    return 0
    
def Leakage(Ini, T_Dif):
    return Ini*math.exp(-T_Dif*Leakage_TC)

def STDP_Print():
    x=list(range(-1000,1000))
    y=[]
    for i in range(len(x)):
        y.append(STDP(x[i]))
    plt.plot(x,y)
    plt.title ('Learning window')  # Set title
    plt.xlabel('X Values')  # Set X-axis label
    plt.ylabel('Y Values')  # Set Y-axis label
    plt.grid(True)          # Show grid
    plt.show()              # Display figure
   
def Leakage_Print():
    x=list(range(0,100))
    y=[]
    for i in range(len(x)):
        y.append(Leakage(1, x[i]))
    plt.plot(x,y)
    plt.title ('Leakage curve')  # Set title
    plt.xlabel('X Values')  # Set X-axis label
    plt.ylabel('Y Values')  # Set Y-axis label
    plt.grid(True)          # Show grid
    plt.show()              # Display figure

def Weight_Print():
    print("Weight:")
    for row in Weight:
        print(' '.join(f'{num:3}' for num in row))

def BWeight_Print():
    print("Best Weight:")
    for row in BWeight:
        print(' '.join(f'{num:3},' for num in row))

def DWeight_Print():
    print("Delta Weight:")
    for row in DWeight:
        print(' '.join(f'{num:4}' for num in row))    

def Result_Time_Print(sample_id):
    # Print row by row, column by column
    print("Result_Time: ", sample_id)
    for row in Result_Time:
        print(" ".join(f"{num:.2f}" for num in row))

def HM_Out(list_array):
    TN=0
    TNP=0
    TP=list_array[0][0]
    FN=sum(row[0] for row in list_array[1:Label_num])
    FP=sum(list_array[0][1:Label_num])
    for i in range(1,Label_num):
        TN+=sum(list_array[i][1:Label_num])
    for i in range(1,Label_num):
        TNP+=list_array[i][i]
    High_Acc=(TP+TN)/(TP+FN+FP+TN) if (TP+FN+FP+TN)!=0 else 0   
    Medi_Acc=TNP/TN if TN!=0 else 0
    Precision=TP/(TP+FP) if(TP+FP)!=0 else 0
    Recall=TP/(TP+FN) if(TP+FN)!=0 else 0
    F1=2*(Precision*Recall)/(Precision+Recall) if(Precision+Recall)!=0 else 0
    return High_Acc*100.0, Medi_Acc*100.0, Precision*100.0, Recall*100.0, F1*100.0

def Acc_Print(Train_Acc, Test_Acc,Train_PT,Test_PT,Test_HighAcc, Test_MediAcc, Test_Precision, Test_Recall, Test_F1):
    print("\nTraining result:\nTR_Acc, TR_Acc, TR_PT, IF_PT, IF_HighAcc, IF_MediAcc, IF_Precision, IF_Recall, IF_F1")
    for i in range(Epoch_num):
        print(i, "  {:.2f}   {:.2f}   {:.2f}   {:.2f}  {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}".format(Train_Acc[i], Test_Acc[i], Train_PT[i], Test_PT[i], Test_HighAcc[i], Test_MediAcc[i], Test_Precision[i], Test_Recall[i], Test_F1[i]))   
    print( "\nBIF:  {:.2f}   {:.2f}   {:.2f}   {:.2f}  {:.2f}   {:.2f}   {:.2f}   {:.2f}   {:.2f}".format(Train_Acc[Test_Acc.index(max(Test_Acc))], max(Test_Acc), Train_PT[Test_Acc.index(max(Test_Acc))], Test_PT[Test_Acc.index(max(Test_Acc))], Test_HighAcc[Test_Acc.index(max(Test_Acc))], Test_MediAcc[Test_Acc.index(max(Test_Acc))], Test_Precision[Test_Acc.index(max(Test_Acc))], Test_Recall[Test_Acc.index(max(Test_Acc))], Test_F1[Test_Acc.index(max(Test_Acc))]))
    
    fig, ax1 = plt.subplots()
    
    # Plot Train_list and Test_list (left y-axis)
    ax1.plot(Train_Acc, marker='o', label='Training Acc')
    ax1.plot(Test_Acc, marker='x', label='Testing Acc')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Acc (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Create secondary y-axis
    ax2 = ax1.twinx()
    # Plot Train_PT and Test_PT (right y-axis)
    ax2.plot(Train_PT, marker='s', label='Training PT', color='r')
    ax2.plot(Test_PT, marker='d', label='Testing PT', color='c')
    ax2.set_ylabel('PT', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')
    
    plt.title(Runtime)
    plt.show()

#TFT IV curve Length=52, [-1.45V, 1.1V， 0.05V]
TFT_IV = [1.18E-12, 1.36E-12, -1.93E-12, -1.23E-12, 2.92E-12, 1.55E-12, -7.7E-13, 2.7E-12, 5.26E-12, 8.42E-12, 2.022E-11, 4.448E-11, 8.961E-11, 
1.7357E-10, 3.2406E-10, 5.6876E-10, 9.525E-10, 1.5804E-9, 2.1637E-9, 3.1398E-9, 4.3094E-9, 5.7866E-9, 7.5711E-9, 9.6556E-9, 1.26458E-8, 1.55056E-8, 
1.87308E-8, 2.2312E-8, 2.61792E-8, 3.05622E-8, 3.53196E-8, 4.04844E-8, 4.60788E-8, 5.20634E-8, 5.84844E-8, 6.52716E-8, 7.2527E-8, 8.02744E-8, 8.85018E-8, 
9.70736E-8, 1.06118E-7, 1.1554E-7, 1.2546E-7, 1.3586E-7, 1.4674E-7, 1.5814E-7, 1.7004E-7, 1.8222E-7, 1.95E-7, 2.08E-7, 2.2168E-7, 2.3596E-7]

TFT_IV_5bit = [1.18E-12, -9.17146E-13, -5.907E-14, 1.66921E-12, 1.07216E-12, 5.88441E-12, 1.82208E-11, 6.46684E-11, 1.91832E-10, 5.14601E-10,
1.22993E-9, 2.23204E-9, 3.98649E-9, 6.44833E-9, 9.74386E-9, 1.45712E-8, 1.98547E-8, 2.60475E-8, 3.34353E-8, 4.18887E-8, 5.14648E-8, 6.21581E-8,
7.39866E-8, 8.71508E-8, 1.01396E-7, 1.16789E-7, 1.33471E-7, 1.5145E-7, 1.70816E-7, 1.91259E-7, 2.12756E-7, 2.3596E-7]

TFT_IV_6bit = [1.18E-12, 1.59324E-12, -8.12885E-13, -2.0603E-12, -3.31739E-13, 2.90724E-12, 1.81523E-12, -3.98754E-13, 6.10206E-13, 3.44055E-12, 
5.51538E-12, 8.08365E-12, 1.58361E-11, 3.12185E-11, 5.64958E-11, 9.81623E-11, 1.68669E-10, 2.8198E-10, 4.51783E-10, 6.83692E-10, 1.0652E-9,
1.5804E-9, 2.04422E-9, 2.72343E-9, 3.61669E-9, 4.62831E-9, 5.86486E-9, 7.30554E-9, 8.85065E-9, 1.10547E-8, 1.34626E-8, 1.57903E-8, 1.84059E-8,
2.12627E-8, 2.42884E-8, 2.75814E-8, 3.12212E-8, 3.50845E-8, 3.9215E-8, 4.36321E-8, 4.83097E-8, 5.32549E-8, 5.84844E-8, 6.39483E-8, 6.97035E-8,
7.57828E-8, 8.21985E-8, 8.89021E-8, 9.58224E-8, 1.03058E-7, 1.10553E-7, 1.18319E-7, 1.2643E-7, 1.3485E-7, 1.4358E-7, 1.52637E-7, 1.62071E-7,
1.71758E-7, 1.81625E-7, 1.91931E-7, 2.02391E-7, 2.13112E-7, 2.24354E-7, 2.3596E-7]

TFT_IV_7bit = [1.18E-12, 1.66923E-12, 1.59887E-12, 8.44417E-13, -7.61234E-13, -1.93999E-12, -2.07075E-12, -1.59982E-12, -4.57636E-13, 1.76565E-12,
2.91672E-12, 2.56418E-12, 1.88643E-12, 9.91606E-13, -2.82422E-13, -7.51373E-13, 3.93164E-13, 2.12625E-12, 3.29388E-12, 4.31157E-12, 5.34311E-12,
6.51074E-12, 7.83937E-12, 9.91602E-12, 1.47188E-11, 2.08474E-11, 2.91518E-11, 3.99087E-11, 5.28747E-11, 7.09686E-11, 9.22677E-11, 1.21107E-10,
1.58346E-10, 2.03398E-10, 2.63714E-10, 3.34301E-10, 4.22836E-10, 5.28938E-10, 6.44952E-10, 7.87583E-10, 9.88837E-10, 1.23828E-9, 1.49715E-9, 
1.73665E-9, 1.96191E-9, 2.21284E-9, 2.5701E-9, 3.00195E-9, 3.4433E-9, 3.90822E-9, 4.41221E-9, 4.97578E-9, 5.59552E-9, 6.26598E-9, 6.98751E-9, 
7.73114E-9, 8.48308E-9, 9.3683E-9, 1.04922E-8, 1.17233E-8, 1.29181E-8, 1.4057E-8, 1.52051E-8, 1.64188E-8, 1.77197E-8, 1.90834E-8, 2.05014E-8, 
2.19633E-8, 2.346E-8, 2.50097E-8, 2.66346E-8, 2.836E-8, 3.01655E-8, 3.20233E-8, 3.39339E-8, 3.59069E-8, 3.79491E-8, 4.00618E-8, 4.22457E-8,
4.44975E-8, 4.68107E-8, 4.91833E-8, 5.16246E-8, 5.41426E-8, 5.67289E-8, 5.93708E-8, 6.207E-8, 6.48323E-8, 6.767E-8, 7.05937E-8, 7.35942E-8,
7.66699E-8, 7.98328E-8, 8.3082E-8, 8.63949E-8, 8.97614E-8, 9.31767E-8, 9.66586E-8, 1.00227E-7, 1.03874E-7, 1.07576E-7, 1.11334E-7, 1.15161E-7, 
1.19072E-7, 1.23071E-7, 1.27147E-7, 1.31299E-7, 1.35525E-7, 1.39828E-7, 1.44211E-7, 1.48675E-7, 1.5322E-7, 1.57863E-7, 1.62601E-7, 1.67393E-7, 
1.72218E-7, 1.77074E-7, 1.82023E-7, 1.87103E-7, 1.9226E-7, 1.97437E-7, 2.0264E-7, 2.07896E-7, 2.13286E-7, 2.18824E-7, 2.24444E-7, 2.30153E-7, 2.3596E-7]

TFT_IV_8bit = [1.18E-12, 1.4936E-12, 1.6684E-12, 1.7044E-12, 1.6016E-12, 1.36E-12, 8.60055E-13, 1.01766E-13, -7.3555E-13, -1.47258E-12, -1.93E-12,
-2.09094E-12, -2.07541E-12, -1.9094E-12, -1.61893E-12, -1.23E-12, -5.17821E-13, 5.57142E-13, 1.69102E-12, 2.57993E-12, 2.92E-12, 2.81339E-12,
2.58813E-12, 2.27918E-12, 1.92149E-12, 1.55E-12, 1.05362E-12, 4.08136E-13, -2.21157E-13, -6.68966E-13, -7.7E-13, -4.12205E-13, 2.89242E-13, 1.15879E-12,
2.0209E-12, 2.7E-12, 3.22111E-12, 3.73082E-12, 4.23597E-12, 4.74342E-12, 5.26E-12, 5.80982E-12, 6.40773E-12, 7.04673E-12, 7.71982E-12, 8.42E-12, 9.60815E-12,
1.16028E-11, 1.41833E-11, 1.71293E-11, 2.022E-11, 2.37818E-11, 2.81684E-11, 3.32E-11, 3.86971E-11, 4.448E-11, 5.11869E-11, 5.93667E-11, 6.8706E-11, 7.88916E-11,
8.961E-11, 1.01997E-10, 1.17074E-10, 1.34352E-10, 1.53347E-10, 1.7357E-10, 1.96662E-10, 2.24055E-10, 2.54951E-10, 2.88552E-10, 3.2406E-10, 3.63605E-10,
4.09048E-10, 4.59188E-10, 5.12826E-10, 5.6876E-10, 6.2675E-10, 6.90062E-10, 7.62752E-10, 8.48879E-10, 9.525E-10, 1.07104E-9, 1.19676E-9, 1.32601E-9,
1.45511E-9, 1.5804E-9, 1.69823E-9, 1.81062E-9, 1.92243E-9, 2.0385E-9, 2.1637E-9, 2.31459E-9, 2.49775E-9, 2.70309E-9, 2.92047E-9, 3.1398E-9, 3.35904E-9,
3.58399E-9, 3.81629E-9, 4.05755E-9, 4.3094E-9, 4.5757E-9, 4.85817E-9, 5.1552E-9, 5.46521E-9, 5.7866E-9, 6.12165E-9, 6.47125E-9, 6.83179E-9, 7.19962E-9,
7.5711E-9, 7.93913E-9, 8.31204E-9, 8.70939E-9, 9.15073E-9, 9.6556E-9, 1.02215E-8, 1.08211E-8, 1.14368E-8, 1.20509E-8, 1.26458E-8, 1.32194E-8, 1.37857E-8,
1.43512E-8, 1.49223E-8, 1.55056E-8, 1.611E-8, 1.6738E-8, 1.73863E-8, 1.80516E-8, 1.87308E-8, 1.94233E-8, 2.01296E-8, 2.08477E-8, 2.15758E-8, 2.2312E-8,
2.30566E-8, 2.38131E-8, 2.45839E-8, 2.53718E-8, 2.61792E-8, 2.70117E-8, 2.78702E-8, 2.87508E-8, 2.96495E-8, 3.05622E-8, 3.14865E-8, 3.24232E-8, 3.33735E-8,
3.43385E-8, 3.53196E-8, 3.63177E-8, 3.73331E-8, 3.8366E-8, 3.94164E-8, 4.04844E-8, 4.15702E-8, 4.26733E-8, 4.3793E-8, 4.49284E-8, 4.60788E-8, 4.72433E-8,
4.84227E-8, 4.96183E-8, 5.08314E-8, 5.20634E-8, 5.33149E-8, 5.45844E-8, 5.58703E-8, 5.71708E-8, 5.84844E-8, 5.98112E-8, 6.11528E-8, 6.25098E-8, 6.38825E-8,
6.52716E-8, 6.66805E-8, 6.81118E-8, 6.95641E-8, 7.10362E-8, 7.2527E-8, 7.40356E-8, 7.5563E-8, 7.71109E-8, 7.86808E-8, 8.02744E-8, 8.1889E-8, 8.35202E-8,
8.51669E-8, 8.68278E-8, 8.85018E-8, 9.01871E-8, 9.18843E-8, 9.3596E-8, 9.53249E-8, 9.70736E-8, 9.8845E-8, 1.00638E-7, 1.0245E-7, 1.04278E-7, 1.06118E-7,
1.07971E-7, 1.09838E-7, 1.11721E-7, 1.13621E-7, 1.1554E-7, 1.17481E-7, 1.19444E-7, 1.2143E-7, 1.23436E-7, 1.2546E-7, 1.27502E-7, 1.29564E-7, 1.31644E-7,
1.33743E-7, 1.3586E-7, 1.37996E-7, 1.40152E-7, 1.42327E-7, 1.44523E-7, 1.4674E-7, 1.48976E-7, 1.51231E-7, 1.53508E-7, 1.5581E-7, 1.5814E-7, 1.60494E-7,
1.62863E-7, 1.65245E-7, 1.67638E-7, 1.7004E-7, 1.72446E-7, 1.74858E-7, 1.77285E-7, 1.79736E-7, 1.8222E-7, 1.84739E-7, 1.87284E-7, 1.89848E-7, 1.92422E-7,
1.95E-7, 1.97579E-7, 2.00166E-7, 2.02763E-7, 2.05373E-7, 2.08E-7, 2.10663E-7, 2.13372E-7, 2.16117E-7, 2.1889E-7, 2.2168E-7, 2.24488E-7, 2.2732E-7, 2.30176E-7,
2.33056E-7, 2.3596E-7]

if __name__ == "__main__":
    main()
