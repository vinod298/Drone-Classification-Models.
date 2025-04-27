import pandas as pd
import tkinter as tk
from tkinter import*
from tkinter import filedialog

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import joblib
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
import pickle

import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
main = tk.Tk()
main.title(" MACHINE LEARNING ANALYSIS OF DRONE CLASSIFICATION MODELS: INSPIRE, MAVIC, PHANTOM, AND NO DRONE")
main.geometry("1600x1300")
title = tk.Label(main, text="MACHINE LEARNING ANALYSIS OF DRONE CLASSIFICATION MODELS: INSPIRE, MAVIC, PHANTOM, AND NO DRONE",justify='center')

model_folder = 'model'
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=r"Dataset"
model_folder = 'model'

#create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
flat_data_file = os.path.join(datadir, 'flat_data.npy')
target_file = os.path.join(datadir, 'target.npy')

active_model = None  # Will store 'svm' or 'rfc' to track which model is active
svm_model = None
rf_model = None

# Store metrics for comparison
svm_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'fscore': 0}
rf_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'fscore': 0}

def upload():
    global filename
    global dataset,categories
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    path = r"Dataset"
    model_folder = "model"
    categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    categories
    text.insert(END,"Total Categories Found In Dataset"+str(categories)+'\n\n')

Categories=['dji_inspire', 'dji_mavic', 'dji_phantom', 'no_drone']
flat_data_arr=[] #input array
target_arr=[] #output array
datadir=r"Dataset"
model_folder = 'model'

def imageprocessing():
    global flat_data,target,categories
    
    flat_data_arr=[] #input array
    target_arr=[] #output array
    datadir=r"Dataset"
    #create file paths by combining the datadir (data directory) with the filenames 'flat_data.npy
    flat_data_file = os.path.join(datadir, 'flat_data.npy')
    target_file = os.path.join(datadir, 'target.npy')

    if os.path.exists(flat_data_file) and os.path.exists(target_file):
        # Load the existing arrays
        flat_data = np.load(flat_data_file)
        target = np.load(target_file)
        text.insert(END,"Total Images Found In Dataset : "+str(flat_data.shape[0])+'\n\n')
        
    else:
        #path which contains all the categories of images
        for i in Categories:
        
            print(f'loading... category : {i}')
            path=os.path.join(datadir,i)
            #create file paths by combining the datadir (data directory) with the i
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))#Reads the image using imread.
                img_resized=resize(img_array,(150,150,3)) #Resizes the image to a common size of (150, 150, 3) pixels.
                flat_data_arr.append(img_resized.flatten()) #Flattens the resized image array and adds it to the flat_data_arr.
                target_arr.append(Categories.index(i)) #Adds the index of the category to the target_arr.
                    #this index is being used to associate the numerical representation of the category (index) with the actual image data. This is often done to provide labels for machine learning algorithms where classes are represented numerically. In this case, 'ORGANIC' might correspond to label 0, and 'NONORGANIC' might correspond to label 1.
                print(f'loaded category:{i} successfully')
                #After processing all images, it converts the lists to NumPy arrays (flat_data and target).
                flat_data=np.array(flat_data_arr)
                target=np.array(target_arr)
        # Save the arrays(flat_data ,target ) into the files(flat_data.npy,target.npy)
        np.save(os.path.join(datadir, 'flat_data.npy'), flat_data)
        np.save(os.path.join(datadir, 'target.npy'), target)
        
        
def splitting():
    global x_train,x_test,y_train,y_test
    
    df=pd.DataFrame(flat_data)
    df['Target']=target #associated the numerical representation of the category (index) with the actual image data
    
    x_train,x_test,y_train,y_test=train_test_split(flat_data,target,test_size=0.20,random_state=77)
    text.insert(END,"Total Images Used For Training : "+str(x_train.shape[0])+'\n\n')
    text.insert(END,"Total Images Used For Testing : "+str(x_test.shape[0])+'\n\n')


labels=Categories
precision = []
recall = []
fscore = []
accuracy = []

#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(algorithm, predict, testY):
    global svm_metrics, rf_metrics
    
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict,average='macro') * 100
    r = recall_score(testY, predict,average='macro') * 100
    f = f1_score(testY, predict,average='macro') * 100
    a = accuracy_score(testY,predict)*100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    # Store metrics for comparison
    if algorithm == "Support Vector Machine Classifier":
        svm_metrics = {'accuracy': a, 'precision': p, 'recall': r, 'fscore': f}
    elif algorithm == "RandomForestClassifier":
        rf_metrics = {'accuracy': a, 'precision': p, 'recall': r, 'fscore': f}
    
    text.insert(END,algorithm+' Accuracy    : '+str(a)+'\n')
    text.insert(END,algorithm+' Precision   : '+str(p)+'\n')
    text.insert(END,algorithm+' Recall      : '+str(r)+'\n')
    text.insert(END,algorithm+' FSCORE      : '+str(f)+'\n')
    report=classification_report(predict, testY,target_names=labels)
    conf_matrix = confusion_matrix(testY, predict)
    text.insert(END,algorithm+" Accuracy : "+str(a)+'\n\n')
    text.insert(END,algorithm+"Classification Report: "+'\n'+str(report)+'\n\n')
    plt.figure(figsize =(5, 5)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="Blues" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()


def SVM():
    global active_model, svm_model
    active_model = 'svm'
    text.delete('1.0', END)
    
    try:
        if os.path.exists('SVM_model.pkl'):
            # Load the trained model from the file
            svm_model = joblib.load('SVM_model.pkl')
            text.insert(END, "SVM model loaded successfully.\n\n")
        else:
            # Train the model
            svm_model = SVC(C=2,
                kernel='rbf',
                degree=3,
                gamma='scale',
                coef0=0.0,
                shrinking=False,
                probability=False,
                tol=0.001,
                cache_size=200,
                class_weight=None,
                verbose=False,)
            svm_model.fit(x_train, y_train)
            # Save the trained model to a file
            joblib.dump(svm_model, 'SVM_model.pkl')
            text.insert(END, "SVM model trained and saved successfully.\n\n")
        
        predict = svm_model.predict(x_test)
        calculateMetrics("Support Vector Machine Classifier", predict, y_test)
    except Exception as e:
        text.insert(END, f"Error in SVM: {str(e)}\nPlease ensure data is loaded and processed first.\n\n")
    
def RFC():
    global active_model, rf_model
    active_model = 'rfc'
    text.delete('1.0', END)
    
    try:
        # Due to potential version compatibility issues, we'll retrain the model
        text.insert(END, "Training Random Forest model...\n\n")
        
        # Create and train a new Random Forest model with optimized parameters for speed
        rf_model = RandomForestClassifier(
            n_estimators=50,        # Fewer trees for faster training
            max_depth=15,           # Limit tree depth
            min_samples_split=5,    # Require more samples to split
            max_features='sqrt',    # Use sqrt of features for faster splits
            n_jobs=-1               # Use all available CPU cores
        )
        rf_model.fit(x_train, y_train)
        
        # Save the model weights with the current scikit-learn version
        Model_file = os.path.join(model_folder, "RF_Classifier.pkl")
        
        # Make sure the model directory exists
        os.makedirs(model_folder, exist_ok=True)
        
        # Save the model with the current version format
        joblib.dump(rf_model, Model_file)
        text.insert(END, "Random Forest model trained and saved successfully.\n\n")
            
        # Evaluate the model
        predict = rf_model.predict(x_test)
        calculateMetrics("RandomForestClassifier", predict, y_test)
    except Exception as e:
        text.insert(END, f"Error in Random Forest: {str(e)}\nPlease ensure data is loaded and processed first.\n\n")

def compare_models():
    if svm_metrics['accuracy'] == 0 or rf_metrics['accuracy'] == 0:
        text.delete('1.0', END)
        text.insert(END, "Please train both SVM and RFC models first before comparing.\n\n")
        return
    
    text.delete('1.0', END)
    text.insert(END, "Comparing SVM and RFC Models\n\n")
    
    # Create comparison table
    comparison_text = "Performance Metrics Comparison:\n"
    comparison_text += "----------------------------------\n"
    comparison_text += "Metric      | SVM       | RFC       | Difference (RFC-SVM)\n"
    comparison_text += "----------------------------------\n"
    
    # Add each metric with difference
    for metric in ['accuracy', 'precision', 'recall', 'fscore']:
        svm_val = svm_metrics[metric]
        rf_val = rf_metrics[metric]
        diff = rf_val - svm_val
        comparison_text += f"{metric.capitalize():<12} | {svm_val:.2f}% | {rf_val:.2f}% | {diff:+.2f}%\n"
    
    text.insert(END, comparison_text + "\n")
    
    # Create bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F-Score']
    svm_values = [svm_metrics['accuracy'], svm_metrics['precision'], svm_metrics['recall'], svm_metrics['fscore']]
    rf_values = [rf_metrics['accuracy'], rf_metrics['precision'], rf_metrics['recall'], rf_metrics['fscore']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.axes()
    rects1 = ax.bar(x - width/2, svm_values, width, label='SVM')
    rects2 = ax.bar(x + width/2, rf_values, width, label='RFC')
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison: SVM vs RFC')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.show()

def prediction():
    global rf_model, Model_file, active_model, svm_model
    
    if active_model is None:
        text.insert(END, "Please train either SVM or RFC model first before making predictions.\n\n")
        return
        
    path = filedialog.askopenfilename(
        initialdir = "testing",
        title = "Select an image file",
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
            ("All files", "*.*")
        )
    )
    
    # Check if a file was actually selected
    if not path:
        text.insert(END, "No file selected. Please select an image file.\n\n")
        return
        
    # Check if the path is a directory
    if os.path.isdir(path):
        text.insert(END, "You selected a directory. Please select an image file instead.\n\n")
        return
        
    try:
        img = imread(path)
        img_resize = resize(img, (150, 150, 3))
        img_preprocessed = [img_resize.flatten()]
        
        if active_model == 'svm':
            if svm_model is None:
                try:
                    svm_model = joblib.load('SVM_model.pkl')
                except:
                    text.insert(END, "SVM model not found. Please train SVM model first.\n\n")
                    return
            text.insert(END, "Using SVM model for prediction...\n\n")
            model = svm_model
        else:  # active_model == 'rfc'
            if rf_model is None:
                try:
                    Model_file = os.path.join(model_folder, "RF_Classifier.pkl")
                    rf_model = joblib.load(Model_file)
                except:
                    text.insert(END, "Random Forest model not found. Please train RFC model first.\n\n")
                    return
            text.insert(END, "Using Random Forest model for prediction...\n\n")
            model = rf_model
        
        output_number = model.predict(img_preprocessed)[0]
        output_name = categories[output_number]

        plt.figure(figsize=(5, 5))  # Reduced from 10x10 to 5x5
        plt.imshow(img)
        plt.text(10, 10, f'Predicted Output: {output_name}', color='white', fontsize=12, weight='bold', backgroundcolor='black')
        plt.axis('off')
        plt.show()
        
        text.insert(END, f"Prediction completed successfully using {active_model.upper()} model.\n")
        text.insert(END, f"Predicted class: {output_name}\n\n")
    except Exception as e:
        text.insert(END, f"Error processing image: {str(e)}\nPlease select a valid image file.\n\n")
    
   
title.grid(column=0, row=0)
font=('times', 13, 'bold')
title.config(bg='purple', fg='white')
title.config(font=font)
title.config(height=3,width=120)
title.place(x=60,y=5)

uploadButton = Button(main, text="Upload Dataset   ",command=upload)
uploadButton.config(bg='Skyblue', fg='Black')
uploadButton.place(x=50,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Image Processing ",command=imageprocessing)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=250,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Splitting   ",command=splitting)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=450,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="SVM_classifier",command=SVM)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=600,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="RFC  Classifier ",command=RFC)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=770,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Compare Models",command=compare_models)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=950,y=100)
uploadButton.config(font=font)

uploadButton = Button(main, text="Prediction   ",command=prediction)
uploadButton.config(bg='skyblue', fg='Black')
uploadButton.place(x=1150,y=100)
uploadButton.config(font=font)

font1 = ('times', 12, 'bold')
text=Text(main,height=28,width=180)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=15,y=250)
text.config(font=font1)
main.mainloop()
