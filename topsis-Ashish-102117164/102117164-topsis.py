import pandas as pd
import numpy as np 
import csv
import logging
import sys
if __name__ == "__main__":
    filename=sys.argv[1]
    weights=list((sys.argv[2]).split(","))
    w=[]
    for i in weights:
        w.append(int(i))
    weights=w
    impact=list((sys.argv[3]).split(","))



    try: 
        data=pd.read_csv(filename)
    except FileNotFoundError:
        logging.warning("File Nhi Mili || File Not Found")
        exit()



    if(len(weights)!=len(impact)!=len(data.columns)-1):
        logging.warning("Number of Impact,number of weights either are not equal or are different from Number of attributes!")
        exit()



    if(len(data.columns)<=2):
        l=len(data.columns)
        logging.warning("Number of columns must be 3 or more, your file has only "+str(l))
        exit()



    all_col=data.columns
    name_col_val=data.iloc[:,0]

    data.drop([data.columns[0]],axis=1,inplace=True)

    catagorical_attributes=[i for i in data.columns if (data[i]).dtype=="object"]

    from sklearn.preprocessing import LabelEncoder
    my_encoder=LabelEncoder()
    enc_data=my_encoder.fit_transform(data[catagorical_attributes])
    data.drop(catagorical_attributes,axis=1)

    enc_data=pd.DataFrame(enc_data)
    enc_data.columns=catagorical_attributes




    final_data=pd.concat([data[[i for i in data.columns if i not in catagorical_attributes]],enc_data],axis=1)




    for i in range(0,len(final_data.columns)):
        temp = 0
        
        for j in range(len(final_data)):
            temp = temp + final_data.iloc[j, i]**2
        temp = temp**0.5
        
        for j in range(len(final_data)):
            final_data.iat[j, i] = (final_data.iloc[j, i] / temp)*weights[i]


    
    best=[]
    worst=[]
    for i in range(0,len(final_data.columns)):

        if(impact[i-1]=="+"):
                best.append(final_data.iloc[:,i].max())
                worst.append(final_data.iloc[:,i].min())
        
        else:
                best.append(final_data.iloc[:,i].min())
                worst.append(final_data.iloc[:,i].max())
            

    best=pd.DataFrame({"best":best}).transpose()
    worst=pd.DataFrame({"worst":worst}).transpose()
    best.columns=final_data.columns=worst.columns
    final_data=pd.concat([final_data,best,worst])

    Positive_distance=[]
    Negative_distance=[]
    for i in range(len(final_data)-2):
        t1=0
        t2=0
        for j in range(len(final_data.columns)):
            t1=t1+(abs(final_data.iloc[i,j]-final_data.iloc[len(final_data)-2,j]))**2
            
            t2=t2+(abs(final_data.iloc[i,j]-final_data.iloc[len(final_data)-1,j]))**2
        Positive_distance.append(t1**0.5)
        Negative_distance.append(t2**0.5)
    pod=pd.DataFrame({"Positive_distance":Positive_distance})
    ned=pd.DataFrame({"Negative_distance":Negative_distance})




    avs=[]
    for i in range(len(pod)):
        avs.append(Positive_distance[i]/(Positive_distance[i]+Negative_distance[i]))

    avg_scr=pd.DataFrame({"Average_Score":avs})

    rank=avg_scr.rank(ascending=False)
    rank.columns=["RANK"]
    final_data=pd.concat([final_data,pod,ned,avg_scr,rank],axis=1)
    print(final_data)
    final_data.to_csv("102117164-TOPSIS_Result.csv")