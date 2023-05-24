from os import listdir
import sys, os, numpy as np
import random
from os.path import isfile, join, isdir
import shutil

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, concatenate
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, BatchNormalization, SpatialDropout1D, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

def getAllFilesRecursive(root):
    files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
    dirs = [ d for d in listdir(root) if isdir(join(root,d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root,d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root,f))
    return files


modelname='quality'
UNSORTED_PATH= ########fill this in
QUALITY_PATH = #############fill this in
MAXLEN=300
batch_size = 100
epochs=100
mode='predict'
#mode='train'
disallowList=[b'\xff\xd8', b'\x4d\x5a', b'\xca\xfe', b'\x89\x50', b'\x50\x4b', b'\x00\x00', b'\x52\x49', b'\x47\x49', b'\x7f\x45', b'\x3c\x48', b'\xd0\xcf', b'\x3c\x68', b'\x3c\x21', b'\x4d\x53', b'\x52\x61', b'\x42\x4d']
#              JPEG,        EXE,         Java,        PNG,         Zip,         misc bins,   RIFF,        GIF,         ELF,         HTML,        OLE2,        HTML,        HTML,        CAB,        Rar,          BMP
disallowList2= [b'SIMPLE  = ', b'auto doc=[struct', b'/* XPM *', b'auto doc\x0d\x0a=\x0d\x0a[struct', b'auto saved_art_cache=[struct', b'id=ImageMagick', b'id=MagickCache',
                b'LBLSIZE=', #image formats that look like text
                b'~^\x0d\x0a#ERROR messages', #source code marked up with compiler errors. better to use original.
                b'From: ', b'\x0d\x0a\xc4 Area: ', b'Date: ', b'Received: ', b'Produced by Qmail', b'Produced By O_QWKer', 'To: ', #mail messages. should at least be preprocessed.
                b'.--------------------------------------------------------------------.', b'Session Start: ', b'\x0d\x0aSession Start: ', #instant messages/IRC. should at least be preprocessed.
                b'************************************************************\x0d\x0aMicrosoft Setup Log File Opened', b'\x0d\x0a Volume in drive', b'***  Installation Started',
                b'\xfe   KAV for ', b'FindVirus version', b'Virus scanning report', b'TechFacts 95 System Watch Report', b'Microsoft Office Find Fast Indexer',
                b'*********************************************************************************\x0d\x0a*\x0d\x0a* Log opened:', b'***********  Start log **************'
                b'**********************************************************************\x0d\x0a                            Scan Results',
                b'                               System Information', b'Norman Sandbox Information',
                b'                              File Fix dBASE Repair', b'Microsoft Anti-Virus.', b'Ghost CRC32 Verification list file', #various logs
                b'///////////////////////////////////////////////////////////////////////////////\x0d\x0a// All Platform Dat Update script.', #mcafee ini file
                b'    Offset  String', b'  Offset  String' #output of strings utility
               ]


disallowList3= ['"$CHICAGO$"', '"$Windows NT$"', ' Msg#: ', '$$   Processing started on ', '\x00\x00']
#               Windows inf files,               mail,      logs (Integrity Master),       generic binaries
disallowList3 = [s.lower() for s in disallowList4]


if mode=='predict':
    modelqual = tf.keras.models.load_model('qual.h5')
    file_list=getAllFilesRecursive(UNSORTED_PATH)
    print('enumeration done')
    for file_path in file_list:
        filesize=os.path.getsize(file_path)
        if filesize<=2048 or filesize>=2048*1024:
            continue
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        two_bytes = raw_data[:2]
        if two_bytes in disallowList:
            print('skipping for 2-byte header')
            continue
            
        bytesfound=False
        for byte_identifier in disallowList2:
            if raw_data[:len(byte_identifier)] == byte_identifier:
                bytesfound=True
        if bytesfound:
            print('skipping for larger header')
            continue

        np_data=np.expand_dims(np.array(list(raw_data[:MAXLEN])), axis=0)
        result=modelqual.predict(np_data, batch_size=1, steps=1)[0]
        if result<.5:
            print('skipping for uninteresting')
            continue

        string_data=raw_data.decode('latin1').lower()
        stringfound=False 
        for string_identifier in disallowList4:
            if string_identifier in string_data:
                stringfound=True
        if stringfound:
            print('skipping for string scan')
            continue

        print (file_path, result)
        shutil.copy(file_path, QUALITY_PATH)
    sys.exit(0)
           
p=[0.09336338915014664, 0.7467402871108353, 0.09487947443039324, 0.09024963034733091, 0.47954483095032985, 0.16562219340802387, 0.12354303622108254, 0.4264934505645026, 0.2625761394210968, 0.6685184164613716, 0.9323433840490996, 0.08665733080118132, 0.6317928977869808, 0.47036462098446974, 0.0316981214129296, 0.9592358112708037, 0.21285285562423273]

target_data=[]
file_list=listdir(modelname+'/1/')
x_train=[]
my_sample_weight =[]
for file_path in file_list:
    with open(modelname+'/1/'+file_path, 'rb') as f:
        data = list(f.read(MAXLEN))
        x_train.append(data)
        target_data.append(1)
        my_sample_weight.append(20.)

file_list=listdir(modelname+'/0/')
for file_path in file_list:
    with open(modelname+'/0/'+file_path, 'rb') as f:
        data = list(f.read(MAXLEN))
        x_train.append(data)
        target_data.append(0)
        my_sample_weight.append(1.)

npx_train=np.array(x_train)
nptarget_data=np.array(target_data)
np_my_sample_weight=np.array(my_sample_weight)
bestp=p.copy()
print(p)
print(bestp)
bestacc=0
bestloss=9999
mutatep=False

while(1==1):
    earlystop_callback = EarlyStopping (monitor='val_weighted_binary_crossentropy', min_delta=0.0001, patience=2, verbose=1, restore_best_weights=True)
    p=bestp.copy()
    if mutatep==True:
        p[random.randint(0,16)]=random.random()
        p[random.randint(0,16)]=random.random()
    mutatep=True

    if p[5]<.25:
        p5="relu"
    elif p[5]<.5:
        p5="sigmoid"
    elif p[5]<.75:
        p5="tanh"
    else:
        p5="hard_sigmoid"

    if p[12]<.1:
        p12="glorot_normal"
    elif p[12]<.2:
        p12="glorot_uniform"
    elif p[12]<.3:
        p12="he_normal"
    elif p[12]<.4:
        p12="he_uniform"
    elif p[12]<.5:
        p12="lecun_normal"
    elif p[12]<.6:
        p12="lecun_uniform"
    elif p[12]<.7:
        p12="truncated_normal"
    elif p[12]<.8:
        p12="orthogonal"
    elif p[12]<.9:
        p12="random_normal"
    else:
        p12="random_uniform"

    if p[16]<.2:
        p16="sgd"
    elif p[16]<.4:
        p16="rmsprop"
    elif p[16]<.6:
        p16="adam"
    elif p[16]<.8:
        p16="adagrad"
    else:
        p16="nadam"

    print ("Features: ",int(p[0]*128)+32,"SpatialDrop: ",p[7]*.5,"Bi" if p[8]<.5 else "","GRU " if (p[8]<.75 and p[8]>.25) else "LSTM", "Size: ",int(p[1]*128)+32, "Drop: ",p[13]*.5, "RecDrop ",p[14]*.5)
    if p[9]>.5:
        print ("Conv1D Filt: ",int(p[10]*90)+8,"Pad: ",'same' if p[11]<.5 else 'valid',"Init: ",p12,"Size: ",int(p[15]*3)+2)

    if p[2]>.75:
        p2="Avg"
    elif p[2]>.50:
        p2="Max"
    elif p[2]>.25:
        p2="Both"
    else:
        p2="None"
    print("Pool: ",p2,"Dropout1: ",p[3]*.5,"Dense: ",int(p[4]*128)+32, "Activation: ", p5,"Dropout2: ",p[6]*.5,"Opt: ",p16)
    inp = Input(shape=(MAXLEN, )) #maxlen as defined earlier

    try:
        x = Embedding(256, int(p[0]*128)+32)(inp)
        x = SpatialDropout1D(p[7]*.5)(x)
        if p[8]>.75:
            x = LSTM(int(p[1]*128)+32, return_sequences=True, dropout=p[13]*.5, recurrent_dropout=p[14]*.5)(x)
        elif p[8]>.5:
            x = GRU(int(p[1]*128)+32, return_sequences=True, dropout=p[13]*.5, recurrent_dropout=p[14]*.5)(x)
        elif p[8]>.25:
            x = Bidirectional(GRU(int(p[1]*128)+32, return_sequences=True, dropout=p[13]*.5, recurrent_dropout=p[14]*.5))(x)
        else:
            x = Bidirectional(LSTM(int(p[1]*128)+32, return_sequences=True, dropout=p[13]*.5, recurrent_dropout=p[14]*.5))(x)

        if p[9]>.5:
            x = Conv1D(int(p[10]*90)+8, kernel_size=int(p[15]*3)+2, padding='same' if p[11]<.5 else 'valid', kernel_initializer = p12)(x)

        if p[2]>.75:
            x = GlobalAveragePooling1D()(x)
        elif p[2]>.50:
            x = GlobalMaxPool1D()(x)
        elif p[2]>.25:
            avg_pool = GlobalAveragePooling1D()(x)
            max_pool = GlobalMaxPool1D()(x)
            x = concatenate([avg_pool, max_pool]) 
        else:
            x = Flatten()(x)
        x = Dropout(p[3]*.5)(x)
        x = Dense(int(p[4]*128)+32, activation=p5)(x)
        x = Dropout(p[6]*.5)(x)
        x = Dense(1, activation="sigmoid")(x)
        model = Model(inputs=inp, outputs=x)
        model.compile(loss='binary_crossentropy',optimizer=p16,metrics=['accuracy'], weighted_metrics=['accuracy','binary_crossentropy'])

    except:
        print("!!!!!!!!!!!!!!!!!!!!!fail to compile!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        del x
        del inp
        del model
        clear_session()
        continue
    model.summary()
    history = model.fit(npx_train, nptarget_data, sample_weight=np_my_sample_weight, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True, callbacks=[earlystop_callback])
    print(history)
    val_acc = history.history['val_weighted_acc']
    val_loss = history.history['val_weighted_binary_crossentropy']

    print ("Cumulative:")
    print ("Accuracy Best:", bestacc) 
    print ("Loss Best:", bestloss)

    print ("This Round:")
    print ("Accuracy Final:", val_acc[-1])
    print ("Accuracy Best:", max(val_acc))
    print ("Loss Final:", val_loss[-1])
    print ("Loss Best:", min(val_loss))
    if min(val_loss)<bestloss:
        print("****************************************************** New Best ****************************************************")
        bestp=p.copy()
        file_object  = open("bestqualp.txt", "w") 
        file_object.write(str(bestp))
        file_object.close()

        model.save ('qual.h5')
        bestacc=max(val_acc)
        bestloss=min(val_loss)

    print(str(p))
    del x
    del inp
    del model
    clear_session()
