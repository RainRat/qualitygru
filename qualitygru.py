import os
import sys
import shutil
import numpy as np
import tensorflow as tf
import argparse
import logging
import configparser

from tensorflow.keras.layers import (
    Dense, Input, LSTM, Embedding, Dropout, 
    Activation, GRU, Bidirectional, GlobalMaxPool1D, 
    GlobalAveragePooling1D, SpatialDropout1D, Conv1D, Flatten, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session

from two_byte_headers import DISALLOW_LIST as TWO_BYTE_LIST
from variable_headers import DISALLOW_LIST as HEADER_LIST
from body_strings import DISALLOW_LIST as BODY_LIST

try:
    from optional_two_byte_headers import DISALLOW_LIST as OPTIONAL_TWO_BYTE_LIST
except ImportError:
    OPTIONAL_TWO_BYTE_LIST = []
try:
    from optional_variable_headers import DISALLOW_LIST as OPTIONAL_HEADER_LIST
except ImportError:
    OPTIONAL_HEADER_LIST = []
try:
    from optional_body_strings import DISALLOW_LIST as OPTIONAL_BODY_LIST
except ImportError:
    OPTIONAL_BODY_LIST = []

TWO_BYTE_LIST = TWO_BYTE_LIST + OPTIONAL_TWO_BYTE_LIST
HEADER_LIST = HEADER_LIST + OPTIONAL_HEADER_LIST
BODY_LIST = BODY_LIST + OPTIONAL_BODY_LIST

MODEL_NAME = 'quality'

config = load_config()

def get_all_files_recursive(root):
    return [os.path.join(dirpath, filename) 
            for dirpath, dirnames, filenames in os.walk(root) 
            for filename in filenames]

def parse_arguments():
    parser = argparse.ArgumentParser(description='QualityGRU')
    parser.add_argument('--mode', type=str, default='predict', choices=['train', 'predict'], help='Mode of operation: train or predict')
    parser.add_argument('--unsorted_path', type=str, default='', help='Path to unsorted data')
    parser.add_argument('--quality_path', type=str, default='', help='Path to store quality data')
    return parser.parse_args()

def load_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return {
        'MAX_LEN': config.getint('DEFAULT', 'max_len', fallback=300),
        'BATCH_SIZE': config.getint('DEFAULT', 'batch_size', fallback=100),
        'EPOCHS': config.getint('DEFAULT', 'epochs', fallback=100)
    }

def predict_files(config):
    modelqual = tf.keras.models.load_model('qual.h5')
    file_list=get_all_files_recursive(UNSORTED_PATH)
    logger.info('enumeration done')
    for file_path in file_list:
        logger.info(f"Processing file: {file_path}")
        filesize=os.path.getsize(file_path)
        # Filter by file size
        if filesize<=2048 or filesize>=2048*1024:
            continue
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        two_bytes = raw_data[:2]
        if two_bytes in TWO_BYTE_LIST:
            print('skipping for 2-byte header')
            continue
            
        bytesfound=False
        for byte_identifier in HEADER_LIST:
            if raw_data[:len(byte_identifier)] == byte_identifier:
                bytesfound=True
        if bytesfound:
            print('skipping for larger header')
            continue

        np_data=np.expand_dims(np.array(list(raw_data[:MAX_LEN])), axis=0)
        result=modelqual.predict(np_data, batch_size=1, steps=1)[0]
        if result<.5:
            print('skipping for uninteresting')
            continue

        string_data=raw_data.decode('latin1').lower()
        stringfound=False 
        for string_identifier in BODY_LIST:
            if string_identifier in string_data:
                stringfound=True
        if stringfound:
            print('skipping for string scan')
            continue

        print (file_path, result)
        shutil.copy(file_path, QUALITY_PATH)

def train_model(config):
    p=[0.09336338915014664, 0.7467402871108353, 0.09487947443039324, 0.09024963034733091, 0.47954483095032985, 0.16562219340802387, 0.12354303622108254, 0.4264934505645026, 0.2625761394210968, 0.6685184164613716, 0.9323433840490996, 0.08665733080118132, 0.6317928977869808, 0.47036462098446974, 0.0316981214129296, 0.9592358112708037, 0.21285285562423273]

    target_data=[]
    file_list=listdir(MODEL_NAME+'/1/')
    x_train=[]
    my_sample_weight =[]
    for file_path in file_list:
        with open(MODEL_NAME+'/1/'+file_path, 'rb') as f:
            data = list(f.read(MAX_LEN))
            x_train.append(data)
            target_data.append(1)
            my_sample_weight.append(20.)

    file_list=listdir(MODEL_NAME+'/0/')
    for file_path in file_list:
        with open(MODEL_NAME+'/0/'+file_path, 'rb') as f:
            data = list(f.read(MAX_LEN))
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

    while True:
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

        print(f"Features: {int(p[0]*128)+32} SpatialDrop: {p[7]*.5} {'Bi' if p[8]<.5 else ''} {'GRU' if (p[8]<.75 and p[8]>.25) else 'LSTM'} Size: {int(p[1]*128)+32} Drop: {p[13]*.5} RecDrop: {p[14]*.5}")
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
        inp = Input(shape=(MAX_LEN, )) #maxlen as defined earlier

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

        except Exception as e:
            print(f"Failed to compile due to: {e}")
            del x
            del inp
            del model
            clear_session()
            continue
        model.summary()
        history = model.fit(npx_train, nptarget_data, sample_weight=np_my_sample_weight, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, shuffle=True, callbacks=[earlystop_callback])
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
            print(("*" * 50) + " New Best "+ ("*" * 50))
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

def main():
    args = parse_arguments()
    MODE = args.mode
    UNSORTED_PATH = args.unsorted_path
    QUALITY_PATH = args.quality_path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if MODE == 'predict':
        predict_files(config)
    elif MODE == 'train':
        train_model(config)

if __name__ == "__main__":
    main()