from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Bidirectional,LSTM,Reshape,CuDNNLSTM,BatchNormalization,Flatten,Dropout,Dense
from keras.layers import add
from keras.utils import plot_model
def resnet_model_dilation(i):
    ''' This model is build using keras module from the paper https://arxiv.org/pdf/1910.12590.pdf
    inputs are to be resized of 256,256*4,1 with dilation_rate
    output is the model
    '''



    input  = Input(shape = (256,256*4,1))

    s = (i,i)

    c1 = Conv2D(64, (7,7), padding='same',strides=2,activation='relu', dilation_rate=(s),kernel_initializer='he_normal')(input)
    b1 = BatchNormalization()(c1)

    c2 = conv1 = Conv2D(32, (3,3),strides=2, padding='same', dilation_rate=(s),kernel_initializer='he_normal')(input)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)

    c4 = conv1 = Conv2D(64, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a3)
    b4 = BatchNormalization()(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)


    #-----------------------------------------------layer 2----------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides=2, padding='same',activation='relu', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b1 = BatchNormalization()(c1)

    c2 = conv1 = Conv2D(64, (3,3),strides=2, padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(128, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(128, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a3)
    b4 = BatchNormalization()(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #----------------------------------------------layer 3------------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides = (1,2) ,padding='same',activation='relu', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b1 = BatchNormalization()(c1)

    c2 = conv1 = Conv2D(128, (3,3),strides = (1,2), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(128, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(128, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a3)
    b4 = BatchNormalization()(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 4---------------------------------------------------------------------------------

    c1 = Conv2D(64, (3,3),strides = (2,2) ,padding='same',activation='relu', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b1 = BatchNormalization()(c1)

    c2 = conv1 = Conv2D(128, (3,3),strides = (2,2), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(64, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a3)
    b4 = BatchNormalization()(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 5-----------------------------------------------------------------------------------
    c1 = Conv2D(32, (3,3),strides = (2,2) ,padding='same',activation='relu', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b1 = BatchNormalization()(c1)

    c2 = conv1 = Conv2D(64, (3,3),strides = (2,2), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(32, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a3)
    b4 = BatchNormalization()(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-----------------------------------------layer 6-------------------------------------------------------------------------
    c1 = Conv2D(16, (3,3),strides = (2,2) ,padding='same',activation='relu', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b1 = BatchNormalization()(c1)

    c2 = conv1 = Conv2D(32, (3,3),strides = (2,2), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a4)
    b2 = BatchNormalization()(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(32, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a2)
    b3 = BatchNormalization()(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(16, (3,3), padding='same', dilation_rate=(s),kernel_initializer='he_normal')(a3)
    b4 = BatchNormalization()(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    f = Flatten()(a4)
    f = Reshape((int(8192/4), 1))(f)

    # #-----------------------------------------layer7---------------------------------------------------------------------------
    bi1 = Bidirectional(CuDNNLSTM(512, return_sequences=True))(f)
    d1  = Dropout(0.2)(bi1)

    bi2 = Bidirectional(CuDNNLSTM(512))(d1)
    d2 = Dropout(0.4)(bi2)

    out = Dense(2,activation='softmax')(d2)

    # create model
    model = Model(inputs=input, outputs=out)
    return model
