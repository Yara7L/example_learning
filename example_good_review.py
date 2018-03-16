# some import code, it is not completed.
import keras 
from keras import layers

model=keras.models.Sequential()
model.add(layers.LSTM(1024,input_shape=(60,95),return_sequence=True))
model.add(layers.LSTM(1024,input_shape=(60,95)))
model.add(layers.Dense(95,activation='softmax'))


chars=sorted(list(set(text)))
print('unique characters:',len(chars))
char_indices=dict((char,chars.index(char)) for char in chars)


def getDataFromChunk(txtChunk,maxlen=60,step=1):
    sentences=[]
    next_chars=[]
    for i in range(0,len(txtChunk)-maxlen,step):
        sentences.append(txtChunk[i:i+maxlen])
        next_chars.append(txtChunk[i+maxlen])
    print('nb sequences:',len(sentences))
    print('vectorization...')
    X=np.zeros((len(sentences),maxlen,len(chars)),dtype=np.bool)
    y=np.zeros((len(sentences),len(chars)),dtype=np.bool)
    for i,sentence in enumerate(sentences):
        for t,char in enumerate(sentence):
            X[i,t,char_indices=[char]]=1
            y[i,char_indices[next_chars[i]]]=1
return [X,y]


from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
filepath="Feb-22-all-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint=ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
reduce_lr=ReduceLROnPlateau(monitor='loss',factor=0.5,patience=1,min_lr=0.00001)
callbacks_list=[checkpoint,reduce_lr]


for iteration in range(1,20):
    print('Iteration',iteration)
    with open("") as f:
        for chunk in iter(lambda: f.read(90000),""):
            X,y=getDataFromChunk(chunk)
            model.fit(X,y,batch_size=128,epochs=1,callbacks=callbacks_list)

def sample(preds,temperaure=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperaure
    exp_preds=np.exp(preds)
    preds=exp_preds/np.sum(exp_preds)
    probas=np.random.multinomial(1,preds,1)
return np.argmax(probas)


for i in range(300):
    sampled=np.zeros(1,maxlen,len(chars))
    for t,char in enumerate(generated_text):
        sampled[0,t,char_indices[char]]=1
    preds=model.predict(sampled,verbose=0)[0]
    next_index=sample(preds,temperaure)
    next_char=chars[next_index]
    generated_text+=next_char
    generated_text=generated_text[1:]
    sys.stdout.write(next_char)
    sys.stdout.flush()
print(generated_text)