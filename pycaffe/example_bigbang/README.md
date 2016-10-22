# About it

这个栗子主要作为一个实现，将数据置于data文件夹下即可，格式为图片原始数据和图片的list文件，当然这个栗子中设计到
语音与图片数据的特征融合，因此想要使用该数据，需要实现mfcc特征的获取，可以使用如下代码。　　

```python

def mfccs(y, sr):
    '''
    calculate mfcc \delta mfcc, \delta2 mfcc
    :param y: audio data
    :param sr: sample rate
    :return: [mfcc, delta_mfcc, delta_delta_mfcc]
    '''
    melspectrogram = librosa.feature.melspectrogram(y, sr=sr, n_fft=0.02*sr, hop_length=0.01*sr)
    mfccs = librosa.feature.mfcc(sr=sr, S=melspectrogram, n_mfcc=25)
    # delta mfcc
    delta_mfccs = np.zeros(mfccs.shape)
    for i in range(2, mfccs.shape[0]-2):
        delta_mfccs[i, :] = -2 * mfccs[i-2, :] - mfccs[i-1, :] + mfccs[i+1, :] + 2*mfccs[i+2, :]

    delta_mfccs /= 3.

    # delta_delta_mfccs
    delta_delta_mfccs = np.zeros(delta_mfccs.shape)
    for i in range(2, mfccs.shape[0]-2):
        delta_delta_mfccs[i, :] = -2 * delta_mfccs[i-2, :] - delta_mfccs[i-1, :] + delta_mfccs[i+1, :] + 2 * delta_mfccs[i+2, :]
    delta_delta_mfccs /= 3.
    all_mfccs = np.concatenate([mfccs, delta_mfccs, delta_delta_mfccs], axis=0)
    return all_mfccs
```　　

## 创新性　　

主要在于融合特征与conv层而非fc层，acc可以达到99%+。
