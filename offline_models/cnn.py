#encoding: utf-8
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import os,sys,psutil
import gc

tf.random.set_random_seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate
FEATURE_VECTOR_SHAPE = [1,110] # height, width

input_dir = '/train_dir/'
train_dir = input_dir + 'train/' #pass.1.txt
test_dir =  input_dir + 'test/'
# CATEGORIES = ['pass', 'crash','wrongcode']
CATEGORIES = ['pass', 'fail']

categories_dir={
    'pass':0,
    'fail':1,
    # 'crash':1,
    # 'wrongcode':2
}
# print(os.listdir(input_dir))
_index_in_epoch = 1

# mnist = input_data.read_data_sets('./mnist', one_hot=True)  # they has been normalized to range (0,1)
# test_x = mnist.test.images[:2000]
# test_y = mnist.test.labels[:2000]

def exccmd2string(cmd):
    p = os.popen(cmd, "r")
    rs = []
    line = ""
    while True:
        line = p.readline()
        if not line:
            break
        rs.append(line)
    ss = ''
    for item in rs:
        ss += item
    gc.collect()  # free mem
    return ss

"""input@param is direct output arr from csmith
return@param feature vector"""
def feature2vector(feature):
    # print("feature is")
    # print(feature)
    # print(len(feature))
    feature_vector = []
    for index in range(len(feature) - 1):
        # print(feature[index])
        # print(feature[index].split(" ")[-1])
        feature_vector.append(int(feature[index].split(" ")[-1]))
    feature_vector = np.array(feature_vector)
    fv_normed = feature_vector / feature_vector.max(axis=0)  # nomalization
    return fv_normed.tolist()

def exccmd2arr(cmd):
    p = os.popen(cmd, "r")
    rs = []
    line = ""
    while True:
        line = p.readline()
        if not line:
            break
        rs.append(line)
    gc.collect() # free mem
    return rs

def test(size):
    test_x = []
    test_y = []
    perm = range(1,size+1)
    for i in perm:
        cmd_feature = "ls " + test_dir + " | grep \'\\." + str(i) + "\\.\' | xargs -I file cat "+ test_dir+"file"
        cmd_label = "ls " + test_dir + " | grep \'\\." + str(i) + "\\.\' "
        # print(cmd_feature)
        feature = exccmd2arr(cmd_feature)
        fv = feature2vector(feature)
        fv = np.array(fv)
        # fv = fv[np.newaxis, :]
        test_x.append(fv)

        label = exccmd2string(cmd_label).split('.')[0]
        # label_onehot = [0,0,0]
        if label != 'pass':
            label = 'fail'
        label_onehot = [0,0]

        label_onehot[categories_dir[label]] = 1
        label_onehot = np.array(label_onehot)
        # label_onehot = label_onehot[np.newaxis, :]
        test_y.append(label_onehot)

    '''extra data to reach 9/1'''
    perm = range(18001, 18258)
    for i in perm:
        cmd_feature = "ls " + train_dir + " | grep \'\\." + str(i) + "\\.\' | xargs -I file cat "+ train_dir+"file"
        cmd_label = "ls " + train_dir + " | grep \'\\." + str(i) + "\\.\' "
        # print(cmd_feature)
        feature = exccmd2arr(cmd_feature)
        fv = feature2vector(feature)
        fv = np.array(fv)
        # fv = fv[np.newaxis, :]
        test_x.append(fv)

        label = exccmd2string(cmd_label).split('.')[0]
        # label_onehot = [0,0,0]
        if label != 'pass':
            label = 'fail'
        label_onehot = [0,0]

        label_onehot[categories_dir[label]] = 1
        label_onehot = np.array(label_onehot)
        # label_onehot = label_onehot[np.newaxis, :]
        test_y.append(label_onehot)

    return test_x, test_y # testy are int

test_x ,test_y = test(1743)

"""
pass.1.txt, wrongcode.223.txt, crash.4.txt
"""
def next_batch(_index_in_epoch, batch_size, shuffle=True):
    feature_vectors = []
    labels = []
    perm = np.arange(_index_in_epoch, _index_in_epoch +batch_size/2)
    perm += np.arange(_index_in_epoch+10000, _index_in_epoch+10000 +batch_size/2)

    if shuffle:
        np.random.shuffle(perm)
    res = open('train_status.txt', 'a')
    res.flush()
    res.write('epoch:' +str(_index_in_epoch)+'\n')
    for i in perm.tolist():
        cmd_feature = "ls "+train_dir+" | grep \'\\."+str(i)+"\\.\' | xargs -I file cat "+ train_dir+"file"
        cmd_label = "ls "+train_dir+" | grep \'\\."+str(i)+"\\.\' "
        # print(cmd_feature)
        # print(cmd_label)
        feature = exccmd2arr(cmd_feature)
        # print(feature)
        # print(len(feature))
        fv = feature2vector(feature)
        if len(fv)==0:
            continue
        fv = np.array(fv)
        fv = fv[np.newaxis, :]
        feature_vectors.append(fv)

        label = exccmd2string(cmd_label).split('.')[0]
        # label_onehot = [0, 0, 0]
        if label != 'pass':
            label = 'fail'
        label_onehot = [0,0]
        label_onehot[categories_dir[label]] = 1
        label_onehot = np.array(label_onehot)
        label_onehot = label_onehot[np.newaxis, :]
        labels.append(label_onehot)
        res.write('id:'+ str(i)+', category:'+label+'\n')
        print('id:'+ str(i)+', category:'+label)
    res.close()
    return feature_vectors, labels #labels are int


# # plot one example
# print(mnist.train.images.shape)     # (55000, 28 * 28)
# print(mnist.train.labels.shape)   # (55000, 10)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0])); plt.show()

tf_x = tf.placeholder(tf.float32, [None, FEATURE_VECTOR_SHAPE[0]*FEATURE_VECTOR_SHAPE[1] ])
fv = tf.reshape(tf_x, [-1, FEATURE_VECTOR_SHAPE[0], FEATURE_VECTOR_SHAPE[1], 1])              # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, len(CATEGORIES)])            # output y

# CNN
conv1 = tf.layers.conv2d(   # shape (1, 110, 1)
    inputs=fv,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (1, 110, 16)
# pool1 = tf.layers.max_pooling2d(
#     conv1,
#     pool_size=2,
#     strides=2,
# )           # -> (14, 14, 16)
# conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
conv2 = tf.layers.conv2d(conv1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (1, 110, 32)
# pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
# flat = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )
flat = tf.reshape(conv2, [-1, FEATURE_VECTOR_SHAPE[0]*FEATURE_VECTOR_SHAPE[1]*32])          # -> (7*7*32, )

output = tf.layers.dense(flat, len(CATEGORIES))              # output layer

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph

# following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
#
# plt.ion()
teststep=0
for step in range(17975):
    # b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    print(_index_in_epoch)
    b_x, b_y = next_batch(_index_in_epoch, BATCH_SIZE)
    _index_in_epoch += BATCH_SIZE/2
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x[0], tf_y: b_y[0]})
    res = open('train_status.txt', 'a')
    res.flush()
    res.write("loss:" + str(loss_))
    res.close()
    print("loss:" + str(loss_))
    if step % 50 == 0:
        teststep +=1
        accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x[:teststep], tf_y: test_y[:teststep]})
        print('Step:'+ str(step)+ '| train loss: %.4f' + str( loss_)+ '| test accuracy: %.2f'+ str(accuracy_))
        res = open('train_status.txt', 'a')
        res.flush()
        res.write('Step:'+ str(step)+'| train loss: %.4f' +str(loss_)+ '| test accuracy: %.2f' +str( accuracy_))
        res.close()
        # if HAS_SK:
        #     # Visualization of trained flatten layer (T-SNE)
        #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
        #     low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
        #     labels = np.argmax(test_y[:teststep], axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
# plt.ioff()

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')

res = open('train_status.txt', 'a')
res.flush()
res.write(str(pred_y)+ 'prediction number\n')
res.write(str(np.argmax(test_y[:10], 1))+ 'real number')
res.close()


sess.close()