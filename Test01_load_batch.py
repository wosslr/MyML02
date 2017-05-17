import pickle
import scipy
import scipy.misc
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def transf_data_to_rcd(data):
    lv_d_arr = []
    # transform 1d to 2d with rgb mode per element
    data_index = 0;
    loop_len = len(data) / 3
    print("retrieve deepth...")
    while data_index < loop_len:
        lv_arr = []
        lv_arr.append(data[data_index])
        lv_arr.append(data[data_index + 1024])
        lv_arr.append(data[data_index + 1024 + 1024])
        lv_d_arr.append(lv_arr)
        data_index += 1

    data_index = 0
    lv_rcd_arr = []
    while data_index < 32:
        lv_arr = lv_d_arr[data_index * 32 : (data_index+1) * 32]
        lv_rcd_arr.append(lv_arr)
        data_index += 1
    return lv_rcd_arr


def save_data_to_pkl(file_name, X, Y):
    dict = unpickle(file_name)
    for data in dict[b'data']:
        img_arr = transf_data_to_rcd(data)
        X.append(img_arr)
        print("process " + file_name + " X len:")
        print(len(X))
    Y = Y + dict[b'labels']


# dict1 = unpickle("data_batch_1")
# dict2 = unpickle("data_batch_2")
# dict3 = unpickle("data_batch_3")
# dict4 = unpickle("data_batch_4")
# dict5 = unpickle("data_batch_5")
#
# dict_test = unpickle("test_batch")
#
# X_batches = []
# X_batches.append(dict1[b'data'])
# X_batches.append(dict2[b'data'])
# X_batches.append(dict3[b'data'])
# X_batches.append(dict4[b'data'])
# X_batches.append(dict5[b'data'])
# Y = dict1[b'labels'] + dict2[b'labels'] + dict3[b'labels'] + dict4[b'labels'] + dict5[b'labels']
# b'data'
# b'labels'
# b'filenames'
# b'batch_label'

# print(len(dict1[b'data']))
# print(dict1[b'data'])

# img_index = 0
# for data in dict1[b'data']:
#     img_arr = transf_data_to_rcd(data)
#     tmp_img = scipy.misc.toimage(img_arr)
#     img_index += 1
#     filename = 'images/image' + str(img_index) + '.jpg'
#     print("saving " + filename)
#     tmp_img.save(filename)

# img_index = 0
# X = []
# for batch in X_batches:
#     index = 0
#     for data in batch:
#         img_arr = transf_data_to_rcd(data)
#         X.append(img_arr)
#         np.delete(batch, index)
#         index += 1
#         print("X len:")
#         print(len(X))
#
# X_test_data = dict_test[b'data']
# Y_test = dict_test[b'labels']

# X_test = []
# for data in X_test_data:
#     img_arr = transf_data_to_rcd(data)
#     X_test.append(img_arr)
#     print("X test len:")
#     print(len(X_test))

X = []
Y = []
X_test = []
Y_test = []

save_data_to_pkl("data_batch_1", X, Y)
save_data_to_pkl("data_batch_2", X, Y)
save_data_to_pkl("data_batch_3", X, Y)
save_data_to_pkl("data_batch_4", X, Y)
save_data_to_pkl("data_batch_5", X, Y)

save_data_to_pkl("test_batch", X_test, Y_test)

pickle.dump((X, Y, X_test, Y_test), open("full_dataset.pkl", "xb"))