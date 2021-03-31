from keras.applications import inception_v3
from keras import backend as K

#--0 補助関数の作成

import scipy
from keras.preprocessing import image

#画像サイズを変更する関数
def resize_img(img, size):
    img = np.copy(img)
    factors = (1,
               float(size[0]) / img.shape[1],
               float(size[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)

#画像を保存する関数
def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


#画像を開いてサイズ変更し、InceptionV3が処理できるテンソルに変換する関数
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

#テンソルを有効な画像に変換
def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#--1 学習済みのInceptionV3モデルを読み込む

#訓練関連の演算をすべて無効にする
K.set_learning_phase(0)

#InceptionV3モデルを畳み込みベース無しで構築
#ImageNetで学習した時の重みを読み込む
model = inception_v3.InceptionV3(weights='imagenet',include_top=False)

#--2 DeepDreamの構成
#層の名前を係数にマッピングする辞書を定義
#この係数は、最大化する損失値に各層がどのくらい貢献するかの係数
#名前はモデルで定義されている　他の層に係数を付けたい場合は　model.summary()で確認されたし
layer_contributions = {
    'mixed2': 0.2,
    'mixed3': 3.,
    'mixed4': 2.,
    'mixed5': 1.5,
}

#--3 最大化の対象となる損失値を定義
#損失値を保持するテンソルを定義

#層の名前を層のインスタンスにマッピングする辞書を作成
#layer_dictにモデルの層の名前が登録される
layer_dict = dict([(layer.name, layer) for layer in model.layers])

#損失値の定義
loss = K.variable(0.)

#係数を設定した層の数分繰り返す
for layer_name in layer_contributions:
    #名前から係数を取り出して　coeffとして係数を定義する。
    coeff = layer_contributions[layer_name]

    #layer_dictに入っている　層の出力値を取得
    activation = layer_dict[layer_name].output

    #層の出力値をスケーリングして扱えるようにする
    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    #層の特徴量であるL2ノルムをlossに加算
    #非境界ピクセルのみをlossに適用することで、周辺効果を回避
    loss += coeff * K.sum(K.square(activation[:, 2: -2, 2: -2, :])) / scaling

#--４　勾配上昇法のプロセスを設定
#生成された画像を保持するテンソル モデルの出力値を記憶
dream = model.input

#損失関数の勾配を計算
grads = K.gradients(loss, dream)[0]

#勾配を正規化する（重要）
grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

#入力画像(dream)に基づいて損失と勾配の値(fetch_loss_and_grads)を取得するKras関数を定義
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

#勾配上昇法を指定された回数(iterations)にわたって実行する関数
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

#--5 異なる尺度にわたって勾配上昇法を実行
import numpy as np

#以下はハイパーパラメータで、変更することで新しい効果が得られる
step = 0.01         # 勾配上昇法のステップサイズ
num_octave = 3      # 勾配上昇法を実行する尺度の数（オクターブ）
octave_scale = 1.4  # 尺度感の拡大率(初期値として40%を設定)
iterations = 20     # 尺度ごとの上昇ステップの数

#損失値が１０を超えた場合は見た目が酷くなるため勾配上昇法を中止
max_loss = 10.

# 使用したい画像のパスを表示
base_image_path = './input/image.jpg'

# パスを使って画像を獲得し、NumPy配列にする
img = preprocess_image(base_image_path)

#　勾配上昇法を実行する様々な尺度を定義する形状タプルのリストを準備

original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)

# 形状リストを逆にして昇順になるようにする
successive_shapes = successive_shapes[::-1]

# 画像のNumPy配列のサイズを最も小さな尺度に変換
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    #ドリーム画像をリサイズによって拡大
    img = resize_img(img, shape)

    #勾配上昇法を実行(gradient_ascent())してドリーム画像を加工
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    
    #元の画像を縮小したものを拡大：画像が画素化される
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    #元の画像の高品質バージョンを計算
    same_size_original = resize_img(original_img, shape)

    #失われたディティールをドリーム画像に再注入
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    save_img(img, fname='./output/dream_at_scale_' + str(shape) + '.png')

save_img(img, fname='./output/final_dream.png')