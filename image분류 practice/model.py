import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_horse_dir = '/Users/ibyeong-gwon/Desktop/datasets/horse-or-human/horses'
train_human_dir = '/Users/ibyeong-gwon/Desktop/datasets/horse-or-human/humans'

# horses 파일 이름 리스트
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

# humans 파일 이름 리스트
train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

# horses/humans 총 이미지 파일 개수
# print('total training horse images:', len(os.listdir(train_horse_dir)))
# print('total training human images:', len(os.listdir(train_human_dir)))

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off')

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()