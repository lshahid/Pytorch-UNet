from PIL import Image
import numpy as np
import imageio

man_img = Image.open("test3\\output3\\truth\\BPH009_Phase023_0057.gif").convert('L')
man_arr = np.array(man_img)

auto_img = Image.open("test3\\output3\\BPH009_Phase023_0057.gif").convert('L')
auto_arr = (np.array(auto_img) - 255) * -1
auto_arr = auto_arr.astype('uint8')

diff_arr = auto_arr - man_arr
print(np.max(diff_arr))
print(np.min(diff_arr))

imageio.imsave('BPH009_Phase023_0057_diff.gif', diff_arr)
